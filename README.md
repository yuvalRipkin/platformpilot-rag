# platformpilot-rag

The retrieval-augmented generation service for PlatformPilot — indexes platform knowledge and serves grounded answers to the rest of the system. Sibling repos: `platformpilot-infra`, `platformpilot-operator`, `platformpilot-integration-agent`, `platformpilot-manifests`. Phase 1, in progress.

## Local development

### Prerequisites

- [Docker](https://www.docker.com/) (for the local Postgres + pgvector container)
- [uv](https://docs.astral.sh/uv/) (Python package + project manager)

### Setup

```bash
cp .env.example .env
docker compose up -d
uv sync
make migrate     # apply schema migrations
make run
```

The API is then served at `http://localhost:8000`.

On startup the service loads the `all-MiniLM-L6-v2` embedder model (~90 MB) into the FastAPI app state. Expect 5–15s of startup time; `/ready` will not return 200 until both the database is reachable **and** the lifespan has finished loading the model, so readiness inherently covers "model loaded" too.

### Endpoints

| Method | Path       | Purpose                                                                                            |
|--------|------------|----------------------------------------------------------------------------------------------------|
| GET    | `/health`  | Liveness — always 200                                                                              |
| GET    | `/ready`   | Readiness — 200 only when the DB is reachable AND the embedder has loaded (503 otherwise)          |
| POST   | `/ingest`  | Ingest a markdown document: chunk, embed, persist                                                  |
| POST   | `/search`  | Vector retrieval — returns top-K chunks for a query. No LLM call.                                  |
| POST   | `/query`   | Full RAG path — retrieves chunks then asks Claude for a grounded, cited answer.                    |
| GET    | `/metrics` | Prometheus metrics (request counters, retrieval/LLM histograms, FastAPI default instrumentation).  |

`/search` is the fast path (~50 ms) for debugging retrieval quality on its own. `/query` composes `/search` with the LLM and adds a `~1–2 s` Anthropic round-trip on top.

Examples:

```bash
# Ingest
curl -X POST http://localhost:8000/ingest \
  -H 'content-type: application/json' \
  -d '{
    "source": "platformpilot-operator/README.md",
    "title": "Operator README",
    "content": "# Operator\n\nWhat the operator does..."
  }'
# -> {"document_id":"...","chunks_created":3,"is_replacement":false}

# Search
curl -X POST http://localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "how does the operator handle failures?"}'
# -> {"query_id":"...","chunks":[{"chunk_id":"...","source":"...","similarity":0.81,"text":"..."}, ...],"latency_ms":42}

# Query
curl -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query": "how does the operator handle failures?"}'

# Metrics (Prometheus exposition format)
curl http://localhost:8000/metrics | grep '^rag_'
```

Example `/query` response:

```json
{
  "query_id": "9f1b2a44-4f25-4cf8-9b1e-3f5b9c7d8e10",
  "answer": "The operator retries transient errors with exponential backoff and surfaces permanent errors to the alert pipeline, as described in [1].",
  "chunks": [
    {
      "chunk_id": "0514d0f3-8e93-4e59-903e-b39cfd3d32ea",
      "document_id": "a0537927-7827-463a-8276-134b80fd2e92",
      "source": "platformpilot-operator/README.md",
      "chunk_index": 0,
      "text": "The operator handles failures by retrying transient errors with exponential backoff. Permanent errors are surfaced to the alert pipeline...",
      "similarity": 0.58
    }
  ],
  "latency_ms": 1820
}
```

If retrieval finds no chunks above the similarity threshold, `answer` is the fixed fallback `"I don't have that information in the indexed documents."` and `chunks` is `[]` — no LLM call is made.

Re-ingesting the same `source` replaces its chunks (`is_replacement: true`). Both `/search` and `/query` return a server-generated `query_id` — grep service logs by that id when debugging a user-reported issue.

### Configuration

These are read from environment (or `.env` via pydantic-settings):

| Variable                | Default              | Meaning                                              |
|-------------------------|----------------------|------------------------------------------------------|
| `DATABASE_URL`          | _(required)_         | asyncpg DSN, e.g. `postgresql+asyncpg://...`         |
| `ANTHROPIC_API_KEY`     | _(required)_         | Claude API key. Used by `/query`.                    |
| `ANTHROPIC_MODEL`       | `claude-sonnet-4-6`  | Model name for `/query`.                             |
| `TOP_K`                 | `4`                  | Default `k` for `/search` and `/query` retrieval.    |
| `EMBEDDER_MAX_WORKERS`  | _(derived)_          | Threads serving embedder inference. Derived as `clamp(cpus, 1, 8)`, where `cpus` is the lower of the cgroup CPU quota (`limits.cpu`) and the scheduler affinity mask. The resolved value and which source won are logged at startup as `embed_workers` / `workers_source`. |
| `EMBEDDER_TORCH_THREADS`| `1`                  | torch intra-op threads per inference. Keep at 1 so `EMBEDDER_MAX_WORKERS` bounds CPU use truthfully. |
| `SIMILARITY_THRESHOLD`  | `0.5`                | Minimum cosine similarity for a chunk to be kept.    |
| `MAX_CONTEXT_TOKENS`    | `8000`               | Hard cap on the LLM user prompt's token count.       |
| `LLM_MAX_TOKENS`        | `1024`               | Anthropic `max_tokens` on each `/query` call.        |
| `LLM_TEMPERATURE`       | `0.0`                | Deterministic by default — RAG wants reproducibility.|

### Concurrency

`Embedder.encode()` is synchronous, CPU-bound torch inference, and it was called inline from two async paths (`Retriever.retrieve` and the `/ingest` handler). That blocked the event loop for the whole process: concurrent requests were served one at a time, and everything else on the loop — other requests' DB I/O, `/query`'s Anthropic round-trips, the health probes — waited too. Inference now runs on a dedicated bounded thread pool, sized from the cgroup CPU quota, with each worker pinned to one torch intra-op thread.

Two changes were needed, and it is worth separating them. Moving the call off the loop is what fixes responsiveness. Bounding the pool and pinning torch are what stop the fix from replacing serialization with oversubscription: `asyncio.to_thread` submits to the loop's default executor (`min(32, cpu_count + 4)` — 12 here, 32 on a 64-core node), and each of those inferences would otherwise fan out across every core, so a burst of requests can put ~100 runnable threads on a 4-core quota.

Measured through `POST /search` with a fake embedder calibrated to ~50 ms of single-threaded work, inside CPU-limited containers. `cores` is `process_time / wall`, i.e. the mean number of cores kept busy; `lag` is how late a 5 ms heartbeat coroutine fired, i.e. how blocked the loop was.

**50 concurrent requests, CPU-bound fake**

| quota | config | wall (median of 5) | cores | CPU used vs. minimum | max loop lag |
|---|---|---|---|---|---|
| `--cpus=2` | blocking (pre-fix) | 4.967s | 2.0 | 3.65x | 4182ms |
| `--cpus=2` | unbounded `to_thread` | 3.675s | 2.0 | 2.71x | 198ms |
| `--cpus=2` | unbounded + pinned torch | 3.170s | 2.0 | 2.33x | 85ms |
| `--cpus=2` | **bounded pool + pinned** | **2.572s** | 1.0 | **0.95x** | **55ms** |
| `--cpus=4` | blocking (pre-fix) | 1.556s | 4.0 | 2.49x | 1248ms |
| `--cpus=4` | unbounded `to_thread` | 1.191s | 4.1 | 1.91x | 56ms |
| `--cpus=4` | unbounded + pinned torch | 1.105s | 4.1 | 1.79x | 58ms |
| `--cpus=4` | **bounded pool + pinned** | **0.954s** | 2.9 | **1.11x** | **27ms** |

"CPU used vs. minimum" is measured core-seconds over the 2.5–2.7 core-seconds the same work costs run serially. The bounded pool lands within ~10% of that floor; the unbounded configurations burn roughly twice it and are still slower in wall time, because the extra threads spend the quota on context switching, cache thrash and CFS throttle stalls rather than on inference.

The pool runs one worker per available core. An earlier version reserved one for the event loop; that reservation was measured rather than assumed, and it lost — 19–49% less throughput for identical loop lag (37ms vs 37ms at `--cpus=2`, 35ms vs 36ms at `--cpus=4`, both at N=50), so it was removed. Reproduce with `--cell cpu:bounded --workers N`.

The tradeoff is real and goes the other way at low concurrency. Pinning torch to one intra-op thread means a single request no longer uses every core: at `--cpus=4`, N=1 costs **0.019s before and 0.051s after** (2.7x worse). At `--cpus=2` there is no such penalty (0.081s → 0.051s) — eight torch threads on two cores of quota was already a net loss. We accept worse latency on an idle service in exchange for bounded, predictable behaviour under load, which is the regime that matters when several pods share a node.

One caveat on reading these numbers: a `time.sleep()` fake consumes no CPU, so under it threads never contend and the bounded pool is pure queueing overhead — at `--cpus=2`/N=50 it measures 2.708s against unbounded's 0.289s, which would argue for reverting this change. Only the CPU-bound fake tests the hypothesis. The sleep fake is still the clearest demonstration of the original bug, though: blocking at N=50 stalls the event loop for **2194ms**.

Reproduce (Docker required; the matrix is meaningless on a laptop, where core count, BLAS kernel and background load all differ from the deployment):

```bash
make bench-concurrency CPUS=4   # full matrix: 2 fakes x 4 configs x N in {1,10,50}
uv run pytest -m concurrency    # fast assertions only, no container
```

### Migrations

Migrations live in `migrations/versions/`. After changing models, create a migration with `make migration name='describe change'` and edit the generated file.

```bash
make migrate         # apply all pending migrations
make migrate-down    # roll back one migration
```

### Tests

```bash
uv run pytest   # unit tests (fast, no DB)
```

Unit tests use mocked database sessions — no Postgres container required. Integration tests are skipped by default.

#### Integration tests

```bash
uv run pytest -m integration
```

Prerequisites:
- The Postgres container from `docker compose up -d` is running
- The configured user (`POSTGRES_USER`, default `rag`) has the `CREATEDB` privilege — the default `pgvector/pgvector:pg16` superuser created by the compose file already does

Lifecycle: a fresh `ragdb_test` database (the value of `DATABASE_URL`'s database, suffixed `_test`) is dropped if present and recreated at the start of the session, has all migrations applied via Alembic, and is dropped again at the end of the session — even if migrations fail. Each test runs inside a connection-level transaction that rolls back on teardown so individual tests don't pollute each other.
