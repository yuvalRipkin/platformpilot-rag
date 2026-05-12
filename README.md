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
| `SIMILARITY_THRESHOLD`  | `0.5`                | Minimum cosine similarity for a chunk to be kept.    |
| `MAX_CONTEXT_TOKENS`    | `8000`               | Hard cap on the LLM user prompt's token count.       |
| `LLM_MAX_TOKENS`        | `1024`               | Anthropic `max_tokens` on each `/query` call.        |
| `LLM_TEMPERATURE`       | `0.0`                | Deterministic by default — RAG wants reproducibility.|

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
