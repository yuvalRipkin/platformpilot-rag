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

| Method | Path      | Purpose                                                       |
|--------|-----------|---------------------------------------------------------------|
| GET    | `/health` | Liveness — always 200                                         |
| GET    | `/ready`  | Readiness — 200 only when the DB is reachable (503 otherwise) |
| POST   | `/ingest` | Ingest a markdown document: chunk, embed, persist             |

Example:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'content-type: application/json' \
  -d '{
    "source": "platformpilot-operator/README.md",
    "title": "Operator README",
    "content": "# Operator\n\nWhat the operator does..."
  }'
# -> {"document_id":"...","chunks_created":3,"is_replacement":false}
```

Re-ingesting the same `source` replaces its chunks (`is_replacement: true`).

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
