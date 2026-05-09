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
uv run uvicorn app.main:app --reload
```

The API is then served at `http://localhost:8000`. Probes:

- `GET /health` — liveness, returns 200 unconditionally
- `GET /ready` — readiness, returns 200 only when the database is reachable (503 otherwise)

### Tests

```bash
uv run pytest
```

Tests use mocked database sessions — no Postgres container required.
