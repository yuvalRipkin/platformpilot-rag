.PHONY: install run test test-concurrency bench-concurrency lint format migrate migrate-down migration

install:
	uv sync

run:
	uv run uvicorn app.main:app --reload

test:
	uv run pytest

test-concurrency: ## Wall-clock tests proving the event loop is not blocked
	uv run pytest -m concurrency

CPUS ?= 4
bench-concurrency: ## Full concurrency matrix, in a CPU-limited container
	docker build -f Dockerfile.bench -t rag-bench .
	docker run --rm --cpus=$(CPUS) -v "$(PWD):/src:ro" -w /src \
		-e DATABASE_URL='postgresql+asyncpg://u:p@localhost:5432/db' \
		-e ANTHROPIC_API_KEY=dummy -e PYTHONPATH=/src \
		rag-bench python -m tests.concurrency_harness --label "--cpus=$(CPUS)"

lint:
	uv run ruff check .

format:
	uv run ruff format .

migrate: ## Apply pending migrations
	uv run alembic upgrade head

migrate-down: ## Roll back one migration
	uv run alembic downgrade -1

migration: ## Create a new empty migration (usage: make migration name="add foo")
	uv run alembic revision -m "$(name)"
