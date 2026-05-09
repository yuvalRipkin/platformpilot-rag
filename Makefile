.PHONY: install run test lint format migrate migrate-down migration

install:
	uv sync

run:
	uv run uvicorn app.main:app --reload

test:
	uv run pytest

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
