import os

# Settings() requires ANTHROPIC_API_KEY at import time. Provide a placeholder
# that won't reach the real Anthropic API — every test that touches the LLM
# path injects a fake LLMClient via app.dependency_overrides. setdefault
# preserves an explicit ANTHROPIC_API_KEY=... env if the operator sets one.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-placeholder")

import asyncio  # noqa: E402
from collections.abc import AsyncIterator, Iterator  # noqa: E402

import asyncpg  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy.engine.url import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from app.core.config import settings  # noqa: E402


def _parsed_url():
    return make_url(settings.database_url)


def _test_db_name() -> str:
    return f"{_parsed_url().database}_test"


def _admin_asyncpg_dsn() -> str:
    u = _parsed_url()
    return f"postgresql://{u.username}:{u.password}@{u.host}:{u.port}/postgres"


async def _admin_exec(sql: str) -> None:
    conn = await asyncpg.connect(_admin_asyncpg_dsn())
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def test_db_url() -> str:
    return (
        _parsed_url()
        .set(database=_test_db_name())
        .render_as_string(hide_password=False)
    )


@pytest.fixture(scope="session")
def setup_test_db(test_db_url: str) -> Iterator[None]:
    dbname = _test_db_name()
    asyncio.run(_admin_exec(f'DROP DATABASE IF EXISTS "{dbname}"'))
    asyncio.run(_admin_exec(f'CREATE DATABASE "{dbname}"'))

    try:
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", test_db_url)
        command.upgrade(cfg, "head")
        yield
    finally:
        asyncio.run(_admin_exec(f'DROP DATABASE IF EXISTS "{dbname}"'))


@pytest.fixture
async def db_session(
    setup_test_db: None, test_db_url: str
) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(test_db_url)
    async with engine.connect() as conn:
        trans = await conn.begin()
        SessionLocal = async_sessionmaker(bind=conn, expire_on_commit=False)
        async with SessionLocal() as session:
            try:
                yield session
            finally:
                if trans.is_active:
                    await trans.rollback()
    await engine.dispose()
