import asyncio
from collections.abc import AsyncIterator, Iterator

import asyncpg
import pytest
from sqlalchemy import select
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import selectinload

from alembic import command
from alembic.config import Config
from app.core.config import settings
from app.db.models import Chunk, Document


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
    return _parsed_url().set(database=_test_db_name()).render_as_string(
        hide_password=False
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


def _vector(seed: int) -> list[float]:
    return [((seed + i) % 7) / 7.0 for i in range(384)]


@pytest.mark.integration
async def test_document_persist_and_query(db_session: AsyncSession) -> None:
    doc = Document(source="repo/foo.md", title="Foo")
    db_session.add(doc)
    await db_session.flush()

    fetched = (
        await db_session.execute(select(Document).where(Document.id == doc.id))
    ).scalar_one()

    assert fetched.source == "repo/foo.md"
    assert fetched.title == "Foo"
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


@pytest.mark.integration
async def test_document_with_chunks_ordered(db_session: AsyncSession) -> None:
    doc = Document(source="repo/bar.md", title="Bar")
    db_session.add(doc)
    await db_session.flush()
    for i in range(3):
        db_session.add(
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                text=f"chunk {i}",
                embedding=_vector(i),
            )
        )
    await db_session.flush()
    db_session.expunge_all()

    fetched = (
        await db_session.execute(
            select(Document)
            .where(Document.source == "repo/bar.md")
            .options(selectinload(Document.chunks))
        )
    ).scalar_one()

    chunks_sorted = sorted(fetched.chunks, key=lambda c: c.chunk_index)
    assert [c.chunk_index for c in chunks_sorted] == [0, 1, 2]
    assert [c.text for c in chunks_sorted] == ["chunk 0", "chunk 1", "chunk 2"]


@pytest.mark.integration
async def test_chunks_unique_document_chunk_index(db_session: AsyncSession) -> None:
    doc = Document(source="repo/baz.md")
    db_session.add(doc)
    await db_session.flush()
    db_session.add(
        Chunk(document_id=doc.id, chunk_index=0, text="first", embedding=_vector(0))
    )
    db_session.add(
        Chunk(document_id=doc.id, chunk_index=0, text="dup", embedding=_vector(1))
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.integration
async def test_chunks_cascade_delete(db_session: AsyncSession) -> None:
    doc = Document(source="repo/qux.md")
    db_session.add(doc)
    await db_session.flush()
    for i in range(2):
        db_session.add(
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                text=f"c{i}",
                embedding=_vector(i),
            )
        )
    await db_session.flush()
    doc_id = doc.id
    db_session.expunge_all()

    fetched = (
        await db_session.execute(
            select(Document)
            .where(Document.id == doc_id)
            .options(selectinload(Document.chunks))
        )
    ).scalar_one()
    await db_session.delete(fetched)
    await db_session.flush()

    remaining = (
        await db_session.execute(select(Chunk).where(Chunk.document_id == doc_id))
    ).all()
    assert remaining == []
