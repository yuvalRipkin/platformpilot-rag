import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Chunk, Document


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
