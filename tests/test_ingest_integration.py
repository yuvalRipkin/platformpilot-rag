"""Integration tests for /ingest.

These tests exercise the real DB path (chunks persisted, replacement
deletes old rows, vector column round-trips through asyncpg) but use a
fake embedder that returns deterministic 384-d unit vectors. The fake
keeps the test suite ~10s faster on first run, avoids loading torch +
the all-MiniLM-L6-v2 weights, and the embedding model itself isn't what
these tests are verifying — the DB path is.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_embedder
from app.db.models import Chunk, Document
from app.db.session import get_db
from app.main import app
from app.services.embedder import Embedder


class FakeEmbedder(Embedder):
    def encode(self, texts: list[str]) -> list[list[float]]:
        # Deterministic unit-normalized 384-d vectors.
        out: list[list[float]] = []
        for i, _ in enumerate(texts):
            v = [0.0] * 384
            v[i % 384] = 1.0
            out.append(v)
        return out


@pytest.fixture
def override_app(db_session: AsyncSession):
    async def _db():
        yield db_session

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    yield
    app.dependency_overrides.clear()


@pytest.mark.integration
async def test_ingest_round_trip(
    db_session: AsyncSession, override_app: None
) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            json={
                "source": "repo/round-trip.md",
                "title": "Round Trip",
                "content": (
                    "# Round Trip\n\n"
                    "This document exercises the full ingest path through "
                    "the chunker, the fake embedder, and the chunks table "
                    "with its pgvector column.\n\n"
                    "## Section\n\nA second short section follows."
                ),
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_replacement"] is False

    chunks = (
        await db_session.execute(
            select(Chunk).where(Chunk.document_id == body["document_id"])
        )
    ).scalars().all()
    assert len(chunks) == body["chunks_created"]
    for c in chunks:
        assert len(c.embedding) == 384


@pytest.mark.integration
async def test_ingest_replaces_chunks(
    db_session: AsyncSession, override_app: None
) -> None:
    source = "repo/replace.md"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/ingest",
            json={
                "source": source,
                "content": "# Original\n\nFirst version of the content.",
            },
        )
        assert first.status_code == 200, first.text
        assert first.json()["is_replacement"] is False
        original_doc_id = first.json()["document_id"]

        second = await client.post(
            "/ingest",
            json={
                "source": source,
                "title": "Replaced",
                "content": (
                    "# Replaced\n\nCompletely different content. "
                    + "Sentence after sentence. " * 30
                ),
            },
        )
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["is_replacement"] is True
    assert second_body["document_id"] == original_doc_id

    chunks = (
        await db_session.execute(
            select(Chunk).where(Chunk.document_id == original_doc_id)
        )
    ).scalars().all()
    assert len(chunks) == second_body["chunks_created"]
    for c in chunks:
        assert "First version of the content" not in c.text

    doc = (
        await db_session.execute(
            select(Document).where(Document.id == original_doc_id)
        )
    ).scalar_one()
    assert doc.title == "Replaced"
