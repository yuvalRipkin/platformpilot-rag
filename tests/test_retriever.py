"""Integration tests for the Retriever.

We seed the chunks table with vectors that have known cosine similarity to a
fixed query vector ([1, 0, 0, ...]) and verify the retriever orders + filters
correctly. The HNSW index is approximate, but six rows is small enough that
ORDER BY <=> still returns exact-cosine ordering.
"""

import math

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.services.embedder import Embedder
from app.services.retriever import Retriever


def _vec_with_similarity_to_x_axis(target: float) -> list[float]:
    """Unit vector whose cosine similarity to [1,0,0,...] is `target`."""
    a = target
    b = math.sqrt(max(0.0, 1.0 - a * a))
    return [a, b] + [0.0] * 382


class StubEmbedder(Embedder):
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _ in texts]


@pytest.mark.integration
async def test_retrieves_top_k(db_session: AsyncSession) -> None:
    doc = Document(source="repo/retr-1.md", title="R1")
    db_session.add(doc)
    await db_session.flush()

    similarities = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45]
    for i, sim in enumerate(similarities):
        db_session.add(
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                text=f"chunk {i}",
                embedding=_vec_with_similarity_to_x_axis(sim),
            )
        )
    await db_session.flush()

    retriever = Retriever(StubEmbedder([1.0, 0.0] + [0.0] * 382))
    results = await retriever.retrieve(
        db_session, "anything", k=4, threshold=0.0
    )

    assert len(results) == 4
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True), "results must be ordered desc"
    expected = [0.95, 0.85, 0.75, 0.65]
    for actual, exp in zip(sims, expected, strict=True):
        assert abs(actual - exp) < 0.01, f"sim {actual} != {exp}"


@pytest.mark.integration
async def test_threshold_filters(db_session: AsyncSession) -> None:
    doc = Document(source="repo/retr-2.md", title="R2")
    db_session.add(doc)
    await db_session.flush()

    similarities = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45]
    for i, sim in enumerate(similarities):
        db_session.add(
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                text=f"chunk {i}",
                embedding=_vec_with_similarity_to_x_axis(sim),
            )
        )
    await db_session.flush()

    retriever = Retriever(StubEmbedder([1.0, 0.0] + [0.0] * 382))
    results = await retriever.retrieve(
        db_session, "anything", k=10, threshold=0.9
    )

    assert len(results) == 1
    assert results[0].chunk_index == 0
    assert results[0].similarity >= 0.9


@pytest.mark.integration
async def test_empty_corpus(db_session: AsyncSession) -> None:
    retriever = Retriever(StubEmbedder([1.0] + [0.0] * 383))
    results = await retriever.retrieve(
        db_session, "anything", k=4, threshold=0.0
    )
    assert results == []
