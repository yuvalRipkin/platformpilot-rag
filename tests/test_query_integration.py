"""Integration test for the full /ingest -> /query path.

Uses a fake embedder (constant 384-d vector) and an echoing fake LLM. The
echoing LLM returns the user message it was given verbatim, so the test can
assert that the assembled context (numbered chunks with source attribution)
is what would have been sent to the real Anthropic API.

We do NOT call the real Anthropic API here — the only network call is to
local Postgres.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_answer_generator,
    get_embedder,
    get_retriever,
)
from app.core.config import settings
from app.db.session import get_db
from app.main import app
from app.services.answer_generator import AnswerGenerator
from app.services.embedder import Embedder
from app.services.llm_client import LLMClient
from app.services.retriever import Retriever


class FakeEmbedder(Embedder):
    def encode(self, texts: list[str]) -> list[list[float]]:
        # Identical 384-d unit vector for any input — every chunk and every
        # query lands at cosine similarity 1.0 to each other, so retrieval
        # returns the top_k by insertion order under the HNSW index.
        return [[1.0] + [0.0] * 383 for _ in texts]


class EchoLLM(LLMClient):
    """Returns the user prompt verbatim. Lets the test assert what was sent."""

    async def generate(self, system, user, max_tokens, temperature):
        return user


@pytest.fixture
def override_app(db_session: AsyncSession):
    embedder = FakeEmbedder()
    retriever = Retriever(embedder)
    llm = EchoLLM()
    gen = AnswerGenerator(retriever, llm, settings)

    async def _db():
        yield db_session

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_answer_generator] = lambda: gen
    yield
    app.dependency_overrides.clear()


@pytest.mark.integration
async def test_query_round_trip(
    db_session: AsyncSession, override_app: None
) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ingest = await client.post(
            "/ingest",
            json={
                "source": "test/known.md",
                "title": "Known",
                "content": (
                    "# Known Document\n\n"
                    + "This document discusses eigenvalues of certain "
                    "operators in functional analysis with specific "
                    "terminology like Hermitian operators and spectral "
                    "decomposition. " * 4
                    + "\n\n## Spectral theorem\n\n"
                    + "A Hermitian operator on a finite-dimensional inner "
                    "product space admits an orthonormal basis of "
                    "eigenvectors and the corresponding eigenvalues are "
                    "real numbers. " * 4
                ),
            },
        )
        assert ingest.status_code == 200, ingest.text

        response = await client.post(
            "/query", json={"query": "What does this document say?"}
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert isinstance(body["answer"], str) and body["answer"]
    assert len(body["chunks"]) >= 1
    sources = [c["source"] for c in body["chunks"]]
    assert "test/known.md" in sources

    # The echoing LLM returns the user prompt; assert the assembled context
    # wrapped each chunk in a <context_chunk> tag with the source carried as
    # an attribute and the citation index as the id attribute.
    assert "<context_chunk " in body["answer"]
    assert 'source="test/known.md"' in body["answer"]
    assert 'id="1"' in body["answer"]
