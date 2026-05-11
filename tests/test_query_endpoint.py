from uuid import uuid4

from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_answer_generator
from app.db.session import get_db
from app.main import app
from app.services.retriever import RetrievedChunk


class FakeAnswerGenerator:
    def __init__(
        self,
        answer: str = "test answer",
        chunks: list[RetrievedChunk] | None = None,
    ) -> None:
        self.answer = answer
        self.chunks = chunks or []
        self.last_query: str | None = None

    async def generate_answer(self, db, query):
        self.last_query = query
        return self.answer, self.chunks


def _override(gen: FakeAnswerGenerator) -> None:
    async def _db():
        yield None

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_answer_generator] = lambda: gen


async def _post(body: dict) -> tuple[int, dict]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json=body)
    return response.status_code, response.json()


def _chunk(idx: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        source=f"src{idx}.md",
        text=f"text {idx}",
        chunk_index=idx,
        similarity=0.9 - idx * 0.1,
    )


async def test_query_returns_answer_and_chunks():
    gen = FakeAnswerGenerator(answer="generated reply", chunks=[_chunk(0)])
    _override(gen)
    try:
        status, body = await _post({"query": "What is X?"})
        assert status == 200, body
        assert body["answer"] == "generated reply"
        assert len(body["chunks"]) == 1
        assert "query_id" in body
        assert isinstance(body["latency_ms"], int)
        assert gen.last_query == "What is X?"
    finally:
        app.dependency_overrides.clear()


async def test_query_empty_query_400():
    gen = FakeAnswerGenerator()
    _override(gen)
    try:
        status, body = await _post({"query": "   "})
        assert status == 400
        assert "query" in body["detail"].lower()
    finally:
        app.dependency_overrides.clear()


async def test_query_no_chunks_still_returns_200_with_fallback_answer():
    fallback = "I don't have information about that in the indexed documents."
    gen = FakeAnswerGenerator(answer=fallback, chunks=[])
    _override(gen)
    try:
        status, body = await _post({"query": "obscure thing"})
        assert status == 200, body
        assert "don't have" in body["answer"].lower()
        assert body["chunks"] == []
    finally:
        app.dependency_overrides.clear()
