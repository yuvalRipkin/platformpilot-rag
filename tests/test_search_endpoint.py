from uuid import uuid4

from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_retriever
from app.db.session import get_db
from app.main import app
from app.services.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.last_query: str | None = None
        self.last_k: int | None = None
        self.last_threshold: float | None = None

    async def retrieve(self, db, query, k, threshold):
        self.last_query = query
        self.last_k = k
        self.last_threshold = threshold
        return self.chunks


def _override(retriever: FakeRetriever) -> None:
    async def _db():
        yield None

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_retriever] = lambda: retriever


async def _post(body: dict) -> tuple[int, dict]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/search", json=body)
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


async def test_search_returns_chunks_and_query_id():
    retriever = FakeRetriever([_chunk(0), _chunk(1)])
    _override(retriever)
    try:
        status, body = await _post({"query": "what is X?"})
        assert status == 200, body
        assert "query_id" in body
        assert "latency_ms" in body
        assert isinstance(body["latency_ms"], int)
        assert len(body["chunks"]) == 2
        assert body["chunks"][0]["text"] == "text 0"
        assert body["chunks"][0]["similarity"] == 0.9
    finally:
        app.dependency_overrides.clear()


async def test_search_empty_query_400():
    retriever = FakeRetriever()
    _override(retriever)
    try:
        status, body = await _post({"query": "   "})
        assert status == 400
        assert "query" in body["detail"].lower()
    finally:
        app.dependency_overrides.clear()


async def test_search_overrides_k_and_threshold():
    retriever = FakeRetriever()
    _override(retriever)
    try:
        status, _ = await _post(
            {"query": "anything", "k": 7, "threshold": 0.42}
        )
        assert status == 200
        assert retriever.last_k == 7
        assert retriever.last_threshold == 0.42
    finally:
        app.dependency_overrides.clear()
