from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from sqlalchemy.sql.dml import Delete

from app.api.dependencies import get_embedder
from app.db.models import Chunk, Document
from app.db.session import get_db
from app.main import app
from app.services.embedder import Embedder


class FakeEmbedder(Embedder):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("model unavailable")
        self.calls.append(list(texts))
        return [[0.1] * 384 for _ in texts]


def _mock_session(existing_doc: Document | None) -> MagicMock:
    session = MagicMock()
    call_log: list[tuple[str, object]] = []

    async def execute_side_effect(stmt):
        call_log.append(("execute", stmt))
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=existing_doc)
        return result

    session.execute = AsyncMock(side_effect=execute_side_effect)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()

    def add(obj: object) -> None:
        call_log.append(("add", obj))
        # New documents get their id from a real INSERT in production;
        # simulate that here so the endpoint can use doc.id for chunks.
        if isinstance(obj, Document) and getattr(obj, "id", None) is None:
            obj.id = uuid4()

    session.add = MagicMock(side_effect=add)
    session.call_log = call_log
    return session


def _override(session: object, embedder: Embedder) -> None:
    async def _db():
        yield session

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_embedder] = lambda: embedder


async def _post(body: dict) -> tuple[int, dict]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ingest", json=body)
    return response.status_code, response.json()


async def test_ingest_new_document():
    embedder = FakeEmbedder()
    session = _mock_session(existing_doc=None)
    _override(session, embedder)
    try:
        status, body = await _post(
            {
                "source": "test/foo.md",
                "content": (
                    "# Title\n\n"
                    + "This is a normal paragraph with several sentences. "
                    * 12
                ),
            }
        )
        assert status == 200, body
        assert body["is_replacement"] is False
        assert body["chunks_created"] >= 1
        assert "document_id" in body

        chunks_added = [
            obj for kind, obj in session.call_log
            if kind == "add" and isinstance(obj, Chunk)
        ]
        assert len(chunks_added) == body["chunks_created"]
        assert len(embedder.calls) == 1
        assert len(embedder.calls[0]) == body["chunks_created"]
    finally:
        app.dependency_overrides.clear()


async def test_ingest_existing_document():
    embedder = FakeEmbedder()
    existing = Document()
    existing.id = uuid4()
    existing.source = "test/foo.md"
    existing.title = "old"
    session = _mock_session(existing_doc=existing)

    _override(session, embedder)
    try:
        status, body = await _post(
            {
                "source": "test/foo.md",
                "title": "new",
                "content": (
                    "Replacement content with enough tokens to clear the "
                    "minimum threshold. " * 12
                ),
            }
        )
        assert status == 200, body
        assert body["is_replacement"] is True
        assert body["document_id"] == str(existing.id)

        first_delete_idx = next(
            (
                i for i, (kind, obj) in enumerate(session.call_log)
                if kind == "execute" and isinstance(obj, Delete)
            ),
            None,
        )
        assert first_delete_idx is not None, "DELETE for chunks must be issued"

        first_chunk_add_idx = next(
            (
                i for i, (kind, obj) in enumerate(session.call_log)
                if kind == "add" and isinstance(obj, Chunk)
            ),
            None,
        )
        assert first_chunk_add_idx is not None, "chunks must be inserted"
        assert first_delete_idx < first_chunk_add_idx, (
            "DELETE must precede chunk INSERTs"
        )
        assert existing.title == "new"
        assert len(embedder.calls) == 1
        assert any("Replacement" in t for t in embedder.calls[0])
    finally:
        app.dependency_overrides.clear()


async def test_ingest_empty_content_400():
    embedder = FakeEmbedder()
    session = _mock_session(existing_doc=None)
    _override(session, embedder)
    try:
        status, body = await _post(
            {"source": "test/foo.md", "content": "   \n\n  "}
        )
        assert status == 400
        assert "content" in body["detail"].lower()
    finally:
        app.dependency_overrides.clear()


async def test_ingest_too_short_content_400():
    embedder = FakeEmbedder()
    session = _mock_session(existing_doc=None)
    _override(session, embedder)
    try:
        status, body = await _post(
            {"source": "test/foo.md", "content": "x"}
        )
        assert status == 400
        assert "too short" in body["detail"].lower()
    finally:
        app.dependency_overrides.clear()


async def test_ingest_embedder_failure_500():
    embedder = FakeEmbedder(fail=True)
    session = _mock_session(existing_doc=None)
    _override(session, embedder)
    try:
        status, body = await _post(
            {
                "source": "test/foo.md",
                "content": (
                    "Content long enough to clear the minimum-token check. "
                    * 12
                ),
            }
        )
        assert status == 500
        # Real exception text must NOT leak into the response.
        assert "model unavailable" not in str(body)
        assert "ingestion failed" in str(body).lower()
        # Rollback must have been called when the embedder blew up mid-txn.
        session.rollback.assert_awaited_once()
    finally:
        app.dependency_overrides.clear()
