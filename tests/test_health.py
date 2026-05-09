from contextlib import contextmanager
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient

from app.db.session import get_db
from app.main import app


@contextmanager
def _staged_embedder():
    """Lifespan doesn't run under ASGITransport; stage app.state.embedder
    so /ready's embedder check sees a loaded embedder."""
    app.state.embedder = object()
    try:
        yield
    finally:
        if hasattr(app.state, "embedder"):
            del app.state.embedder


async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_db_up():
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=None)

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with _staged_embedder():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
    finally:
        app.dependency_overrides.clear()


async def test_ready_db_down():
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=RuntimeError("connection refused"))

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with _staged_embedder():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready", "reason": "database unavailable"}
    finally:
        app.dependency_overrides.clear()


async def test_ready_embedder_not_loaded():
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=None)

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        # Note: do NOT stage an embedder. Lifespan never ran, so
        # app.state.embedder is absent — /ready must report 503.
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready", "reason": "embedder not loaded"}
    finally:
        app.dependency_overrides.clear()
