import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, ingest
from app.core.config import settings
from app.services.embedder import SentenceTransformerEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading embedder model")
    app.state.embedder = SentenceTransformerEmbedder()
    logger.info("Embedder ready")
    yield
    from app.db.session import engine

    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(ingest.router)
