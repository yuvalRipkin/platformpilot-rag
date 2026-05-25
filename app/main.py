import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import health, ingest, query, search
from app.core.config import settings
from app.services.answer_generator import AnswerGenerator
from app.services.embedder import SentenceTransformerEmbedder
from app.services.llm_client import AnthropicClient
from app.services.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading embedder model")
    app.state.embedder = SentenceTransformerEmbedder()
    logger.info("Embedder ready")
    app.state.retriever = Retriever(app.state.embedder)
    app.state.llm = AnthropicClient(
        settings.anthropic_api_key,
        settings.anthropic_model,
        settings.anthropic_timeout_seconds,
    )
    app.state.answer_generator = AnswerGenerator(
        app.state.retriever, app.state.llm, settings
    )
    yield
    from app.db.session import engine

    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(query.router)
