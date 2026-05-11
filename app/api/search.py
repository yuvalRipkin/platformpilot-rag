import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_retriever
from app.api.schemas import (
    RetrievedChunkResponse,
    SearchRequest,
    SearchResponse,
)
from app.core.config import settings
from app.core.metrics import (
    rag_chunks_retrieved,
    rag_queries_total,
    rag_retrieval_duration_seconds,
)
from app.db.session import get_db
from app.services.retriever import Retriever

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    db: AsyncSession = Depends(get_db),
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    query_id = uuid.uuid4()
    k = body.k if body.k is not None else settings.top_k
    threshold = (
        body.threshold
        if body.threshold is not None
        else settings.similarity_threshold
    )

    started = time.perf_counter()
    try:
        with rag_retrieval_duration_seconds.time():
            chunks = await retriever.retrieve(db, body.query, k, threshold)
    except Exception:
        rag_queries_total.labels(endpoint="search", status="error").inc()
        logger.exception(
            "search failed",
            extra={"query_id": str(query_id)},
        )
        raise HTTPException(status_code=500, detail="search failed")

    latency_ms = int((time.perf_counter() - started) * 1000)
    rag_queries_total.labels(endpoint="search", status="ok").inc()
    rag_chunks_retrieved.observe(len(chunks))

    logger.info(
        "search processed",
        extra={
            "query_id": str(query_id),
            "chunks_returned": len(chunks),
            "latency_ms": latency_ms,
            "top_similarity": chunks[0].similarity if chunks else None,
        },
    )

    return SearchResponse(
        query_id=query_id,
        chunks=[
            RetrievedChunkResponse(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                source=c.source,
                chunk_index=c.chunk_index,
                text=c.text,
                similarity=c.similarity,
            )
            for c in chunks
        ],
        latency_ms=latency_ms,
    )
