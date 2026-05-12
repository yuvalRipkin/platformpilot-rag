import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_answer_generator
from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievedChunkResponse,
)
from app.core.metrics import rag_queries_total
from app.db.session import get_db
from app.services.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    answer_gen: AnswerGenerator = Depends(get_answer_generator),
) -> QueryResponse:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    query_id = uuid.uuid4()
    started = time.perf_counter()
    try:
        # Retrieval and LLM histograms are observed inside AnswerGenerator
        # so /search and /query share the same per-stage timing buckets.
        answer, chunks = await answer_gen.generate_answer(db, body.query)
    except Exception:
        rag_queries_total.labels(endpoint="query", status="error").inc()
        logger.exception(
            "query failed",
            extra={"query_id": str(query_id)},
        )
        raise HTTPException(status_code=500, detail="query failed")

    latency_ms = int((time.perf_counter() - started) * 1000)
    rag_queries_total.labels(endpoint="query", status="ok").inc()

    logger.info(
        "query processed",
        extra={
            "query_id": str(query_id),
            "chunks_returned": len(chunks),
            "latency_ms": latency_ms,
        },
    )

    return QueryResponse(
        query_id=query_id,
        answer=answer,
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
