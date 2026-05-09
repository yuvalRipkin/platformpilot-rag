import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_embedder
from app.api.schemas import IngestRequest, IngestResponse
from app.db.models import Chunk, Document
from app.db.session import get_db
from app.services.chunker import chunk_markdown
from app.services.embedder import Embedder

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    db: AsyncSession = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
) -> IngestResponse:
    if not body.source.strip():
        raise HTTPException(status_code=400, detail="source must not be empty")
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")

    try:
        existing = (
            await db.execute(select(Document).where(Document.source == body.source))
        ).scalar_one_or_none()

        is_replacement = existing is not None
        if existing is not None:
            await db.execute(delete(Chunk).where(Chunk.document_id == existing.id))
            existing.title = body.title
            doc = existing
        else:
            doc = Document(source=body.source, title=body.title)
            db.add(doc)
            await db.flush()

        chunks = chunk_markdown(body.content)
        embeddings = embedder.encode([c.text for c in chunks])
        for chunk, vector in zip(chunks, embeddings, strict=True):
            db.add(
                Chunk(
                    document_id=doc.id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    embedding=vector,
                )
            )
        await db.commit()
        return IngestResponse(
            document_id=doc.id,
            chunks_created=len(chunks),
            is_replacement=is_replacement,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Ingestion failed")
        await db.rollback()
        raise HTTPException(status_code=500, detail="ingestion failed")
