from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embedder import Embedder


@dataclass
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    source: str
    text: str
    chunk_index: int
    similarity: float


# Pre-normalized embeddings + vector_cosine_ops means the <=> operator returns
# true cosine distance, so similarity = 1 - distance. Casting :vec via text()
# keeps us off the pgvector ORM helpers (more readable for a one-liner
# similarity search than mapping <=> through SQLAlchemy operator overloads).
_RETRIEVE_SQL = text(
    """
    SELECT
        chunks.id AS chunk_id,
        chunks.document_id AS document_id,
        documents.source AS source,
        chunks.text AS text,
        chunks.chunk_index AS chunk_index,
        1 - (chunks.embedding <=> CAST(:vec AS vector)) AS similarity
    FROM chunks
    JOIN documents ON chunks.document_id = documents.id
    ORDER BY chunks.embedding <=> CAST(:vec AS vector)
    LIMIT :k
    """
)


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class Retriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        k: int,
        threshold: float,
    ) -> list[RetrievedChunk]:
        vector = self.embedder.encode([query])[0]
        result = await db.execute(
            _RETRIEVE_SQL,
            {"vec": _vector_literal(vector), "k": k},
        )
        rows = result.mappings().all()
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                source=row["source"],
                text=row["text"],
                chunk_index=row["chunk_index"],
                similarity=float(row["similarity"]),
            )
            for row in rows
            if float(row["similarity"]) >= threshold
        ]
