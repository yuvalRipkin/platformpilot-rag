import uuid

from pydantic import BaseModel


class IngestRequest(BaseModel):
    source: str
    title: str | None = None
    content: str


class IngestResponse(BaseModel):
    document_id: uuid.UUID
    chunks_created: int
    is_replacement: bool


class SearchRequest(BaseModel):
    query: str
    k: int | None = None
    threshold: float | None = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source: str
    chunk_index: int
    text: str
    similarity: float


class SearchResponse(BaseModel):
    query_id: uuid.UUID
    chunks: list[RetrievedChunkResponse]
    latency_ms: int


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query_id: uuid.UUID
    answer: str
    chunks: list[RetrievedChunkResponse]
    latency_ms: int
