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
