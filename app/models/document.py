from typing import Optional
from pydantic import BaseModel


class ChunkMetadata(BaseModel):

    document_id: str
    filename: str

    page: int
    chunk_id: int

    parent_id: Optional[str] = None
    section: Optional[str] = None

    created_at: str

    source_type: str = "pdf"


class DocumentChunk(BaseModel):

    text: str

    metadata: ChunkMetadata