from typing import Literal, Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: int
    parent_id: str
    created_at: str
    source_type: str = "pdf"
    title: Optional[str] = None
    url: Optional[str] = None


class DocumentChunk(BaseModel):
    text: str  # child chunk: what gets embedded and scored
    parent_text: str  # parent chunk: what the LLM sees
    metadata: ChunkMetadata


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    n: int
    document_id: str
    filename: str
    page: int
    chunk_id: int
    excerpt: str
    title: Optional[str] = None
    url: Optional[str] = None
    score: float


class ChatResponse(BaseModel):
    answer_id: str
    question: str
    answer: str
    sources: list[Source]
    needs_review: bool = False
    review_reasons: list[str] = []


class FeedbackRequest(BaseModel):
    answer_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None
