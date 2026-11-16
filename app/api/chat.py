from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.models.document import ChatRequest, ChatResponse
from app.services.container import get_container

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Model inference is blocking; keep it off the event loop.
    return await run_in_threadpool(get_container().rag.generate_answer, request.question, request.top_k)


@router.get("/search")
async def search(q: str, top_k: int = 5):
    """Retrieval only, no LLM: useful for debugging what the model would see."""
    docs = await run_in_threadpool(get_container().rag.retrieve, q, top_k)
    return [
        {k: d.get(k) for k in ("title", "filename", "page", "document_id", "rerank_score", "text")}
        for d in docs
    ]
