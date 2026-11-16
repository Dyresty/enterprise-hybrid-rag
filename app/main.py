from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.services.container import get_container


@asynccontextmanager
async def lifespan(app):
    get_container()  # load models at startup, not on the first request
    yield


app = FastAPI(
    title="Enterprise Hybrid RAG Platform",
    version="1.0.0",
    description="Hybrid retrieval (dense + BM25 + RRF), cross-encoder reranking and grounded, cited answers.",
    lifespan=lifespan,
)
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok", "chunks": get_container().vector_store.count()}
