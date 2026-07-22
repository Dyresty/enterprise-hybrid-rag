from fastapi import FastAPI
from app.api.upload import router as upload_router

app = FastAPI(
    title="Enterprise Hybrid RAG Platform",
    version="1.0.0",
    description="Enterprise Retrieval-Augmented Generation Platform"
)

app.include_router(upload_router)

@app.get("/")
async def root():
    return {
        "message": "Enterprise Hybrid RAG Platform Running"
    }