from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.parser import extract_pdf_text
from app.ingestion.chunker import chunk_documents
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vectorstore import VectorStore

router = APIRouter()

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # Generate unique filename
    document_id = str(uuid.uuid4())
    original_filename = Path(file.filename).name
    save_path = UPLOAD_FOLDER / original_filename

    if save_path.exists():
        stem = save_path.stem
        suffix = save_path.suffix
        save_path = (
            UPLOAD_FOLDER /
            f"{stem}_{document_id[:8]}{suffix}"
        )

    # Save file
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Extract text
    pages = extract_pdf_text(
        str(save_path)
    )

    chunks = chunk_documents(
        pages
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]
    embedding_model = EmbeddingModel()
    embeddings = (
        embedding_model
        .generate_embeddings(texts)
    )

    vector_store = VectorStore()
    vector_store.create_collection()
    vector_store.insert_chunks(
        chunks,
        embeddings,
        document_id,
        original_filename
    )

    return {
        "document_id": document_id,
        "filename": original_filename,
        "stored_as": save_path.name,
        "pages": len(pages),
        "total_chunks": len(chunks),
        "first_chunk":
        chunks[0]["text"]
            if chunks
            else ""
    }