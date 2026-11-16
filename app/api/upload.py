import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.services.container import get_container

router = APIRouter()

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    document_id = str(uuid.uuid4())
    filename = Path(file.filename).name  # strips any client-supplied path
    save_path = UPLOAD_FOLDER / f"{document_id[:8]}_{filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = await run_in_threadpool(get_container().ingestion.ingest_pdf, str(save_path), document_id, filename)
    return {"document_id": document_id, "filename": filename, **result}
