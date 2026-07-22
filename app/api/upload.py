from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.parser import extract_pdf_text

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
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    save_path = UPLOAD_FOLDER / filename

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

    return {
        "document_id": file_id,
        "filename": file.filename,
        "stored_as": filename,
        "pages": len(pages),
        "preview":
            pages[0]["text"][:500]
            if pages
            else ""
    }