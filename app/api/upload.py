from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.parser import extract_pdf_text

router = APIRouter()

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    save_path = UPLOAD_FOLDER / file.filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages = extract_pdf_text(str(save_path))

    return {
        "filename": file.filename,
        "pages": len(pages),
        "preview": pages[0]["text"][:500] if pages else ""
    }