import pymupdf
import pymupdf4llm

from app.ingestion.ocr import ocr_page

# A page with less extractable text than this is treated as scanned and sent to OCR.
MIN_TEXT_CHARS = 50


def extract_pdf_text(pdf_path: str):
    """
    Extract every page of a PDF as Markdown, keeping the document structure
    (headings, lists, tables) so chunks split along sections instead of mid-table.
    Scanned pages without a text layer fall back to OCR.

    Returns:
        list of {"page": int, "text": str, "ocr": bool}
    """

    with pymupdf.open(pdf_path) as document:
        markdown_pages = pymupdf4llm.to_markdown(document, page_chunks=True)

        pages = []
        for index, chunk in enumerate(markdown_pages):
            text = chunk["text"]
            used_ocr = False

            if len(text.strip()) < MIN_TEXT_CHARS and document[index].get_images():
                text = ocr_page(document[index])
                used_ocr = True

            pages.append({"page": index + 1, "text": text, "ocr": used_ocr})

    return pages
