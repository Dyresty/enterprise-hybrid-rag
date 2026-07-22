import fitz  # PyMuPDF


def extract_pdf_text(pdf_path: str):
    """
    Extract text from every page of a PDF.

    Returns:
        List[dict]
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        pages.append(
            {
                "page": page_number + 1,
                "text": text
            }
        )

    document.close()

    return pages