from functools import lru_cache

import numpy as np

OCR_LANGUAGES = ["en", "de"]
OCR_DPI = 200


@lru_cache
def get_reader():
    # Loaded only when a scanned page actually appears (the model download is ~100 MB).
    import easyocr

    return easyocr.Reader(OCR_LANGUAGES)


def ocr_page(page) -> str:
    """Render a PDF page to an image and read it with EasyOCR, in reading order."""

    pixmap = page.get_pixmap(dpi=OCR_DPI)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n == 4:
        image = image[:, :, :3]

    # paragraph=True merges nearby boxes into lines/paragraphs in reading order.
    results = get_reader().readtext(image, paragraph=True)
    return "\n".join(text for _, text in results)
