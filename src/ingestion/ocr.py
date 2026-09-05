"""
OCR fallback for scanned PDF pages.

Plain Tesseract is noticeably weaker on Arabic script (ligatures,
diacritics, RTL) than on Latin text. Two tiers are supported:

1. Tesseract + `ara` language pack — free, works offline, "good enough"
   for clean scans. Used by default.
2. Cloud OCR (Google Vision / Azure Document Intelligence) — much more
   accurate on Arabic, recommended for production if any of the source
   PDFs are scanned rather than digitally generated. Wire in your own
   credentials via `cloud_ocr_page` below.
"""
from __future__ import annotations

import io

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

TESSERACT_LANG = "ara"  # requires: apt-get install tesseract-ocr-ara
RENDER_DPI = 300         # higher DPI significantly improves Arabic OCR accuracy


def ocr_pdf_page(doc: "fitz.Document", page_index: int) -> str | None:
    """Render a PDF page to an image and OCR it with Tesseract (Arabic)."""
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=RENDER_DPI)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, lang=TESSERACT_LANG)
        return text
    except Exception as exc:  # pragma: no cover - defensive, OCR is best-effort
        print(f"[ocr] Tesseract OCR failed on page {page_index + 1}: {exc}")
        return None


def cloud_ocr_page(image_bytes: bytes) -> str | None:
    """
    Placeholder for a higher-accuracy cloud OCR call (Google Vision or
    Azure Document Intelligence). Swap this in for `ocr_pdf_page` if
    Tesseract quality is insufficient for your scanned sources.

    Left unimplemented here since it requires API credentials that are
    specific to your deployment.
    """
    raise NotImplementedError(
        "Wire this up to Google Cloud Vision or Azure Document Intelligence "
        "if scanned-PDF OCR quality with Tesseract is not sufficient."
    )
