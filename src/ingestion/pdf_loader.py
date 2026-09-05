"""
PDF ingestion.

Uses PyMuPDF (fitz) because it is fast, handles Arabic/RTL text well,
and gives per-page text — which we keep, since ParentWise citations
reference page numbers (see prompt.py / citation format).

Falls back to OCR (src/ingestion/ocr.py) for pages that come back with
little or no extractable text — i.e. scanned pages.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import fitz  # PyMuPDF

from src.ingestion.ocr import ocr_pdf_page
from src.schema import RawDocument

MIN_CHARS_PER_PAGE = 20  # below this we assume the page is a scanned image


def load_pdf(
    path: Path,
    organization: str,
    title: str,
    age_range: str | None = None,
    topics: list[str] | None = None,
    priority_stars: int = 3,
    license_status: str = "unknown",
    url: str | None = None,
) -> RawDocument:
    """Extract text from a PDF, page by page, with OCR fallback."""
    doc = fitz.open(path)
    page_map: dict[int, str] = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")

        if len(text.strip()) < MIN_CHARS_PER_PAGE:
            # Likely a scanned page — try OCR instead.
            text = ocr_pdf_page(doc, page_index) or text

        page_map[page_index + 1] = text  # 1-indexed for human-readable citations

    full_text = "\n\n".join(page_map.values())

    return RawDocument(
        doc_id=str(uuid.uuid4()),
        text=full_text,
        organization=organization,
        title=title,
        source_format="pdf",
        age_range=age_range,
        topics=topics or [],
        priority_stars=priority_stars,
        license_status=license_status,
        url=url,
        page_map=page_map,
    )


def load_pdf_directory(directory: Path, manifest: list[dict]) -> list[RawDocument]:
    """
    Bulk-load every PDF listed in a manifest, e.g.:

        [
          {
            "filename": "unicef_positive_parenting_guide.pdf",
            "organization": "UNICEF Egypt",
            "title": "دليل التربية الإيجابية",
            "age_range": "0-18",
            "topics": ["التربية الإيجابية", "التأديب الإيجابي"],
            "priority_stars": 5,
            "license_status": "yellow",
            "url": "https://www.unicef.org/egypt/..."
          },
          ...
        ]

    This mirrors the knowledge-base entries you already catalogued —
    the manifest is typically generated from that markdown file.
    """
    documents = []
    for entry in manifest:
        pdf_path = directory / entry["filename"]
        if not pdf_path.exists():
            print(f"[pdf_loader] WARNING: missing file {pdf_path}, skipping.")
            continue
        documents.append(
            load_pdf(
                path=pdf_path,
                organization=entry["organization"],
                title=entry["title"],
                age_range=entry.get("age_range"),
                topics=entry.get("topics", []),
                priority_stars=entry.get("priority_stars", 3),
                license_status=entry.get("license_status", "unknown"),
                url=entry.get("url"),
            )
        )
    return documents
