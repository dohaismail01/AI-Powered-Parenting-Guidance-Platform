"""
Ingestion entry point.

Reads manifests describing the knowledge-base sources (the catalog you
already built — UNICEF/WHO/academic entries with age/topic/priority/
license metadata), loads + cleans + chunks them, and writes the result
to data/processed/chunks/ as JSON, ready for build_index.py to embed
and index.

Usage:
    python ingest.py
"""
from __future__ import annotations

import json

from config import CHUNKS_DIR, RAW_PDFS_DIR, RAW_RESEARCH_DIR
from src.chunking.chunker import chunk_documents
from src.ingestion.pdf_loader import load_pdf_directory
from src.ingestion.web_loader import load_web_manifest
from src.schema import Chunk

# Manifests describe each source's metadata (organization, age_range,
# topics, priority_stars, license_status). In practice, generate these
# from the ParentWise knowledge-base markdown file rather than typing
# them by hand — a small script can parse the "#### N. title" headings
# and bullet lists into exactly this shape.
PDF_MANIFEST_PATH = RAW_PDFS_DIR / "manifest.json"
WEB_MANIFEST_PATH = RAW_RESEARCH_DIR.parent / "web" / "manifest.json"


def _load_manifest(path) -> list[dict]:
    if not path.exists():
        print(f"[ingest] No manifest at {path}, skipping this source type.")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "organization": chunk.organization,
        "title": chunk.title,
        "section": chunk.section,
        "page": chunk.page,
        "age_range": chunk.age_range,
        "topics": chunk.topics,
        "priority_stars": chunk.priority_stars,
        "license_status": chunk.license_status,
        "url": chunk.url,
    }


def main() -> None:
    all_documents = []

    pdf_manifest = _load_manifest(PDF_MANIFEST_PATH)
    if pdf_manifest:
        print(f"[ingest] Loading {len(pdf_manifest)} PDF sources...")
        all_documents.extend(load_pdf_directory(RAW_PDFS_DIR, pdf_manifest))

    web_manifest = _load_manifest(WEB_MANIFEST_PATH)
    if web_manifest:
        print(f"[ingest] Loading {len(web_manifest)} web sources...")
        all_documents.extend(load_web_manifest(web_manifest))

    # Research-paper PDFs (Shamaa/university repositories/etc.) reuse the
    # same PDF loader — only the manifest and folder differ.
    research_manifest = _load_manifest(RAW_RESEARCH_DIR / "manifest.json")
    if research_manifest:
        print(f"[ingest] Loading {len(research_manifest)} research sources...")
        all_documents.extend(load_pdf_directory(RAW_RESEARCH_DIR, research_manifest))

    print(f"[ingest] Loaded {len(all_documents)} documents total. Chunking...")
    chunks = chunk_documents(all_documents)
    print(f"[ingest] Produced {len(chunks)} chunks.")

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHUNKS_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([_chunk_to_dict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    print(f"[ingest] Wrote chunks to {output_path}")


if __name__ == "__main__":
    main()
