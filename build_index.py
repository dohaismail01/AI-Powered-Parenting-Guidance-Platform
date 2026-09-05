"""
Index-building entry point.

Reads data/processed/chunks/chunks.json (produced by ingest.py) and
builds both indexes ParentWise needs at query time:
  - dense vectors in Chroma (src/retrieval/dense_retriever.py)
  - a BM25 index persisted to vector_store/bm25/ (src/retrieval/bm25_retriever.py)

Usage:
    python build_index.py
"""
from __future__ import annotations

import json

from config import CHUNKS_DIR
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.schema import Chunk

CHUNKS_PATH = CHUNKS_DIR / "chunks.json"


def _load_chunks() -> list[Chunk]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [Chunk(**item) for item in raw]


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise SystemExit(
            f"No chunks found at {CHUNKS_PATH}. Run `python ingest.py` first."
        )

    chunks = _load_chunks()
    print(f"[build_index] Loaded {len(chunks)} chunks.")

    print("[build_index] Building dense (BGE-M3 + Chroma) index...")
    dense_retriever = DenseRetriever()
    # Embed/index in batches so a large corpus doesn't spike memory.
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        dense_retriever.index(batch)
        print(f"  indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print("[build_index] Building BM25 index...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.index(chunks)
    bm25_retriever.save()

    print("[build_index] Done. Indexes are ready for querying.")


if __name__ == "__main__":
    main()
