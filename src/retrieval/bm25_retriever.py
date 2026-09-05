"""
Sparse (lexical) retrieval via BM25.

Complements dense retrieval: embeddings capture *meaning* ("مش بيسمع
الكلام" -> content about "رفض التعليمات"), but BM25 is what reliably
surfaces a chunk when the parent's question uses the *exact* clinical
or common term the source uses (e.g. "نوبات الغضب", "الأليكسيثيميا").
Arabic's rich morphology means a naive BM25 over raw text underperforms,
so we tokenize the *normalized* text (see arabic_cleaner.py) — stripped
diacritics, unified alef/yaa forms — before indexing and querying.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import BM25_DIR, BM25_TOP_K
from src.preprocessing.arabic_cleaner import simple_arabic_tokenize
from src.schema import Chunk, RetrievedChunk

_INDEX_PATH = Path(BM25_DIR) / "bm25_index.pkl"


class BM25Retriever:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized_corpus = [simple_arabic_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self, path: Path = _INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load(self, path: Path = _INDEX_PATH) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[RetrievedChunk]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded — call index() or load() first.")

        tokenized_query = simple_arabic_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]

        return [
            RetrievedChunk(chunk=chunk, bm25_score=float(score))
            for chunk, score in ranked
            if score > 0
        ]
