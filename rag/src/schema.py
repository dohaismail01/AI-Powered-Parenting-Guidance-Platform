"""
Shared data structures used across ingestion, chunking, retrieval and
generation. Keeping one schema avoids every module inventing its own
dict shape for a "document" or "chunk".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RawDocument:
    """One ingested source before cleaning/chunking."""
    doc_id: str
    text: str
    organization: str
    title: str
    source_format: str
    age_range: Optional[str] = None
    topics: list[str] = field(default_factory=list)
    priority_stars: int = 3
    license_status: str = "unknown"
    url: Optional[str] = None
    page_map: Optional[dict[int, str]] = None


@dataclass
class Chunk:
    """A retrievable unit stored in the vector DB / BM25 index."""
    chunk_id: str
    doc_id: str
    text: str
    organization: str
    title: str
    section: Optional[str] = None
    page: Optional[int] = None
    age_range: Optional[str] = None
    topics: list[str] = field(default_factory=list)
    priority_stars: int = 3
    license_status: str = "unknown"
    url: Optional[str] = None

    def metadata(self) -> dict:
        """Flat metadata dict, as required by Chroma (no nested lists)."""
        age_range = self.age_range
        if isinstance(age_range, dict):
            low, high = age_range.get("min"), age_range.get("max")
            age_range = f"{low}-{high}" if low is not None and high is not None else None

        return {
            "doc_id": self.doc_id,
            "organization": self.organization,
            "title": self.title,
            "section": self.section or "",
            "page": self.page if self.page is not None else -1,
            "age_range": age_range or "",
            "topics": ",".join(self.topics),
            "priority_stars": self.priority_stars,
            "license_status": self.license_status,
            "url": self.url or "",
        }


@dataclass
class RetrievedChunk:
    """A chunk plus the scores accumulated as it moves through the pipeline."""
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    rerank_score: float = 0.0
