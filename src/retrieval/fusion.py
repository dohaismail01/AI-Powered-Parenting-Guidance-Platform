"""
Reciprocal Rank Fusion (RRF).

Dense and BM25 scores live on different, incomparable scales (cosine
similarity vs. BM25's unbounded lexical score), so we can't just add
them together. RRF sidesteps that by fusing based on each result's
*rank* in its own list rather than its raw score — a standard,
parameter-light way to combine heterogeneous retrievers.

    RRF(chunk) = sum( 1 / (k + rank_in_list) )  over every list the
    chunk appears in.
"""
from __future__ import annotations

from config import FUSED_TOP_K, RRF_K
from src.schema import RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = RRF_K,
    top_k: int = FUSED_TOP_K,
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk.chunk_id
        merged.setdefault(chunk_id, result)
        merged[chunk_id].dense_score = result.dense_score
        merged[chunk_id].fused_score += 1.0 / (k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result.chunk.chunk_id
        if chunk_id not in merged:
            merged[chunk_id] = result
        else:
            merged[chunk_id].bm25_score = result.bm25_score
        merged[chunk_id].fused_score += 1.0 / (k + rank)

    fused = sorted(merged.values(), key=lambda r: r.fused_score, reverse=True)
    return fused[:top_k]
