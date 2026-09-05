"""
Cross-encoder reranking.

Bi-encoders (the dense retriever) score the query and each chunk
*independently* then compare vectors -- fast, but approximate. A
cross-encoder reranker reads the (query, chunk) pair *together* and
directly estimates relevance, which is far more precise but too slow
to run over the whole corpus. So: retrieve ~30 candidates cheaply
(dense + BM25 + metadata ranking), then rerank only those down to the
5-8 chunks that actually go to the LLM.

BAAI/bge-reranker-v2-m3 pairs naturally with bge-m3 (same lab, same
multilingual/Arabic training emphasis) and is open-source.
"""
from __future__ import annotations

from FlagEmbedding import FlagReranker

from config import MIN_RERANK_SCORE, RERANK_TOP_K, RERANKER_DEVICE, RERANKER_MODEL_NAME
from src.schema import RetrievedChunk

_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(
            RERANKER_MODEL_NAME,
            use_fp16=(RERANKER_DEVICE == "cuda"),
            device=RERANKER_DEVICE,
        )
    return _reranker


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int = RERANK_TOP_K,
    min_score: float = MIN_RERANK_SCORE,
) -> list[RetrievedChunk]:
    """
    Rerank candidates and return the top_k, but only those clearing
    min_score -- so a query with few genuinely relevant chunks returns
    fewer (or zero) results instead of padding out to top_k with noise.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [[query, result.chunk.text] for result in candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    if isinstance(scores, float):
        scores = [scores]

    for result, score in zip(candidates, scores):
        result.rerank_score = float(score)

    reranked = sorted(candidates, key=lambda r: r.rerank_score, reverse=True)
    filtered = [r for r in reranked if r.rerank_score >= min_score]
    return filtered[:top_k]
