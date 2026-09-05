"""
Metadata-aware ranking.

After hybrid retrieval (fusion.py) gives us ~30 relevant-ish candidates,
we adjust their order using ParentWise-specific signals that pure text
similarity cannot see: does the source's age range match the child's
age, does its topic tag match the query's inferred topic, and how
authoritative is the source (star priority from the knowledge base).

These are boosts, not filters, and they're deliberately small relative
to the retrieval score (see config.WEIGHT_*): a highly authoritative
but off-topic chunk must never outrank a highly relevant one — that
would silently make answers worse in the name of "trustworthiness".
"""
from __future__ import annotations

from config import (
    WEIGHT_AGE_MATCH,
    WEIGHT_RETRIEVAL_SCORE,
    WEIGHT_SOURCE_PRIORITY,
    WEIGHT_TOPIC_MATCH,
)
from src.query.query_analyzer import QueryAnalysis
from src.schema import RetrievedChunk


def _parse_age_range(age_range: str | None) -> tuple[int, int] | None:
    if not age_range:
        return None
    cleaned = age_range.replace("سنة", "").replace("سنوات", "").strip()
    parts = cleaned.replace("–", "-").split("-")
    try:
        if len(parts) == 2:
            return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None
    return None


def _age_match_score(chunk_age_range: str | None, child_age: int | None) -> float:
    if child_age is None:
        return 0.0  # no age mentioned in the query -> no boost, no penalty
    bounds = _parse_age_range(chunk_age_range)
    if bounds is None:
        return 0.05  # unspecified/general source: small neutral boost
    low, high = bounds
    span = max(high - low, 1)
    if low <= child_age <= high:
        # Narrower, tightly-matching ranges score higher than broad ones
        # like 0-18 that technically "match" almost anything.
        return 1.0 if span <= 8 else 0.6
    # Small overlap tolerance (off by a year or two) still gets partial credit.
    distance = min(abs(child_age - low), abs(child_age - high))
    if distance <= 2:
        return 0.25
    return -0.3  # clearly irrelevant age group: penalty, not exclusion


def _topic_match_score(chunk_topics: list[str], query_topics: list[str]) -> float:
    if not query_topics:
        return 0.0
    chunk_set = {t.strip() for t in chunk_topics if t.strip()}
    query_set = {t.strip() for t in query_topics}
    overlap = chunk_set & query_set
    if not overlap:
        return 0.0
    return min(len(overlap) / len(query_set), 1.0)


def _priority_boost(priority_stars: int) -> float:
    # 1-5 stars -> 0.0-1.0
    return max(0, min(priority_stars, 5)) / 5.0


def apply_metadata_ranking(
    candidates: list[RetrievedChunk], query_analysis: QueryAnalysis
) -> list[RetrievedChunk]:
    for result in candidates:
        chunk = result.chunk
        age_score = _age_match_score(chunk.age_range, query_analysis.age)
        topic_score = _topic_match_score(chunk.topics, query_analysis.topics)
        priority_score = _priority_boost(chunk.priority_stars)

        result.metadata_score = (
            WEIGHT_AGE_MATCH * age_score
            + WEIGHT_TOPIC_MATCH * topic_score
            + WEIGHT_SOURCE_PRIORITY * priority_score
        )
        result.final_score = (
            WEIGHT_RETRIEVAL_SCORE * result.fused_score + result.metadata_score
        )

    return sorted(candidates, key=lambda r: r.final_score, reverse=True)
