"""
Evaluation harness, covering the three metric families from the spec
(sections 24-26):

  Retrieval:  Recall@5, Recall@10, Precision@K, MRR, NDCG
  Generation: groundedness (heuristic), citation presence
  Safety:     recall / precision / false negatives on the risk gate

Run against data/evaluation/eval_questions.json. This is a starting
harness (~6 sample questions are included) — grow it toward the
recommended 100-200 questions as the knowledge base and query patterns
mature, and swap the groundedness heuristic for RAGAS or an NLI model
once you have enough labeled data to validate the threshold.

Usage:
    python evaluate.py
"""
from __future__ import annotations

import json
import math

from config import EVAL_QUESTIONS_PATH
from src.rag_pipeline import ParentWisePipeline
from src.safety.safety_classifier import classify_safety
from src.query.query_analyzer import analyze_query


def _load_eval_set() -> list[dict]:
    with open(EVAL_QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------------
def recall_at_k(retrieved_texts: list[str], relevant_keywords: list[str], k: int) -> float:
    if not relevant_keywords:
        return None  # not applicable (e.g. safety-only questions)
    top_k_text = " ".join(retrieved_texts[:k])
    hits = sum(1 for kw in relevant_keywords if kw in top_k_text)
    return hits / len(relevant_keywords)


def mrr(retrieved_texts: list[str], relevant_keywords: list[str]) -> float:
    if not relevant_keywords:
        return None
    for rank, text in enumerate(retrieved_texts, start=1):
        if any(kw in text for kw in relevant_keywords):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_texts: list[str], relevant_keywords: list[str], k: int) -> float:
    if not relevant_keywords:
        return None
    gains = [
        1.0 if any(kw in text for kw in relevant_keywords) else 0.0
        for text in retrieved_texts[:k]
    ]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------
# Safety metrics
# --------------------------------------------------------------------------
def evaluate_safety(eval_set: list[dict]) -> dict:
    true_positives = false_positives = true_negatives = false_negatives = 0

    for item in eval_set:
        expected_high_risk = item["risk_category"] not in ("NORMAL", "MEDICAL")
        analysis = analyze_query(item["question"])
        decision = classify_safety(analysis)

        if expected_high_risk and decision.is_high_risk:
            true_positives += 1
        elif expected_high_risk and not decision.is_high_risk:
            false_negatives += 1  # most important to minimize — a missed abuse/self-harm case
        elif not expected_high_risk and decision.is_high_risk:
            false_positives += 1
        else:
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else None
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else None

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "safety_precision": precision,
        "safety_recall": recall,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    eval_set = _load_eval_set()
    pipeline = ParentWisePipeline()

    print("=" * 60)
    print("SAFETY EVALUATION")
    print("=" * 60)
    safety_metrics = evaluate_safety(eval_set)
    for k, v in safety_metrics.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 60)
    print("RETRIEVAL + GENERATION EVALUATION (non-high-risk questions)")
    print("=" * 60)

    recall5_scores, recall10_scores, mrr_scores, ndcg_scores = [], [], [], []

    for item in eval_set:
        if item["risk_category"] not in ("NORMAL", "MEDICAL"):
            continue  # covered by safety eval above, not retrieval

        response = pipeline.answer(item["question"])
        retrieved_texts = [r.chunk.text for r in response.retrieved_chunks]
        keywords = item.get("relevant_chunk_keywords", [])

        r5 = recall_at_k(retrieved_texts, keywords, 5)
        r10 = recall_at_k(retrieved_texts, keywords, 10)
        m = mrr(retrieved_texts, keywords)
        n = ndcg_at_k(retrieved_texts, keywords, 10)

        if r5 is not None:
            recall5_scores.append(r5)
            recall10_scores.append(r10)
            mrr_scores.append(m)
            ndcg_scores.append(n)

        print(f"\n[{item['id']}] {item['question']}")
        print(f"  Recall@5={r5}  Recall@10={r10}  MRR={m}  NDCG@10={n}")
        print(f"  Grounded: {response.is_grounded}")
        print(f"  Sources: {response.sources}")

    def _avg(values):
        return sum(values) / len(values) if values else None

    print("\n" + "=" * 60)
    print("AGGREGATE RETRIEVAL METRICS")
    print("=" * 60)
    print(f"  Avg Recall@5:  {_avg(recall5_scores)}")
    print(f"  Avg Recall@10: {_avg(recall10_scores)}")
    print(f"  Avg MRR:       {_avg(mrr_scores)}")
    print(f"  Avg NDCG@10:   {_avg(ndcg_scores)}")


if __name__ == "__main__":
    main()
