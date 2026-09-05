"""
End-to-end ParentWise RAG pipeline, matching the architecture in
ParentWise_RAG_Pipeline.md section 27:

    Query -> Query Understanding -> Safety Check
        -> [high risk]  -> Safety Response
        -> [safe]       -> Hybrid Retrieval (dense + BM25) -> RRF
                        -> Metadata-Aware Ranking -> Reranker
                        -> Context Construction -> LLM
                        -> Grounding Check -> Answer + Sources
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.generation.generator import check_grounding, generate_answer
from src.generation.prompt import format_source_line
from src.query.query_analyzer import QueryAnalysis, analyze_query
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.metadata_ranker import apply_metadata_ranking
from src.retrieval.reranker import rerank
from src.safety.safety_classifier import classify_safety
from src.schema import RetrievedChunk


@dataclass
class ParentWiseResponse:
    answer: str
    query_analysis: QueryAnalysis
    sources: list[str] = field(default_factory=list)
    is_high_risk: bool = False
    is_grounded: bool | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)


class ParentWisePipeline:
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.load()  # expects build_index.py to have run already

    def answer(self, raw_query: str) -> ParentWiseResponse:
        # 1) Query understanding
        query_analysis = analyze_query(raw_query)

        # 2) Safety gate — runs *before* any retrieval/generation
        safety_decision = classify_safety(query_analysis)
        if safety_decision.is_high_risk:
            return ParentWiseResponse(
                answer=safety_decision.safe_response,
                query_analysis=query_analysis,
                sources=[],
                is_high_risk=True,
            )

        # 3) Hybrid retrieval
        dense_results = self.dense_retriever.search(raw_query)
        bm25_results = self.bm25_retriever.search(raw_query)
        fused = reciprocal_rank_fusion(dense_results, bm25_results)

        # 4) Metadata-aware ranking (age / topic / source priority)
        ranked = apply_metadata_ranking(fused, query_analysis)

        # 5) Cross-encoder reranking down to the final evidence set
        top_chunks = rerank(raw_query, ranked)

        # 6) Generation, grounded in the top chunks
        answer_text = generate_answer(query_analysis, top_chunks)

        # 7) Lightweight grounding check
        is_grounded = check_grounding(answer_text, top_chunks)
        if not is_grounded and top_chunks:
            answer_text += (
                "\n\n(تنويه: بعض أجزاء هذه الإجابة قد لا تكون مدعومة بشكل "
                "كافٍ بالمصادر المسترجعة — يُفضّل التحقق من مصدر رسمي إضافي.)"
            )

        sources = [format_source_line(r) for r in top_chunks]
        # de-duplicate while preserving order
        seen = set()
        unique_sources = [s for s in sources if not (s in seen or seen.add(s))]

        return ParentWiseResponse(
            answer=answer_text,
            query_analysis=query_analysis,
            sources=unique_sources,
            is_high_risk=False,
            is_grounded=is_grounded,
            retrieved_chunks=top_chunks,
        )
