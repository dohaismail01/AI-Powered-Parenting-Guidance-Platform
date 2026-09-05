"""
ParentWise API.

A thin FastAPI wrapper around ParentWisePipeline — kept deliberately
simple so it's easy to swap for a Streamlit UI, a Slack bot, or embed
into a mobile app's backend without touching the RAG logic itself.

Usage:
    uvicorn app:app --reload

Then:
    POST /ask   {"question": "ابني عنده 5 سنين وبيعمل نوبات غضب"}
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_pipeline import ParentWisePipeline

app = FastAPI(title="ParentWise RAG API")
pipeline: ParentWisePipeline | None = None


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    is_high_risk: bool
    is_grounded: bool | None
    detected_age: int | None
    detected_topics: list[str]
    risk_category: str


@app.on_event("startup")
def load_pipeline() -> None:
    global pipeline
    # Loaded once at startup — embedding/reranker models and the BM25
    # index are expensive to initialize per-request.
    pipeline = ParentWisePipeline()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    response = pipeline.answer(request.question)
    return AskResponse(
        answer=response.answer,
        sources=response.sources,
        is_high_risk=response.is_high_risk,
        is_grounded=response.is_grounded,
        detected_age=response.query_analysis.age,
        detected_topics=response.query_analysis.topics,
        risk_category=response.query_analysis.risk_category,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline_loaded": pipeline is not None}
