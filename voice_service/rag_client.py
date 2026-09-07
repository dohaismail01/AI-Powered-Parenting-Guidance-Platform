"""
Client for the teammate's ParentWise RAG API (rag/app.py) — retrieval +
metadata ranking + reranking + the fine-tuned LLM, all wrapped behind
a simple POST /ask endpoint on their own server.

Kept as a plain HTTP client (not a Python import of their code) so the
two teams' services stay independently deployable: their server can
move, restart, or scale separately from this one.
"""

import logging
from dataclasses import dataclass

import httpx

import config

logger = logging.getLogger("parentwise-stt")


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]
    is_high_risk: bool
    is_grounded: bool | None


class RagClient:
    """Thin wrapper around the teammate's RAG API's POST /ask endpoint."""

    def __init__(self, base_url: str = config.RAG_API_URL):
        self.base_url = base_url.rstrip("/")

    def ask(self, question: str) -> RagAnswer:
        url = f"{self.base_url}/ask"
        with httpx.Client(timeout=config.RAG_API_TIMEOUT_SECONDS) as client:
            response = client.post(url, json={"question": question})
            response.raise_for_status()
            data = response.json()

        return RagAnswer(
            answer=data["answer"],
            sources=data.get("sources", []),
            is_high_risk=data.get("is_high_risk", False),
            is_grounded=data.get("is_grounded"),
        )

    def health_check(self) -> bool:
        """Returns True if the RAG API is reachable and its pipeline is loaded."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json().get("pipeline_loaded", False)
        except Exception:
            logger.exception("RAG API health check failed")
            return False
