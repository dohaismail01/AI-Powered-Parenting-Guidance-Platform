"""
Dense retrieval: BAAI/bge-m3 embeddings stored in Chroma.

Why bge-m3: strong multilingual model that is specifically competitive
on Arabic (MIRACL benchmark), open-source (no per-call API cost, can
run locally), and supports long inputs (up to 8192 tokens) so a full
400-500 token chunk embeds cleanly in one pass.

Why Chroma: simplest local vector DB to stand up for an educational
project; swap for Qdrant (see qdrant_store.py-style client, same
interface) if you need production-grade metadata filtering at scale —
the rest of the pipeline doesn't need to change, only this file.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings
from FlagEmbedding import BGEM3FlagModel

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    DENSE_TOP_K,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
)
from src.schema import Chunk, RetrievedChunk

_model: BGEM3FlagModel | None = None


def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel(
            EMBEDDING_MODEL_NAME,
            use_fp16=(EMBEDDING_DEVICE == "cuda"),
            device=EMBEDDING_DEVICE,
        )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    output = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return output["dense_vecs"].tolist()


class DenseRetriever:
    def __init__(self, persist_directory: str = str(CHROMA_DIR)):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, chunks: list[Chunk]) -> None:
        """Embed and upsert a batch of chunks into the vector store."""
        if not chunks:
            return
        embeddings = embed_texts([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )

    def search(self, query: str, top_k: int = DENSE_TOP_K) -> list[RetrievedChunk]:
        query_embedding = embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved: list[RetrievedChunk] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        # Chroma returns cosine *distance*; convert to a similarity score.
        distances = results["distances"][0]

        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            similarity = 1 - distance
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=meta["doc_id"],
                text=text,
                organization=meta["organization"],
                title=meta["title"],
                section=meta["section"] or None,
                page=meta["page"] if meta["page"] != -1 else None,
                age_range=meta["age_range"] or None,
                topics=meta["topics"].split(",") if meta["topics"] else [],
                priority_stars=meta["priority_stars"],
                license_status=meta["license_status"],
                url=meta["url"] or None,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, dense_score=similarity))
        return retrieved
