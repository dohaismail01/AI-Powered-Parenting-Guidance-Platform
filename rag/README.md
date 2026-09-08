# ParentWise — RAG Component

Arabic, evidence-grounded parenting assistant. This is the RAG part
only (retrieval + generation + safety), matching the architecture in
`ParentWise_RAG_Pipeline.md`.

## Pipeline

```
Query → Query Understanding (age/topic/risk) → Safety Gate
                                                   │
                    ┌──────────────────────────────┴─────────┐
                    ↓                                        ↓
              Safe query                              High-risk query
                    ↓                                        ↓
        Hybrid Retrieval (BGE-M3 + BM25)              Safety Response
                    ↓
                   RRF
                    ↓
        Metadata-Aware Ranking (age/topic/priority)
                    ↓
          BGE-reranker-v2-m3 (top 5-8 chunks)
                    ↓
           Context Construction
                    ↓
   Generation (fine-tuned Nile-Chat-4B GGUF)
                    ↓
           Grounding Check
                    ↓
        Answer + Sources + Guidance
```

## Generation backend

Generation uses the project's **fine-tuned `Nile-Chat-4B` model**, quantized to a
`Q4_K_M` GGUF. The provider is pluggable via `GENERATION_PROVIDER` in `config.py`:

| Provider | Default | What it is |
|---|---|---|
| `nile_chat_ollama` | ✅ default | Same GGUF served through **Ollama** (`OLLAMA_MODEL`, `OLLAMA_NUM_GPU`). Faster than the raw CPU wheel; use GPU offload where the machine's Ollama CUDA build supports it. |
| `nile_chat_gguf` | | The GGUF run in-process via `llama-cpp-python` (no external daemon; slow on a no-BLAS CPU wheel). |
| `anthropic` | | Claude API — comparison/fallback baseline only. Needs `ANTHROPIC_API_KEY`. |

No API key is required for the default (local) path.

## Setup

Install deps and download the models + build the indexes. `setup_rag.py` is a
**resumable** helper that pulls `bge-m3`, the reranker, and the GGUF, verifies the
big archive, then builds the Chroma + BM25 indexes:

```bash
pip install -r requirements.txt
# NOTE: pin FlagEmbedding==1.3.5 — 1.4.x passes a `dtype` kwarg that breaks with
# the transformers version here (XLMRobertaModel got an unexpected 'dtype').
# Optional, only for (re)ingesting scanned PDFs:
#   sudo apt-get install tesseract-ocr tesseract-ocr-ara   # Arabic OCR fallback

# one-time: download models (resumable) + build indexes
py -3.11 setup_rag.py           # add --no-gguf to skip the 2.5 GB GGUF (then use anthropic)
```

`setup_rag.py` prints the local snapshot paths to export so the server reuses the
downloaded models instead of re-fetching them:

```bash
export EMBEDDING_MODEL_PATH=...bge-m3 snapshot dir
export RERANKER_MODEL_PATH=...bge-reranker-v2-m3 snapshot dir
```

## Usage

```bash
# The corpus is already ingested (data/processed/chunks/chunks.json, 944 chunks).
# To rebuild from source instead of using setup_rag.py:
#   python ingest.py           # 1. populate data/raw/{pdfs,web,research}/manifest.json first
#   python build_index.py      # 2. build dense (Chroma) + sparse (BM25) indexes
#   python evaluate.py         # 3. retrieval + safety metrics

# Serve the API (with the two *_MODEL_PATH vars exported)
py -3.11 -m uvicorn app:app --port 8000
# POST /ask {"question": "ابني عنده 5 سنين وبيعمل نوبات غضب"}
```

## Why these choices

| Component     | Choice                  | Why |
|---------------|--------------------------|-----|
| PDF extraction| PyMuPDF                  | Fast, handles Arabic/RTL text, gives per-page text for citations |
| OCR fallback  | Tesseract (`ara`)        | Free, offline; swap for cloud OCR if scan quality demands it |
| Web extraction| trafilatura               | Strips nav/ads/boilerplate far better than a raw HTML dump |
| Preprocessing | Custom Arabic normalizer  | Unifies alef/yaa forms, strips diacritics/kashida — measurably improves BM25 + embedding recall on Arabic |
| Chunking      | Structure-aware, 300-500 tokens | Preserves headings/sections so retrieval never returns an orphaned sentence |
| Embeddings    | BAAI/bge-m3               | Multilingual, strong on Arabic (MIRACL), open-source, long context |
| Vector DB     | Chroma                    | Simple local setup for an educational project (swap for Qdrant at scale) |
| Sparse        | BM25 (rank_bm25)          | Catches exact terminology dense embeddings can dilute |
| Fusion        | Reciprocal Rank Fusion    | Combines two incomparable score scales without hand-tuned weights |
| Reranker      | BAAI/bge-reranker-v2-m3   | Cross-encoder precision pass before the LLM sees the evidence |
| Generation    | Fine-tuned Nile-Chat-4B (GGUF) | The parenting-tuned model itself, trained on this knowledge base; Egyptian-Arabic tone + safety behavior. Claude kept only as a comparison baseline. |
| Safety        | Rules + classifier gate   | Deterministic interception before generation, not just a prompt instruction |

## Notes

- `src/generation/generator.py` includes a placeholder for a local
  Qwen-family model (`local_qwen`) if you need a fully offline,
  no-API-cost deployment for the educational version of the project.
- The grounding check in `generator.py` is a cheap vocabulary-overlap
  heuristic. Once `evaluate.py` has enough labeled data, replace it
  with a proper NLI/faithfulness metric (e.g. RAGAS).
- License status (`green`/`yellow`/`red`) travels with every chunk's
  metadata. Wire a filter into `dense_retriever.search()` /
  `bm25_retriever.search()` (Chroma's `where=...`) to exclude `red`
  sources from the corpus until permission is obtained, rather than
  relying on manual curation before ingestion.
