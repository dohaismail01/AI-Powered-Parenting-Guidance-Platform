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
             Claude (generation)
                    ↓
           Grounding Check
                    ↓
        Answer + Sources + Guidance
```

## Setup

```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr tesseract-ocr-ara   # Arabic OCR fallback
export ANTHROPIC_API_KEY=...                            # for generation
```

## Usage

```bash
# 1. Populate data/raw/{pdfs,web,research}/manifest.json
#    (see data/raw/pdfs/manifest.example.json for the shape — this
#    should mirror your existing knowledge-base catalog: organization,
#    title, age_range, topics, priority_stars, license_status)

# 2. Ingest + chunk all sources
python ingest.py

# 3. Build the dense (Chroma) and sparse (BM25) indexes
python build_index.py

# 4. Run the evaluation harness (retrieval + safety metrics)
python evaluate.py

# 5. Serve the API
uvicorn app:app --reload
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
| Generation    | Claude                    | Strong Arabic fluency + safety alignment, appropriate for child-related content |
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
