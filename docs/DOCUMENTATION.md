# Technical Documentation — ParentWise

Arabic-first, evidence-grounded parenting assistant. This document is the technical report
for the whole platform and is organized against the deliverable report checklist.

**Contents**
1. [Problem statement & audience](#1-problem-statement--audience)
2. [System architecture](#2-system-architecture)
3. [Knowledge base & RAG design](#3-knowledge-base--rag-design)
4. [Prompt engineering](#4-prompt-engineering)
5. [Fine-tuning](#5-fine-tuning)
6. [Optimization (quantization)](#6-optimization-quantization)
7. [Voice (STT + TTS)](#7-voice-stt--tts)
8. [Evaluation](#8-evaluation)
9. [Limitations & future work](#9-limitations--future-work)

---

## 1. Problem statement & audience

Parents of children aged 0–12 — especially first-time parents — need fast, reliable,
judgment-free guidance on everyday parenting challenges and currently rely on scattered,
inconsistent social-media advice. ParentWise gives evidence-informed, actionable guidance in
**Egyptian colloquial Arabic**, grounded in recognized parenting frameworks and authoritative
sources (UNICEF, WHO, peer-reviewed Arabic research).

**Deliberate domain boundary.** The system stays in behavioral / developmental / educational
parenting and **refuses** medical diagnosis, medication dosing, and emergency self-treatment,
redirecting those to a professional. This boundary is enforced in two independent places
(the fine-tuned model's training and a deterministic RAG safety gate) so a single failure
does not let an unsafe answer through.

---

## 2. System architecture

```
User (voice/text)
  → React frontend
    → Voice service (FastAPI):  STT (faster-whisper)  ──▶ Arabic text
    → RAG API (FastAPI):
         Query understanding (age · topic · risk)
           → Safety gate ─[high risk]▶ Safety response (→ doctor/ER)
                        └─[safe]▶ Hybrid retrieval (BGE-M3 dense + BM25 sparse)
                                   → Reciprocal Rank Fusion
                                   → Metadata-aware ranking (age/topic/priority)
                                   → Reranker (bge-reranker-v2-m3, top 5–8)
                                   → Context construction (structured, per-source)
                                   → Generation (fine-tuned Nile-Chat-4B, GGUF)
                                   → Grounding check
         → Answer + cited sources
  → React frontend (text)  +  Voice service TTS (NAMAA Egyptian voice) → spoken reply
```

Components are decoupled and communicate over HTTP: the voice service calls the RAG API
(`RAG_API_URL`), and the RAG API's generation layer is pluggable (`nile_chat_gguf` by
default; Claude available as a comparison/fallback). This lets each engineer build and test
their component independently.

---

## 3. Knowledge base & RAG design

**Component:** [`rag/`](../rag/) · owner: RAG Engineer.

### Sources
Curated, authoritative, license-clean material catalogued in `data/raw/{pdfs,web,research}/
manifest.json`, each entry carrying `organization`, `title`, `age_range`, `topics`,
`priority_stars`, `license_status`, and `evidence_level`. Primary sources include **UNICEF
Egypt** (positive-parenting toolkit, early-childhood development), **WHO**, and peer-reviewed
Arabic research.

### Ingestion & preprocessing
| Stage | Choice | Why |
|---|---|---|
| PDF extraction | PyMuPDF | Fast, handles Arabic/RTL, per-page text for citations |
| OCR fallback | Tesseract (`ara`) | Free, offline, for scanned PDFs |
| Web extraction | trafilatura | Strips nav/ads/boilerplate |
| Arabic normalization | custom | Unifies alef/yaa forms, strips diacritics/kashida — measurably improves BM25 + embedding recall |
| Chunking | structure-aware, ~400 tokens (300–500), 50 overlap | Preserves headings/sections; no orphaned sentences |

### Retrieval (hybrid)
1. **Dense** — `BAAI/bge-m3` embeddings (multilingual, strong on Arabic/MIRACL, 1024-dim),
   stored in **Chroma**. Top-25.
2. **Sparse** — **BM25** (`rank_bm25`), catches exact terminology dense vectors dilute. Top-25.
3. **Fusion** — Reciprocal Rank Fusion (`RRF_K=60`) combines the two incomparable score
   scales without hand-tuned weights.
4. **Metadata-aware ranking** — small additive boosts for age match (0.15), topic match
   (0.10), source priority (0.08), kept small so an authoritative-but-irrelevant chunk can't
   outrank a relevant one.
5. **Reranker** — `BAAI/bge-reranker-v2-m3` cross-encoder, final **top 5–8** chunks to the LLM.

### Safety gate
A deterministic rules + classifier gate runs **before** any retrieval/generation. High-risk
queries (emergencies, medical, dosing) short-circuit to a safety response instead of being
answered — interception, not just a prompt instruction.

### Grounding check
After generation, a grounding check verifies the answer is supported by the retrieved
context; answers carry their cited sources.

---

## 4. Prompt engineering

**Component:** [`rag/src/generation/prompt.py`](../rag/src/generation/prompt.py).

The system prompt is built in layers, deliberately starting from the **fine-tuning team's
exact base system message** (the model was tuned against it; a reworded/formal-MSA prompt
would fight the training):

1. **Base system message (verbatim from fine-tuning).** Warm, supportive, non-judgmental
   Egyptian-Arabic persona; practical, concise advice; redirect health/safety/development
   questions to a pediatrician.
2. **RAG additions** (same colloquial register): answer only from retrieved context; admit
   when context is insufficient instead of guessing; distinguish scientific evidence from
   general opinion (don't turn a single-source *correlation* into *causation*); cite sources
   from the attached metadata only (no invented citations); respect the child's age.
3. **Chain-of-thought** — a *silent* reasoning pass (never shown to the user) that checks
   relevance, age-fit, context sufficiency, and medical/risk redirection before writing.
4. **Few-shot** — three worked examples covering the three response shapes: grounded answer
   with citation, medical-boundary redirect, insufficient-context admission.

Context is injected **structured** — each chunk keeps its source/organization/section/page
so the model cites accurately rather than inventing references.

---

## 5. Fine-tuning

**Component:** [`finetune-quantization/`](../finetune-quantization/) · owner: Fine-Tuning &
Optimization Engineer.

**Principle: teach behavior, not facts.** Fine-tuning shapes *how* the model answers (warm
Egyptian-Arabic tone, structure, safe refusals); facts come from RAG at answer time.

### Base model
**`MBZUAI-Paris/Nile-Chat-4B`** — a Gemma-3-based model already oriented to Egyptian Arabic,
a stronger dialect fit than generic Llama/Mistral/Qwen candidates.

### Dataset (507 pairs)
Semi-synthetic, human-reviewed, grounded in the RAG source chunks:
- **483 grounded parenting pairs** mined from the chunks (development, feeding, discipline,
  positive parenting, socialization, digital habits, …).
- **24 safety/refusal pairs** teaching required behavior on emergencies (→ ER/doctor now),
  dosages (no numbers → doctor/pharmacist), dangerous myths (clear "no" + why),
  no-diagnosis, caregiver crisis, and out-of-scope questions.
- Each pair carries `source_text` for provenance. Split **482 train / 25 val** (95/5, seed
  42) in chat format; a **28-item held-out test set** with **0 overlap** and per-item rubrics.

### Training setup (QLoRA)
Base model frozen and loaded in **4-bit**; lightweight **LoRA adapters** trained (< 1% of
parameters). Key config: **LoRA r=16, α=16, dropout 0.05, LR 1e-4, 2 epochs**,
`train_on_responses_only`, keep-best-checkpoint by validation loss. Run on a single free GPU
(Kaggle T4) — see `notebooks/nile_chat_finetune_kaggle.ipynb`.

Small-dataset overfitting is guarded deliberately: low LR, only 2 epochs, dropout, and
best-by-val-loss selection.

---

## 6. Optimization (quantization)

Post-training quantization to **GGUF `Q4_K_M`** makes the fine-tuned model practical for
local, CPU-based deployment alongside the RAG stack.

| | Before | After |
|---|---|---|
| Format | 16-bit | GGUF Q4_K_M |
| Size | ~8 GB | ~2.5 GB (**~3× smaller**) |
| Footprint | GPU-oriented | CPU-capable |
| Inference | — | ~**9.2 tok/s** on a laptop CPU (greedy) |

The quantized model is the RAG layer's default generator (`nile_chat_gguf`), runnable via
llama.cpp / Ollama — no GPU required at serving time.

---

## 7. Voice (STT + TTS)

**Component:** [`voice_service/`](../voice_service/) · FastAPI. Full testing notes:
[`voice_service/docs/testing_documentation.md`](../voice_service/docs/testing_documentation.md).

### Speech-to-text (core)
**`faster-whisper` (small)** — CTranslate2-optimized Whisper; same accuracy as
`openai-whisper`, meaningfully faster on CPU (int8). `small` balances dialectal-Arabic
accuracy against CPU latency (`tiny`/`base` weren't accurate enough on colloquial input).
Language **forced to Arabic** (auto-detect misfired on short clips), VAD enabled to avoid
hallucinating on silence.

> **Tuned via testing, not assumed:** `min_silence_duration_ms` was raised from the library
> default of 500 → **1500 ms** because a natural half-second mid-sentence pause was being
> read as end-of-speech, silently truncating longer sentences.

**Known gap:** the model tends to "correct" colloquial input toward MSA. Retrieval mostly
needs topic keywords to survive, but a dialect→MSA vocabulary mismatch is flagged for the
RAG team; a dialect-specific Whisper variant is a possible drop-in upgrade.

### Text-to-speech (bonus)
**`NAMAA-Egyptian-Voice`** (Chatterbox-based, Egyptian-Arabic), called via `gradio_client`
against its Hugging Face Space. Chosen over Coqui **XTTS-v2**, which produced a
Levantine-leaning accent regardless of the Egyptian reference sample. Running Chatterbox
locally hit irreconcilable `transformers` version conflicts, so a remote Space call
sidesteps the dependency mess. **Trade-off:** depends on a third-party Space and its shared
free-GPU quota — mitigated by optional HF-token auth and Space duplication, and a TTS
startup failure does **not** take down the core `/stt` endpoint.

---

## 8. Evaluation

### Fine-tuned model — held-out test (28 questions, rubric-graded)
Each question has a rubric (`must_include` / `must_avoid`, `safety_critical`); safety-critical
items are hard fails. Greedy decoding, run on the quantized GGUF model.

| Area | Result |
|---|---|
| Safety-critical subset | **8 / 12** passed |
| Overall | **~67%** |
| Speed | **~9.2 tok/s** (laptop CPU) |

**What worked:** warm, on-persona Egyptian-Arabic tone; correct refusals on the FGM myth,
antibiotics, and cold-bath myth; emergencies routed to doctor/ER; runs on CPU, deployable
with the RAG stack.

**What failed (honest read):** a dosage question produced a numeric answer; a choking
scenario was not escalated as an emergency; occasional wrong facts (solids age, screen
time); some out-of-scope questions were answered instead of declined.

### RAG — retrieval & safety
`rag/evaluate.py` runs a retrieval + safety harness over `data/evaluation/eval_questions.json`
(retrieval precision against test queries; safety-gate interception on high-risk queries).

### Response quality
Grounded-vs-hallucinated scoring of end-to-end answers (owned by the Prompt/Frontend
engineer), checking that answers stay within retrieved context and cite sources.

---

## 9. Limitations & future work

**This is an educational project and not a substitute for professional medical or
psychological advice.**

**Limitations**
- Safety refusals are strong but not perfect (dosage/choking misses above) — the single most
  important area to harden.
- Facts are only as good as the retrieved sources and the retrieval quality.
- Topic coverage is intentionally narrow (discipline/tantrums, screen time, sleep).
- STT can drift colloquial → MSA; TTS depends on a third-party hosted Space.

**Future work**
1. **Expand** the safety dataset (more refusal pairs across emergencies/dosing/out-of-scope).
2. **Strengthen** refusal training so safety-critical items pass reliably.
3. **Ground** all factual claims through RAG rather than model memory.
4. Broaden topic and age-band coverage.
5. Integrate a dialect-specific STT model; duplicate the TTS Space for reliable throughput.
