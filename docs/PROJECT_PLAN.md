# Project Plan — AI-Powered Parenting Guidance Platform (*ParentWise*)

> Organized from the original planning notes. This is the **plan of record**: what we set
> out to build, how we scoped it, and who owns what. Where the delivered system diverged
> from the original plan, it is flagged as **[Delivered]**.

---

## 1. Overview

**Product:** *ParentWise* — a general positive-parenting assistant.

**Problem statement.** New and existing parents lack fast, reliable, judgment-free guidance
on everyday parenting challenges (tantrums, discipline, screen time, sibling rivalry) and
instead rely on scattered, inconsistent social-media advice.

**Target audience.** Parents of children aged 0–12, particularly first-time parents.

**Core objective.** Give parents evidence-informed, actionable guidance rooted in recognized
parenting frameworks (positive discipline, attachment parenting, Montessori-style
approaches) — not generic chatbot chat.

The project satisfies every requirement in the brief: **RAG**, **voice input (STT)**,
**fine-tuning + optimization**, a **React/FastAPI** app, and a working **end-to-end
pipeline**.

### Domain boundary (hard rule)

Stay strictly in **behavioral, developmental, and educational** parenting territory
(discipline, communication, sleep routines, screen time, tantrums, milestones). The system
must **not** give medical diagnoses, medication advice, or treat symptoms — that crosses
into the medical domain the brief excludes, and it is not something an LLM should do.

> Baked into the system prompt: *"If a question sounds medical (fever, injury, medication,
> illness), redirect the user to a pediatrician."* Enforced twice — in the fine-tuned
> model's training data and in a deterministic RAG safety gate.

---

## 2. Scope decisions

Deliberately scoped for depth over breadth given a tight timeline:

- **Go deep on 2–3 core topics** rather than covering everything shallowly — *discipline &
  tantrums, screen time, sleep routines*. A defensible scoping decision to state explicitly
  in the report ("depth over breadth given constraints").
- **Curate the RAG sources** to those topics — a few hundred well-chosen chunks, not a huge
  corpus.
- **Align the fine-tuning dataset** to the same topics so RAG and fine-tuning reinforce each
  other instead of covering different ground.
- **Voice:** STT is core scope (speak the question, get a text answer). TTS was planned as a
  stretch goal. **[Delivered]** both STT *and* TTS were implemented (Egyptian-Arabic voice
  in and out).

---

## 3. Architecture

```
User (voice/text) → React frontend
    → FastAPI backend
        → STT (Whisper / faster-whisper) [if voice input]
        → Query understanding (age / topic / risk)
        → Safety gate (deterministic) → redirect high-risk queries
        → RAG retriever (hybrid dense + sparse) → top-k chunks
        → Prompt template (system prompt + few-shot + retrieved context)
        → Fine-tuned LLM (Nile-Chat-4B, QLoRA-adapted, GGUF-quantized) → text response
        → Grounding check
    → React frontend displays response (text)  [+ TTS spoken reply — Delivered]
    → Conversation history stored per session
```

### Component choices

**RAG knowledge base**
- Public-domain / open-access material: UNICEF & WHO parenting guidelines, government
  family-services publications, open parenting-book excerpts, open-access child-development
  papers.
- Topic-tagged chunks: *discipline, sleep, screen time, communication, emotional
  regulation, milestones by age band*.
- Vector DB: **ChromaDB** (dense) alongside **BM25** (sparse), fused with Reciprocal Rank
  Fusion. Metadata filters (child age range, topic) narrow retrieval before the LLM sees it.

**Fine-tuning**
- Q&A dataset of realistic parenting questions + expert-style answers, built
  **semi-synthetically**: draft questions, have a strong LLM draft grounded answers from the
  RAG sources, then **human-review/edit** for quality and tone. Methodology documented
  honestly in the report.
- Base model: an open-source model fine-tunable on a single GPU with QLoRA. **[Delivered]**
  **MBZUAI-Paris/Nile-Chat-4B** (Gemma-3 based, Egyptian-Arabic) — a better dialect fit than
  the originally-listed Llama/Mistral/Qwen candidates.
- Fine-tune for **tone and structure**, not facts — facts come from RAG.

**Optimization**
- **QLoRA (4-bit)** for training itself, plus **post-training quantization to GGUF
  (Q4_K_M)** for CPU inference. Gives a clean before/after story (size, memory, latency,
  quality).

**Voice**
- **STT:** Whisper — **[Delivered]** faster-whisper `small`, Arabic-forced, VAD-tuned.
- **TTS [Delivered]:** NAMAA Egyptian-Arabic voice (Chatterbox-based) via a hosted Gradio
  Space.

---

## 4. Team roles (4 people)

Each member owns a distinct Gen-AI component **plus** app-layer ownership.

### Person 1 — RAG Engineer
- Source, clean, and chunk the knowledge base.
- Build the embedding + vector DB pipeline (ChromaDB / BM25).
- Design retrieval: metadata filtering by age/topic, top-k tuning, hybrid search.
- Evaluate retrieval quality against test queries.

### Person 2 — Fine-Tuning & Optimization Engineer  *(this repo's `finetune+quantization` branch)*
- Build the Q&A fine-tuning dataset (draft questions, LLM-draft grounded answers,
  human-review).
- Run QLoRA fine-tuning on the base model.
- Run the base-vs-fine-tuned comparison (structured eval set + scoring).
- Apply quantization on top of the fine-tuned model; measure size/memory/speed/quality.
- Document training setup, hyperparameters, loss curves.
- **Critical-path / bottleneck role** — dataset-building starts Day 1, in parallel with RAG.

### Person 3 — Prompt Engineering & Frontend Engineer
- Design the system prompt, few-shot examples, and context-aware prompting that ties RAG
  output to the fine-tuned model.
- Run the response-quality side of evaluation (grounded vs. hallucinated).
- Build the React frontend: chat UI, voice controls, conversation history, UX polish.
- Own the end-to-end demo's visual presentation.

### Person 4 — Voice AI & Backend/Integration Engineer
- Integrate Whisper STT (multilingual input, noise handling, latency). **[Delivered]** + TTS.
- Build the FastAPI backend: chat, retrieval, STT, session endpoints.
- Wire STT → prompt → RAG → LLM → response into one pipeline.
- Own conversation history storage and repo setup/deployment instructions.

> Everyone reviews Person 2's fine-tuning Q&A pairs for quality — more eyes, non-sequential.

---

## 5. Timeline (1 week, 4 people in parallel)

Aggressive, minimal slack — everyone starts their own component on Day 1; every step happens
once. If something must slip, let **frontend polish** slip before **fine-tuning quality**.

| Day | Focus |
|---|---|
| **1** | Sync on topics + base model + repo structure. P1 sources docs · P2 drafts Q&A (critical path) · P3 system prompt v1 + React scaffold · P4 Whisper + FastAPI skeleton. |
| **2** | P1 chunk + embed + vector DB, basic retrieval · P2 **finalize dataset** (150–250 clean pairs) · P3 chat UI skeleton · P4 STT→text→placeholder pipeline + core endpoints. |
| **3** | P1 metadata filters, package retrieval module · P2 **kick off QLoRA run — no later** · P3 voice UI + chat styling · P4 STT endpoint + history schema + orchestration stub. |
| **4** | *First integration checkpoint.* P2 monitor training, small base-vs-FT eval, start quantization · P4 first rough end-to-end pass · all fix integration bugs together. |
| **5** | *Full integration.* P2 hand off FT+quantized model · P4 integrate model + wire voice into frontend · highest-risk day. |
| **6** | *Polish + evaluation + report writing starts.* Retrieval tuning, optimization results table, response-quality eval, voice UX, backend hardening. |
| **7** | *Finish + submit.* Combine report, clean repo, finalize README, record demo video, final check against the brief. |

**Risk notes.** Fine-tuning is the tightest step — cut dataset size (even 100–150 pairs)
before pushing its start date. Keep the RAG corpus small (dozens–~100 chunks). Day 4
integration only needs to prove every piece connects; a broken connection found Day 4 is
recoverable, Day 6 is not.

---

## 6. Report checklist (maps to deliverable #1)

- [ ] Problem statement & target audience
- [ ] System architecture diagram
- [ ] Knowledge base & RAG design (sources, chunking, vector DB)
- [ ] Prompt-engineering techniques (with examples)
- [ ] Fine-tuning: dataset construction, training setup, base vs. fine-tuned results
- [ ] Optimization: method + measured effect on size/memory/speed/quality
- [ ] Evaluation results (quantitative + qualitative)
- [ ] Limitations & future improvements (honest — "not a substitute for professional advice")

> The full technical write-up living against this checklist is in
> [`DOCUMENTATION.md`](DOCUMENTATION.md).
