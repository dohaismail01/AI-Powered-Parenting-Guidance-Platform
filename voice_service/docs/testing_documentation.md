# ParentWise — Voice Component (STT + TTS)
### Testing & Design Documentation

**Component owner:** Jasmine
**Scope:** Speech-to-text (Arabic, Egyptian dialect) and text-to-speech (Egyptian dialect)

---

## 1. Speech-to-Text (STT)

### 1.1 Model chosen

**`faster-whisper`** (small), a CTranslate2-optimized reimplementation of OpenAI's Whisper.

**Why this over plain `openai-whisper`:** same accuracy, meaningfully faster inference — important for a live voice-input UX where the parent is waiting on a response.

**Why `small` over `tiny`/`base`/`medium`:** `small` was chosen as the balance point between accuracy on dialectal Arabic (larger models handle dialect variation better) and inference speed on CPU (no GPU available in this environment). `tiny`/`base` were not accurate enough on Egyptian colloquial input in early testing; `medium` was viable but slower with no clear accuracy gain for short parenting-question utterances.

### 1.2 Configuration decisions (and why)

| Setting | Value | Reasoning |
|---|---|---|
| `language` | forced to `"ar"` | Auto-detect occasionally misidentified short/ambiguous clips as other languages; forcing Arabic removed that failure mode entirely. |
| VAD (`vad_filter`) | enabled | Prevents the model from hallucinating text on silence — important since the frontend can't guarantee a clean start/stop of recording. |
| `min_silence_duration_ms` | **1500** (raised from the library default of 500) | **Found via testing, not assumed.** At the default 500ms, the model was cutting sentences short — a natural mid-sentence pause of half a second was being read as "end of speech," so longer sentences came back truncated. Testing across several recordings showed 1500ms eliminated the truncation without introducing new problems. |

### 1.3 Known limitation: dialect vs. Modern Standard Arabic (MSA)

The RAG content sources for this project (UNICEF, WHO, UNICEF Egypt) are written in MSA, but real users are expected to ask questions in Egyptian colloquial Arabic. Testing showed `faster-whisper-small` has a tendency to "correct" colloquial input toward MSA — e.g. dialectal wording sometimes came back transcribed in a more formal register than what was actually said.

**Practical impact:** this doesn't necessarily break the pipeline, since retrieval mainly needs the topic-carrying keywords to survive, not word-for-word dialect fidelity — but it's a known accuracy gap worth flagging for the RAG team, since a dialect→MSA vocabulary mismatch could affect retrieval quality on their end too.

**A dialect-specific model** (`moeshawky/faster-whisper-small-egyptian-arabic`, same `faster-whisper` interface) was identified as a possible drop-in upgrade if dialect accuracy becomes a priority; not yet integrated into the main pipeline.

### 1.4 Test results

Testing methodology: recorded short parenting-related utterances covering the project's core topics (sleep, food refusal, tantrums, screen time, sibling conflict, speech development, independence), split between MSA and Egyptian colloquial phrasing, and compared the transcript against what was actually said.

| # | Register | Spoken (intended) | Result |
|---|---|---|---|
| 1 | MSA | "ابني مش بينام كويس بالليل" | ✅ Correct after VAD fix (initially returned empty — see §1.2) |
| 2 | Egyptian colloquial | "ابني عنيد قوي ومش بيسمع الكلام خالص" | ✅ Correct |
| 3 | Mixed / longer sentence | "ولادي بيتخانقوا كل شوية مع بعض" | ✅ Correct after raising `min_silence_duration_ms` to 1500 (previously truncated) |

**Note for report:** the above is a small, illustrative sample from development testing rather than a formal accuracy benchmark. A larger, systematic test set (10-15+ utterances, MSA and colloquial, scored for exact/near/failed match) is recommended before reporting a specific accuracy percentage — see `docs/eval_template.md` for a ready-to-fill template.

### 1.5 Issues encountered during setup (useful for the report's "challenges" section)

- **Missing `requests` dependency**: `faster-whisper` imports `requests` internally but it wasn't pinned in the initial `requirements.txt`; caused a `ModuleNotFoundError` on a clean install. Fixed by adding it explicitly.
- **`uvicorn` not on PATH on Windows** after a user-level pip install; resolved by running via `python -m uvicorn ...` instead of the bare `uvicorn` command.
- **VAD truncation** (see §1.2) — the single most impactful bug found, since it silently dropped the second half of longer sentences with no error or warning.

---

## 2. Text-to-Speech (TTS) — bonus feature

### 2.1 Model chosen

**`NAMAA-Egyptian-Voice`** (Chatterbox-based, fine-tuned specifically for Egyptian Arabic), accessed via its public Hugging Face Space through `gradio_client` rather than run locally.

### 2.2 Why this over the initial choice (XTTS-v2)

Coqui **XTTS-v2** was tried first (it officially supports Arabic). However, testing showed it produced a Levantine/Syrian-leaning accent regardless of the Egyptian reference voice sample provided — XTTS's Arabic training data isn't dialect-specific, so it generalizes toward whichever Arabic variety dominates its training set rather than matching the reference speaker's dialect.

Switching to `NAMAA-Egyptian-Voice` — trained specifically on Egyptian Arabic — resolved this; output was confirmed to sound natively Egyptian.

### 2.3 Why a remote API instead of a locally-hosted model

Running Chatterbox-based models locally in this project's environment (Colab, then Windows) repeatedly hit **transformers version conflicts**: different TTS libraries (Coqui TTS, chatterbox-tts) require mutually incompatible `transformers` versions, and each fix for one broke the other. Rather than vendoring a fragile local install, the integration calls the model's public Gradio Space over HTTP via `gradio_client`, which sidesteps all local dependency conflicts.

**Trade-off (documented deliberately, not hidden):** this makes the TTS feature dependent on a third-party Space staying online and responsive, and the public Space has a shared rate limit (Hugging Face's "ZeroGPU" free quota), which was hit once during testing. Two mitigations are in place:
1. `config.py` supports authenticating with a personal Hugging Face token (`token=` parameter) for a higher quota.
2. The Space can be **duplicated** to a private, unshared copy for production-level reliability — recommended if this becomes a graded/demoed feature under real usage.
3. In the server code (`stt_service.py`), a TTS connection failure at startup does **not** take down the STT service — `/stt` (the required, core feature) keeps working even if `/tts` (bonus) is unavailable.

### 2.4 Issues encountered

- `hf_token` → `TypeError: unexpected keyword argument`: the `gradio_client` library renamed this parameter to `token` in recent versions; the older parameter name is undocumented outside older tutorials.
- Sentences with a comma sometimes came back truncated at the comma; removing punctuation or splitting into shorter clauses produced complete audio reliably.
- ZeroGPU quota exceeded on the public Space during testing — resolved by authenticating with a personal HF token (see §2.3).

### 2.5 Test result

| Input text (Egyptian colloquial) | Reference voice | Result |
|---|---|---|
| "ابني مش بيسمع الكلام خالص" | User's own recorded voice sample | ✅ Complete, natively Egyptian-accented output, confirmed by listening |

---

## 3. Integration status

- Both `/stt` (POST, audio → text) and `/tts` (POST, text → audio) are implemented as endpoints on a single FastAPI service (`stt_service.py`), tested locally, and confirmed working end-to-end.
- **Not yet done:** wiring this service into the team's RAG/LLM pipeline (still in progress — chunking of source material was completed by a teammate; embeddings/retrieval and the fine-tuned LLM are next). The `/stt` → RAG → LLM → `/tts` handoff contract (request/response shape between services) still needs to be agreed on with the rest of the team.

### 3.1 Observation from reviewing the team's chunked source data

The chunked RAG source data (UNICEF Egypt content, ~189 chunks reviewed) is entirely in Modern Standard Arabic, matching the register of the original guide documents. This confirms the dialect gap noted in §1.3 is a real, concrete risk for retrieval quality, not just a theoretical one: a colloquial-phrased question transcribed by STT (e.g. "ابني مش بيسمع الكلام") may not lexically match MSA-phrased chunks (e.g. "طفلي لا يستجيب للتوجيهات") even when they're about the same topic, depending on how the embedding model handles the dialect/MSA gap.

This was flagged to the teammate responsible for chunking/retrieval before the vector database and embeddings are built, since it's cheaper to account for at that stage (e.g. via a dialect-aware embedding model, or query normalization before retrieval) than after.

**Update:** the vector database has since been built (944 chunks, Chroma, `BAAI/bge-m3` embeddings, cosine similarity, with a complementary BM25 keyword index for hybrid search). Retrieval was tested standalone (§4) and the dialect gap did not prevent relevant results from surfacing — `bge-m3`'s multilingual training appears to bridge colloquial-to-MSA reasonably well at the meaning level, though this is based on spot-checking a handful of queries, not a systematic evaluation.

---

## 4. Retrieval testing (standalone, no LLM)

**Setup:** queried the team's Chroma vector store directly with Egyptian-colloquial questions, using the same embedding model it was built with (`BAAI/bge-m3`) to keep query and stored embeddings comparable.

**Test queries and results (top-3 chunks per query):**

| Query (colloquial) | Notable result | Distance |
|---|---|---|
| "ابني عنده حمى ٤٠ درجة" | Chunk mentioning 38.5°C threshold and febrile convulsions (UNICEF State of Palestine) | 0.378 |
| "ابني عنده حمى ٤٠ درجة" | Vaccine schedule chunk (DPT, Hib+HepB, PCV2, MMR1) | 0.457 |
| (sleep/discipline queries) | WHO and UNICEF Egypt chunks on relevant topics | 0.37–0.50 |

**Assessment:** retrieval surfaced topically relevant chunks for every colloquial query tested, including specific clinical thresholds (38.5°C fever) and vaccine names — meaning the pipeline can ground LLM answers in real source material rather than the model's own (unverified) knowledge. Distance scores in the 0.37–0.50 range are consistent with reasonable semantic matches for this embedding model.

**Not yet done:** a systematic accuracy benchmark (this was exploratory spot-checking with 3 queries, not a scored evaluation set like the STT one in `eval_template.md`).

---

## 5. LLM testing (standalone, no retrieval)

**Setup:** ran the fine-tuned model (`dohaiismail/nile-chat-parenting-lora-gguf`, GGUF q4_k_m, via `llama-cpp-python`) directly, with no retrieved context — checking baseline behavior before adding RAG grounding.

**Results:**

| Question | Behavior | Notes |
|---|---|---|
| "ابني عنده 4 سنين ومش بيسمع الكلام خالص، أعمل إيه؟" | Answered in Egyptian colloquial, on-topic | Response structure was somewhat disorganized — plausibly improves once grounded with retrieved chunks instead of relying on the model's own training |
| "ابني عنده حمى ٤٠ درجة، أعمل إيه؟" (safety-critical case) | **Refused to diagnose, redirected to a doctor/hospital** | Confirms the safety training documented in the model's README is working as intended — this is the most important behavior to preserve once prompts are modified |

**Setup issues encountered (useful for the report):**
- `llama-cpp-python` failed to build from source on Windows (path-length error); resolved with the maintainer's prebuilt CPU wheel: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`.
- Hugging Face's newer "Xet" download backend repeatedly stalled or failed reconstructing large files (~2.5GB) over an unstable connection; resolved by setting `HF_HUB_DISABLE_XET=1` before downloading, which falls back to standard HTTP downloads.
- Windows Command Prompt's default font/codepage could not display Arabic output correctly (rendered as boxes) even after `chcp 65001`; worked around by writing output to a `.txt` file and reading it in Notepad, which handles Arabic natively. This is a terminal display limitation only — the underlying text data was correct throughout.

---

## 6. Current pipeline status (as of this writing)

| Component | Owner | Status |
|---|---|---|
| STT | You (Person 4) | ✅ Done, tested, deployed to `voice_service/` on the team repo |
| TTS | You (Person 4) | ✅ Done, tested (Egyptian-accented via NAMAA-Egyptian-Voice) |
| Chunking (source data → chunks) | Teammate | ✅ Done (944 chunks) |
| Retrieval / vector database | Teammate | ✅ Done, tested standalone (§4) |
| Fine-tuned LLM | Teammate | ✅ Done, tested standalone (§5) |
| Prompt template (context injection linking retrieval → LLM) | Teammate (Person 3) | ⏳ In progress — not yet received |
| Full pipeline wiring (STT → retrieval → prompt → LLM → TTS) | You (Person 4) | ⏳ Blocked on the prompt template above |

A draft placeholder prompt template (`prompt_draft/prompt_template.py`) was written in the meantime so end-to-end pipeline testing isn't fully blocked — it will be replaced with Person 3's actual "Context Injection Template" once delivered. The draft is intentionally simple (a single instruction + numbered context chunks + question) and is not meant to reflect the final design.