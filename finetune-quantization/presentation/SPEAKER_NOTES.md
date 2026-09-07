# ParentWise - Speaker Notes

*Talking script for `ParentWise_Platform.pptx` (18 slides). Also embedded in the deck's PowerPoint Presenter View (View > Notes).*

## Slide 1 - Title

This is ParentWise — an Arabic-first parenting assistant. A parent asks a question, by voice or text, in Egyptian colloquial Arabic, and gets a warm, evidence-grounded answer that knows when to step back and send them to a professional. It brings together three Gen-AI pieces — retrieval, a fine-tuned model, and voice — and our one-line philosophy is: we tune the model's behaviour, and we ground its facts.

## Slide 2 - Problem & Approach

The problem: parents face everyday challenges — tantrums, discipline, screen time, sleep — and today they get scattered, inconsistent advice from social media. Our users are parents of 0-to-12-year-olds, especially first-timers. Two commitments drive the whole design. First, teach behaviour and ground facts: fine-tuning shapes how it answers, RAG decides what it answers with. Second, a hard safety boundary — anything medical or an emergency is refused and redirected, and that's enforced twice.

## Slide 3 - Architecture

Here's how a question flows. The user speaks or types into the React frontend, which calls the FastAPI backend. Inside the backend: if it's voice, Whisper transcribes it; we build a prompt from the system prompt, few-shot examples and retrieved context; the RAG retriever pulls the top chunks from ChromaDB; and the fine-tuned QLoRA model produces the answer. That grounded answer goes back to the frontend and is shown to the parent.

## Slide 4 - RAG - Knowledge Base

Component one is retrieval. Our knowledge base is 944 chunks drawn from 34 real documents — 11 PDFs, 13 web sources, and 10 research papers — all from UNICEF, WHO and academic parenting research. We built it with structure-aware chunking, so we never return an orphaned sentence, and Arabic Unicode normalization so retrieval works reliably on Arabic. The point: we retrieve real evidence, not hallucinated advice.

## Slide 5 - RAG - Retrieval Engine

Retrieval is hybrid, because neither method alone is enough. Dense BGE-M3 embeddings catch meaning even when the wording differs; BM25 catches exact Arabic terms that embeddings can dilute. We fuse the two with Reciprocal Rank Fusion — no hand-tuned weights — then apply metadata ranking and a cross-encoder reranker, landing on the 8 best chunks for the model. Two retrievers see what one alone would miss.

## Slide 6 - RAG - Safety Gate

Safety is handled before generation, not after. A deterministic gate classifies every query across 7 risk categories. Four are hard-blocked — child abuse, violence, self-harm, and emergencies — and go straight to a safety response with crisis resources, with zero retrieval calls. Medical and mental-health are soft boundaries: we give general information and nudge toward a professional, never a diagnosis. A dangerous question never reaches the model.

## Slide 7 - RAG to Fine-Tuned Model (Context Injection)

This is the bridge between retrieval and the model. Step one, we formulate the retrieval query; step two, we inject the retrieved chunks into a structured, per-source template; step three, the model generates grounded — using only those sources, and citing them. Crucially, the system prompt starts from the exact base message the model was fine-tuned on, so the RAG rules build on the training instead of fighting it.

## Slide 8 - Fine-Tuning (QLoRA)

Component two is fine-tuning. We adapted Nile-Chat-4B with QLoRA: the base model is frozen in 4-bit, and we train lightweight adapters — under one percent of the parameters — with rank 16, learning rate 1e-4, for 2 epochs. The dataset is 507 Q&A pairs: 483 grounded parenting pairs plus 24 safety and refusal pairs, split into 482 train, 25 validation, and a 28-item held-out test with zero overlap. We fine-tune for tone and safety, not facts.

## Slide 9 - Optimization (Quantization)

Then we optimize for deployment. Post-training quantization to GGUF Q4_K_M takes the model from about 8 gigabytes down to 2.5 — roughly three times smaller — and it runs on a laptop CPU at about 9 tokens per second, with no GPU required to serve. Same behaviour, a third of the size, so it runs right next to the RAG stack, anywhere.

## Slide 10 - LLM Evaluation

Now an honest read of the evaluation — 28 rubric-graded held-out questions, where safety-critical items are hard fails, about 67% overall. What worked: warm on-persona tone, correct refusals on several dangerous myths, emergencies routed to a doctor, and it runs on CPU. What we still need to fix is real: a dosage question produced a number, a choking case wasn't escalated, some facts were off, and a few out-of-scope questions got answered. We show the failures, we don't hide them.

## Slide 11 - Prompt Engineering - Router / Generator / Verifier

On the prompt-engineering side, safety is defense-in-depth — the fine-tuned model isn't the only line of defense. First a Router classifies intent with zero-shot, and skips retrieval and generation entirely for medical or out-of-scope questions. Then the Generator uses role, few-shot, and a silent chain-of-thought. Finally an independent Verifier — a separate call, not the model grading itself — checks grounding and safety before anything ships. Three independent checks, each catching what the others miss.

## Slide 12 - Prompt Engineering - Six Techniques

Zooming in, we use six named prompt-engineering techniques, each doing one job — not stacked for show. Role prompting sets the persona; zero-shot classification drives the router; few-shot examples demonstrate the response shape; chain-of-thought does silent safety and age-fit reasoning; retrieval-grounding restricts answers to the sources; and chain-of-verification double-checks the draft. Together they're the glue between RAG and the fine-tuned model.

## Slide 13 - Reliability Safeguards

These are the answer-quality guarantees users actually feel. Answers are age-aware, tailored to the child's age. We distinguish evidence from opinion. Medical questions are redirected to a professional. And when the retrieved context isn't enough, the model says so out loud instead of making something up. Safety here isn't a prompt line — it's enforced before generation.

## Slide 14 - Prompt Engineering - Real-World Testing

Every fix on this slide came from testing against the real model and real UNICEF data, not assumptions. We validated 5 test categories and found and fixed 3 real bugs. The router first misfired — a ball-sharing question was tagged medical — until we added five few-shot examples. And keyword retrieval pulled irrelevant chunks until we added stopword filtering and an overlap threshold, so now it falls back honestly instead of hallucinating. At the bottom you can see routing working: behavioural gets a grounded answer, medical a clean redirect, out-of-scope is politely declined.

## Slide 15 - Frontend (React RTL Chat)

The frontend is a working React chat UI, built right-to-left first — because that's the actual language of the product, down to the sidebar, message alignment and every string. Voice recording uses the browser's MediaRecorder with a live 'listening' state, ready for the STT endpoint. And there's a clean mock-to-real swap point: two isolated functions for chat and transcription, so wiring the real backend touches no UI code.

## Slide 16 - Voice Integration (STT + TTS)

Component three is voice, and it's hands-free by design — parents can ask while their hands are full. Speech-to-text uses faster-whisper, forced to Arabic, with the voice-activity threshold tuned to 1500 milliseconds after testing showed shorter pauses were cutting sentences off. Text-to-speech uses the NAMAA Egyptian voice. It's a standalone FastAPI service wired end-to-end into the RAG and fine-tuned model over HTTP.

## Slide 17 - Voice-to-Voice Pipeline

Put it together and this is the headline result: a full voice-to-voice pipeline. The parent speaks; STT transcribes; RAG plus the fine-tuned Nile-Chat-4B produces a grounded answer; and TTS speaks it back in Egyptian Arabic. This isn't a mock-up — we tested it live, end-to-end.

## Slide 18 - Limitations & Future Improvements

To close, the honest limitations. This is not a substitute for professional advice; the safety refusals aren't perfect yet; facts are only as good as the sources; coverage is deliberately narrow; and STT can drift toward formal Arabic while TTS leans on a hosted service. Our roadmap follows directly: expand the safety data, strengthen refusal training, ground all facts through RAG, broaden topics and ages, and upgrade to a dialect-specific STT and a private TTS. A clean, working small system beats an overextended one. Thank you — happy to take questions.
