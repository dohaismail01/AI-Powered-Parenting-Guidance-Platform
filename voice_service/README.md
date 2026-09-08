# ParentWise — Voice Service

The speech front-door of ParentWise. It turns a parent's **spoken or typed** question
into a **spoken and/or written** answer by orchestrating three steps:

```
audio → faster-whisper (STT) ─┐
                              ├─→ RAG API /ask (retrieval + safety + fine-tuned LLM) → NAMAA TTS → answer (text + audio)
text ─────────────────────────┘
```

STT and TTS live here; the actual answer comes from the [`rag/`](../rag) service over
HTTP (`RAG_API_URL`, default `http://localhost:8000`).

## Run

```bash
pip install -r requirements.txt
cp .env.example .env            # then set HF_TOKEN in .env (needed for the TTS Space)
py -3.11 -m uvicorn stt_service:app --port 8001
```

Startup loads faster-whisper and connects to the TTS Space (~30 s). Use **port 8001** —
the frontends and `stt_service.py` all assume it.

## Endpoints

| Method / path | Purpose |
|---|---|
| `POST /stt` | Audio → Arabic transcript (JSON). |
| `POST /tts` | `{ "text": "…" }` → a `wav` file of Egyptian-Arabic speech. |
| `POST /ask_voice` | **The full pipeline.** Send `audio` **or** `text` (exactly one) plus optional `speak` (default `true`). Always returns **JSON**: `transcript`, `answer`, `sources`, `is_high_risk`, `is_grounded`, and — when `speak` is true and TTS is up — a base64 `audio` (`audio_mime`). TTS is best-effort: if the Space is down you still get the text answer with `audio_error` set. |
| `GET /health` | Model/device info and `tts_available` / `rag_available` flags. |
| `GET /` | Serves the single-file web app (`parentwise_app.html`). |

Because `/ask_voice` returns the answer **text**, callers can render text output, voice
output, or both — that's what the "طريقة الرد: نص / صوت" toggles in the UIs control.

## Frontends

Two interchangeable UIs, both **text-or-voice in, text-or-voice out**:

- **React app** — [`frontend/`](frontend) (Vite). `cd frontend && npm install && npm run dev`
  → http://localhost:5173. Proxies the API to `:8001` (no CORS) and reuses
  [`useVoiceRecorder.js`](useVoiceRecorder.js). See [frontend/README.md](frontend/README.md).
- **Single-file app** — [`parentwise_app.html`](parentwise_app.html), served by this
  service at `http://localhost:8001/`. No build step; good for a quick demo.

`useVoiceRecorder.js` (repo root of this folder) is the standalone hook the React app is
based on.

## Configuration

All settings are environment variables (loaded from `.env` via `python-dotenv`); see
[`config.py`](config.py) and [`.env.example`](.env.example). Common ones:

| Variable | Default | Notes |
|---|---|---|
| `STT_MODEL_SIZE` / `STT_DEVICE` / `STT_COMPUTE_TYPE` | `small` / `cpu` / `int8` | faster-whisper model + placement. |
| `STT_LANGUAGE` | `ar` | forced transcription language (set empty to auto-detect). |
| `STT_ALLOWED_ORIGINS` | `http://localhost:5173` | CORS origins (the React dev server). The single-file app is same-origin, so it needs no entry. |
| `RAG_API_URL` | `http://localhost:8000` | where the RAG service lives. |
| `TTS_SPACE_ID` | `omarelshehy/NAMAA-Egyptian-Voice` | the Hugging Face TTS Space. |
| `HF_TOKEN` | — | read by the HF libraries for the TTS Space; put a fresh token in `.env`. |

## Notes

- **TTS is external and flaky.** The NAMAA Space can be up on one start and refuse the
  next (`WinError 10061`). `/ask_voice` won't fail because of it — text still returns —
  but if `/health` shows `tts_available: false`, just restart.
- The service forces `stdout`/`stderr` to UTF-8 so the Windows console (cp1252) doesn't
  crash on the `✔` that `gradio_client` prints.
