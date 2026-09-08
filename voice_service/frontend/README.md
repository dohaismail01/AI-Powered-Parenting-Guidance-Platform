# ParentWise — React frontend

Vite + React UI for the ParentWise voice service. Supports **text or voice
input** and **text or voice output**, talking to the FastAPI `/ask_voice`
endpoint (one call does STT → RAG → TTS).

## Run

1. Start the backend (from `../`, i.e. `voice_service/`):
   ```
   py -3.11 -m uvicorn stt_service:app --port 8001
   ```
   and the RAG API on `:8000` (see `../../rag`).

2. Start the frontend dev server:
   ```
   cd frontend
   npm install
   npm run dev
   ```
   Open http://localhost:5173.

The dev server proxies `/ask_voice`, `/health`, `/stt`, `/tts` to
`http://localhost:8001` (see `vite.config.js`), so the browser uses a single
origin — no CORS setup needed, and the mic works (localhost is a secure
context).

## Notes
- Answer latency depends on the RAG model backend; the UI shows a live
  "thinking… (N s)" counter.
- Voice output is best-effort: if the TTS Space is down you still get the text
  answer, with a small notice.
- A dependency-free single-file version also exists at
  `../parentwise_app.html`, served by the backend at `http://localhost:8001/`.

## Build for production
```
npm run build      # outputs static files to dist/
npm run preview    # serve the build locally
```
For deployment, serve `dist/` behind the same origin as the API (or set up a
reverse proxy) so the relative `/ask_voice` calls resolve.
