"""
FastAPI STT endpoint for ParentWise.

Receives an audio blob from the React frontend, transcribes it with
faster-whisper (Arabic forced by default), and returns plain text that
gets handed off to the RAG + LLM pipeline unchanged.

Run with:
    uvicorn stt_service:app --reload --port 8001

Install deps:
    pip install -r requirements.txt
"""

import sys

# gradio_client prints progress lines containing "✔"; on Windows the
# default console encoding (cp1252) can't encode that character, so /tts died
# with UnicodeEncodeError. Forcing UTF-8 on our own streams makes the service
# work without needing PYTHONUTF8=1 set in the environment.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import asyncio
import base64
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import config
from transcriber import Transcriber
from tts_client import TTSClient
from rag_client import RagClient

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("parentwise-stt")

app = FastAPI(title="ParentWise STT Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

transcriber: Transcriber | None = None
tts_client: TTSClient | None = None
rag_client: RagClient | None = None
# Bounds how many transcriptions run at once, since faster-whisper isn't
# guaranteed safe for unlimited concurrent calls on a single model instance.
_transcription_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_TRANSCRIPTIONS)


@app.on_event("startup")
def startup():
    global transcriber, tts_client, rag_client
    transcriber = Transcriber.load()
    try:
        tts_client = TTSClient.load()
    except Exception:
        # TTS is a bonus feature that depends on a remote Space being up.
        # Don't let a TTS connection problem take down STT, which is the
        # core, required part of this service.
        logger.exception("TTS client failed to connect - /tts will be unavailable, /stt still works")
        tts_client = None

    rag_client = RagClient()
    if not rag_client.health_check():
        # Same principle as TTS: the RAG API is a separate teammate-owned
        # service. If it's down, /stt and /tts should keep working -
        # only /ask_voice (which needs all three) becomes unavailable.
        logger.warning("RAG API at %s is not reachable - /ask_voice will be unavailable", config.RAG_API_URL)


def _check_api_key(x_api_key: str | None):
    if not config.REQUIRE_API_KEY:
        return
    if not x_api_key or x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.post("/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    request_id = str(uuid.uuid4())[:8]
    _check_api_key(x_api_key)

    if transcriber is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw = await audio.read()
    size_mb = len(raw) / (1024 * 1024)
    logger.info("[%s] received upload: %.2f MB, filename=%s", request_id, size_mb, audio.filename)

    if size_mb == 0:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"Audio too large ({size_mb:.1f} MB)")

    suffix = _guess_suffix(audio.filename, audio.content_type)

    async with _transcription_semaphore:
        start = time.time()
        try:
            result = await asyncio.to_thread(transcriber.transcribe_bytes, raw, suffix)
        except Exception:
            logger.exception("[%s] transcription failed", request_id)
            raise HTTPException(status_code=422, detail="Could not process this audio file")
        elapsed = time.time() - start

    logger.info(
        "[%s] done in %.2fs - warning=%s text_len=%d",
        request_id, elapsed, result.warning, len(result.text),
    )

    return {
        "request_id": request_id,
        "text": result.text,
        "warning": result.warning,
        "language": result.language,
        "language_confidence": round(result.language_confidence, 3),
        "duration_seconds": round(result.duration_seconds, 2),
        "processing_time_seconds": round(elapsed, 2),
    }


def _guess_suffix(filename, content_type):
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1]
    if content_type:
        mapping = {
            "audio/webm": ".webm",
            "audio/wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/ogg": ".ogg",
        }
        return mapping.get(content_type, ".webm")
    return ".webm"


@app.post("/tts")
async def text_to_speech(
    text: str = Body(..., embed=True),
    x_api_key: str | None = Header(default=None),
):
    """Generate Egyptian-Arabic speech audio for `text` and return it as
    a wav file. Uses the NAMAA-Egyptian-Voice Space configured in
    config.TTS_SPACE_ID."""
    request_id = str(uuid.uuid4())[:8]
    _check_api_key(x_api_key)

    if tts_client is None:
        raise HTTPException(status_code=503, detail="TTS is unavailable (failed to connect at startup)")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    logger.info("[%s] TTS request: %d chars", request_id, len(text))

    start = time.time()
    try:
        result = await asyncio.to_thread(tts_client.synthesize, text)
    except Exception:
        logger.exception("[%s] TTS synthesis failed", request_id)
        raise HTTPException(status_code=502, detail="TTS synthesis failed")
    elapsed = time.time() - start

    logger.info("[%s] TTS done in %.2fs -> %s", request_id, elapsed, result.audio_path)

    return FileResponse(
        result.audio_path,
        media_type="audio/wav",
        filename="speech.wav",
        headers={"X-Request-Id": request_id, "X-Processing-Time": f"{elapsed:.2f}"},
    )


_APP_HTML = Path(__file__).parent / "parentwise_app.html"


@app.get("/")
def index():
    """Serve the ParentWise web app (text/voice in, text/voice out) from the
    same origin as the API, so the browser mic works (secure context) and
    there are no cross-origin issues."""
    return FileResponse(_APP_HTML, media_type="text/html")


@app.get("/health")
def health():
    return {
        "status": "ok" if transcriber is not None else "loading",
        "model": config.MODEL_SIZE,
        "device": config.DEVICE,
        "language_forced": config.FORCE_LANGUAGE,
        "max_concurrent": config.MAX_CONCURRENT_TRANSCRIPTIONS,
        "tts_available": tts_client is not None,
        "rag_available": bool(rag_client is not None and rag_client.health_check()),
    }


@app.post("/ask_voice")
async def ask_voice(
    audio: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    speak: bool = Form(default=True),
    x_api_key: str | None = Header(default=None),
):
    """
    Full ParentWise pipeline in one call, supporting text OR voice in and
    text OR voice out. Send exactly one of `audio` / `text`:
        audio -> STT -> RAG (retrieval + LLM, teammate's server) -> answer
        text  ->        RAG                                       -> answer

    The response is always JSON so the caller always gets the answer *text*
    (and its sources). When `speak` is true and the TTS Space is up, a
    base64-encoded wav of the spoken answer is included too, so the frontend
    can offer text output, voice output, or both. TTS is best-effort: if it
    fails or is unavailable, the text answer is still returned (with
    `audio_error` set) rather than failing the whole request.
    """
    request_id = str(uuid.uuid4())[:8]
    _check_api_key(x_api_key)

    typed = text.strip() if text else ""
    has_audio = audio is not None and audio.filename is not None
    if has_audio and typed:
        raise HTTPException(status_code=400, detail="Send either audio or text, not both")
    if not has_audio and not typed:
        raise HTTPException(status_code=400, detail="Send a question as either audio or text")

    if rag_client is None:
        raise HTTPException(status_code=503, detail="RAG client not initialized")

    # 1) Get the question as text - transcribing it first only if it came in
    #    as audio. A typed question needs no STT model at all.
    if typed:
        question = typed
        logger.info("[%s] typed question -> %r", request_id, question)
    else:
        if transcriber is None:
            raise HTTPException(status_code=503, detail="STT model not loaded yet")

        raw = await audio.read()
        if len(raw) == 0:
            raise HTTPException(status_code=400, detail="Empty audio upload")

        suffix = _guess_suffix(audio.filename, audio.content_type)
        async with _transcription_semaphore:
            try:
                stt_result = await asyncio.to_thread(transcriber.transcribe_bytes, raw, suffix)
            except Exception:
                logger.exception("[%s] STT step failed", request_id)
                raise HTTPException(status_code=422, detail="Could not process this audio file")

        if not stt_result.text.strip():
            raise HTTPException(status_code=422, detail="No speech detected in the audio")

        question = stt_result.text
        logger.info("[%s] STT -> %r", request_id, question)

    # 2) RAG: text -> answer (teammate's server does retrieval + LLM + safety)
    try:
        rag_answer = await asyncio.to_thread(rag_client.ask, question)
    except Exception:
        logger.exception("[%s] RAG API call failed", request_id)
        raise HTTPException(status_code=502, detail="RAG service failed to answer")

    logger.info(
        "[%s] RAG -> high_risk=%s grounded=%s answer_len=%d",
        request_id, rag_answer.is_high_risk, rag_answer.is_grounded, len(rag_answer.answer),
    )

    # 3) TTS (optional, best-effort): answer text -> base64 wav. Never fails
    #    the request - if voice output is unavailable the caller still gets
    #    the text and can show it / retry the audio.
    audio_b64 = None
    audio_error = None
    if speak:
        if tts_client is None:
            audio_error = "TTS is unavailable (failed to connect at startup)"
        else:
            try:
                tts_result = await asyncio.to_thread(tts_client.synthesize, rag_answer.answer)
                with open(tts_result.audio_path, "rb") as fh:
                    audio_b64 = base64.b64encode(fh.read()).decode("ascii")
            except Exception:
                logger.exception("[%s] TTS step failed", request_id)
                audio_error = "TTS synthesis failed"

    return {
        "request_id": request_id,
        "transcript": question,
        "answer": rag_answer.answer,
        "sources": getattr(rag_answer, "sources", []),
        "is_high_risk": rag_answer.is_high_risk,
        "is_grounded": rag_answer.is_grounded,
        "audio": audio_b64,          # base64 wav, or null
        "audio_mime": "audio/wav" if audio_b64 else None,
        "audio_error": audio_error,  # null on success / when speak=false
    }
