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

import asyncio
import logging
import time
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import config
from transcriber import Transcriber
from tts_client import TTSClient

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
# Bounds how many transcriptions run at once, since faster-whisper isn't
# guaranteed safe for unlimited concurrent calls on a single model instance.
_transcription_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_TRANSCRIPTIONS)


@app.on_event("startup")
def startup():
    global transcriber, tts_client
    transcriber = Transcriber.load()
    try:
        tts_client = TTSClient.load()
    except Exception:
        # TTS is a bonus feature that depends on a remote Space being up.
        # Don't let a TTS connection problem take down STT, which is the
        # core, required part of this service.
        logger.exception("TTS client failed to connect - /tts will be unavailable, /stt still works")
        tts_client = None


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


@app.get("/health")
def health():
    return {
        "status": "ok" if transcriber is not None else "loading",
        "model": config.MODEL_SIZE,
        "device": config.DEVICE,
        "language_forced": config.FORCE_LANGUAGE,
        "max_concurrent": config.MAX_CONCURRENT_TRANSCRIPTIONS,
        "tts_available": tts_client is not None,
    }
