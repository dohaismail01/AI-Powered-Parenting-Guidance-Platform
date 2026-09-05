"""
Core transcription logic — deliberately separate from the FastAPI layer
so it can be unit tested without spinning up a web server, and reused
from test_stt.py, stt_service.py, or a future batch-processing script.
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass

from faster_whisper import WhisperModel

import config

logger = logging.getLogger("parentwise-stt")


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_confidence: float
    duration_seconds: float
    processing_time_seconds: float
    warning: str | None = None  # "no_speech_detected" | "clip_too_short" | None


class Transcriber:
    """Thin wrapper around a loaded faster-whisper model."""

    def __init__(self, model: WhisperModel):
        self._model = model

    @classmethod
    def load(cls) -> "Transcriber":
        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            config.MODEL_SIZE, config.DEVICE, config.COMPUTE_TYPE,
        )
        model = WhisperModel(
            config.MODEL_SIZE,
            device=config.DEVICE,
            compute_type=config.COMPUTE_TYPE,
        )
        logger.info("Model loaded.")
        return cls(model)

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file already on disk. Raises on decode failure —
        caller is responsible for turning that into an HTTP error."""
        start = time.time()

        segments, info = self._model.transcribe(
            audio_path,
            language=config.FORCE_LANGUAGE,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        segments = list(segments)
        elapsed = time.time() - start

        if not segments:
            return TranscriptionResult(
                text="", language=info.language, language_confidence=info.language_probability,
                duration_seconds=0.0, processing_time_seconds=elapsed,
                warning="no_speech_detected",
            )

        duration = segments[-1].end
        if duration < config.MIN_AUDIO_SECONDS:
            return TranscriptionResult(
                text="", language=info.language, language_confidence=info.language_probability,
                duration_seconds=duration, processing_time_seconds=elapsed,
                warning="clip_too_short",
            )

        text = " ".join(seg.text.strip() for seg in segments)

        return TranscriptionResult(
            text=text,
            language=info.language,
            language_confidence=info.language_probability,
            duration_seconds=duration,
            processing_time_seconds=elapsed,
            warning=None,
        )

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".webm") -> TranscriptionResult:
        """Write bytes to a temp file, transcribe, and always clean up —
        even if transcription raises."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning("Failed to delete temp file: %s", tmp_path)
