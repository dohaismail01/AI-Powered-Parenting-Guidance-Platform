"""
Central config for the ParentWise STT service.
Everything is overridable via environment variables so the same code
runs in dev, in a teammate's local setup, and (if needed) in a deployed
container without code changes.
"""

import os


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small")          # tiny/base/small/medium/large-v3
DEVICE = os.getenv("STT_DEVICE", "cpu")                    # "cpu" or "cuda"
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")        # "int8" (cpu) or "float16" (cuda)
FORCE_LANGUAGE = os.getenv("STT_LANGUAGE", "ar")             # None to auto-detect instead

MIN_AUDIO_SECONDS = float(os.getenv("STT_MIN_AUDIO_SECONDS", "0.4"))
MAX_UPLOAD_MB = float(os.getenv("STT_MAX_UPLOAD_MB", "15"))

# How long a pause (in ms) the VAD treats as "the sentence ended."
# The faster-whisper default (500ms) was cutting off longer sentences
# where the speaker paused briefly mid-sentence — raised to 1500ms
# based on testing with real Arabic recordings.
VAD_MIN_SILENCE_MS = int(os.getenv("STT_VAD_MIN_SILENCE_MS", "1500"))

# Concurrency: how many transcriptions can run at once. faster-whisper
# is not guaranteed safe for unlimited concurrent calls on one model
# instance, so this bounds it with a semaphore rather than hoping for
# the best under load.
MAX_CONCURRENT_TRANSCRIPTIONS = int(os.getenv("STT_MAX_CONCURRENT", "1"))

# CORS - lock this down to your real frontend origin(s) before deploying.
# Comma-separated list, e.g. "http://localhost:5173,https://parentwise.app"
ALLOWED_ORIGINS = os.getenv("STT_ALLOWED_ORIGINS", "http://localhost:5173").split(",")

REQUIRE_API_KEY = _bool_env("STT_REQUIRE_API_KEY", False)
API_KEY = os.getenv("STT_API_KEY", "")  # set this + REQUIRE_API_KEY=true to lock the endpoint

LOG_LEVEL = os.getenv("STT_LOG_LEVEL", "INFO")

# --- TTS (text-to-speech) ---
# Uses the NAMAA-Egyptian-Voice public Gradio Space by default. For
# real (non-demo) use, duplicate the Space to your own HF account and
# point this at your copy instead — the public one may have rate limits.
TTS_SPACE_ID = os.getenv("TTS_SPACE_ID", "omarelshehy/NAMAA-Egyptian-Voice")
TTS_DEFAULT_REFERENCE_AUDIO = os.getenv("TTS_DEFAULT_REFERENCE_AUDIO", "")  # path to a .wav voice sample, optional
TTS_EXAGGERATION = float(os.getenv("TTS_EXAGGERATION", "0.5"))
TTS_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.8"))
TTS_CFGW = float(os.getenv("TTS_CFGW", "0.5"))

