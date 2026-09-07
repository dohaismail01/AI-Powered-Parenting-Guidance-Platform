"""
Central config for the ParentWise STT service.

Everything is overridable via environment variables so the same code
runs in dev, in a teammate's local setup, and (if needed) in a deployed
container without code changes.
"""
import os

# Load variables from a local .env file (sitting next to this file) if present,
# before anything below reads the environment. No-op if python-dotenv isn't
# installed or the .env file doesn't exist, so real shell env vars still work.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass


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

VAD_MIN_SILENCE_MS = int(os.getenv("STT_VAD_MIN_SILENCE_MS", "1500"))

MAX_CONCURRENT_TRANSCRIPTIONS = int(os.getenv("STT_MAX_CONCURRENT", "1"))

ALLOWED_ORIGINS = os.getenv("STT_ALLOWED_ORIGINS", "*").split(",")

REQUIRE_API_KEY = _bool_env("STT_REQUIRE_API_KEY", False)
API_KEY = os.getenv("STT_API_KEY", "")

LOG_LEVEL = os.getenv("STT_LOG_LEVEL", "INFO")

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
RAG_API_TIMEOUT_SECONDS = float(os.getenv("RAG_API_TIMEOUT_SECONDS", "600"))

TTS_SPACE_ID = os.getenv("TTS_SPACE_ID", "omarelshehy/NAMAA-Egyptian-Voice")
TTS_DEFAULT_REFERENCE_AUDIO = os.getenv("TTS_DEFAULT_REFERENCE_AUDIO", "")
TTS_EXAGGERATION = float(os.getenv("TTS_EXAGGERATION", "0.5"))
TTS_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.8"))
TTS_CFGW = float(os.getenv("TTS_CFGW", "0.5"))
