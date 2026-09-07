"""
Text-to-speech client for ParentWise, using the NAMAA-Egyptian-Voice
Gradio Space (Egyptian-dialect TTS). Kept separate from the STT code —
this only handles turning text into audio, and knows nothing about
transcription.

This calls a *remote* Hugging Face Space over the network rather than
running a model locally, so:
  - no heavy model download / GPU needed on this machine
  - it depends on that Space staying up and responsive
  - the public Space may be rate-limited under real traffic — see
    config.TTS_SPACE_ID for how to point this at your own duplicated
    Space instead
"""

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass

from gradio_client import Client, handle_file

import config

logger = logging.getLogger("parentwise-stt")


@dataclass
class TTSResult:
    audio_path: str  # path to a local wav file containing the generated speech


class TTSClient:
    """Thin wrapper around the NAMAA Egyptian Voice Gradio Space."""

    def __init__(self, client: Client):
        self._client = client

    @classmethod
    def load(cls) -> "TTSClient":
        logger.info("Connecting to TTS Space: %s", config.TTS_SPACE_ID)
        client = Client(config.TTS_SPACE_ID)
        logger.info("TTS client ready.")
        return cls(client)

    def synthesize(self, text: str, reference_audio_path: str | None = None) -> TTSResult:
        """Generate speech audio for `text`. If reference_audio_path is
        given, the output voice is modeled on that sample; otherwise
        falls back to config.TTS_DEFAULT_REFERENCE_AUDIO (or the Space's
        own built-in default if that's also unset)."""
        ref_path = reference_audio_path or config.TTS_DEFAULT_REFERENCE_AUDIO or None

        kwargs = dict(
            text_input=text,
            exaggeration_input=config.TTS_EXAGGERATION,
            temperature_input=config.TTS_TEMPERATURE,
            seed_num_input=0,
            cfgw_input=config.TTS_CFGW,
            api_name="/generate_tts_audio",
        )
        kwargs["audio_prompt_path_input"] = handle_file(ref_path) if ref_path else None

        result_path = self._client.predict(**kwargs)

        # The Space returns a path in its own temp storage; copy it
        # somewhere we control so it survives independently and we can
        # clean it up on our own schedule.
        out_fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(out_fd)
        shutil.copyfile(result_path, out_path)

        return TTSResult(audio_path=out_path)

