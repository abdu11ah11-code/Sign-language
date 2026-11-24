"""Whisper speech-to-text (runs locally, no API/network).

The model is loaded lazily on first use and cached, so importing this
module is cheap and the UI starts fast. Accepts either an in-memory float32
waveform (from :func:`sign_bridge.capture.microphone.record_audio`) or a
path to an audio file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sign_bridge.logging_conf import get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_model(name: str) -> Any:
    """Load and cache a Whisper model by size name."""
    import whisper  # heavy import deferred to first call

    log.info("whisper_loading", model=name)
    model = whisper.load_model(name)
    log.info("whisper_loaded", model=name)
    return model


class SpeechToText:
    """Local Whisper transcriber.

    Args:
        model_name: ``tiny`` | ``base`` | ``small`` | ``medium``.
        language: ISO language hint (e.g. ``"en"``) to skip auto-detect.
    """

    def __init__(self, model_name: str = "base", language: str = "en") -> None:
        self._model_name = model_name
        self._language = language

    def transcribe(self, audio: NDArray | str | Path) -> str:
        """Transcribe a waveform or audio file to text.

        Args:
            audio: float32 mono waveform (16 kHz) or a path to an audio file.

        Returns:
            The transcribed text, stripped. Empty string on silence.
        """
        model = _load_model(self._model_name)
        data: Any = str(audio) if isinstance(audio, (str, Path)) else np.asarray(
            audio, dtype=np.float32
        )

        result = model.transcribe(data, language=self._language, fp16=False)
        text = str(result.get("text", "")).strip()
        log.info("transcribed", chars=len(text))
        return text
