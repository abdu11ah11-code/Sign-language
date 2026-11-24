"""Text-to-speech wrapper for the sign→speech direction.

Two backends, selected in config (§tts.engine):

    gtts     — Google TTS; natural voices, needs network; returns MP3 bytes
               that the Streamlit UI can play with ``st.audio``.
    pyttsx3  — offline OS speech engine; speaks directly on the host. Useful
               for fully local/air-gapped demos; cannot return bytes for the
               browser, so the UI notes when audio plays on the server.

Both share the :class:`TextToSpeech` interface so callers stay backend-
agnostic.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys

from sign_bridge.logging_conf import get_logger

log = get_logger(__name__)

_URDU_RE = re.compile("[\u0600-\u06FF]")


class TextToSpeech:
    """Backend-agnostic TTS facade.

    Args:
        engine: ``"gtts"`` or ``"pyttsx3"``.
        gtts_lang: Language code for gTTS (e.g. ``"en"``, ``"ur"``).
        pyttsx3_rate: Words-per-minute for the offline engine.
    """

    def __init__(
        self,
        engine: str = "gtts",
        gtts_lang: str = "en",
        pyttsx3_rate: int = 170,
    ) -> None:
        self._engine = engine
        self._gtts_lang = gtts_lang
        self._rate = pyttsx3_rate

    def to_mp3_bytes(self, text: str) -> bytes | None:
        """Synthesize speech and return MP3 bytes (gTTS only).

        Args:
            text: Text to speak.

        Returns:
            MP3 bytes for browser playback, or None if the active engine
            cannot produce bytes (pyttsx3) or text is empty.
        """
        text = text.strip()
        if not text:
            return None
        if self._engine != "gtts":
            return None
        try:
            from gtts import gTTS

            buffer = io.BytesIO()
            # Urdu script (e.g. words spelled with the PSL alphabet) needs the Urdu voice.
            lang = "ur" if _URDU_RE.search(text) else self._gtts_lang
            gTTS(text=text, lang=lang).write_to_fp(buffer)
            buffer.seek(0)
            log.info("tts_synthesized", chars=len(text), engine="gtts")
            return buffer.read()
        except Exception as exc:  # network/voice errors shouldn't crash the UI
            log.warning("tts_failed", error=str(exc))
            return None

    def speak_blocking(self, text: str, timeout: float = 60.0) -> bool:
        """Speak on the host machine (pyttsx3). Blocks until done.

        pyttsx3 runs in a short-lived child process. Driving it from
        Streamlit's script thread hangs on the second sentence ("run loop
        already started") and fails on Windows threads without COM set up;
        a fresh process per utterance avoids both.

        Args:
            text: Text to speak.
            timeout: Seconds before the speech process is abandoned.

        Returns:
            True on success, False on failure or empty text.
        """
        text = text.strip()
        if not text:
            return False
        script = (
            "import sys, pyttsx3\n"
            "e = pyttsx3.init()\n"
            "e.setProperty('rate', int(sys.argv[1]))\n"
            "e.say(sys.stdin.read())\n"
            "e.runAndWait()\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script, str(self._rate)],
                input=text, text=True, capture_output=True, timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("tts_speak_failed", error=str(exc))
            return False
        if proc.returncode != 0:
            log.warning("tts_speak_failed", error=proc.stderr.strip()[-300:])
            return False
        log.info("tts_spoken", chars=len(text), engine="pyttsx3")
        return True
