"""Microphone capture for Whisper speech-to-text.

Implementation note: the original spec suggested PyAudio; we ship
``sounddevice`` instead (same PortAudio backend, pip-installable wheels on
all platforms, no system headers needed). Documented in README §Setup.

Audio is captured as float32 mono at Whisper's native 16 kHz so transcribe
can consume the array directly with zero resampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sign_bridge.logging_conf import get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)


class MicrophoneError(RuntimeError):
    """Raised when no input device is available or recording fails."""


def record_audio(
    seconds: float,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> NDArray:
    """Record a fixed-length clip from the default input device.

    Args:
        seconds: Duration to record. Clamped to at least 0.5 s.
        sample_rate: Sample rate in Hz (16 kHz = Whisper-native).
        channels: Input channels; output is always mono (channel-averaged).

    Returns:
        ``(n_samples,)`` float32 mono waveform in [-1, 1].

    Raises:
        MicrophoneError: If no input device exists or capture fails.
    """
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing at the OS level
        raise MicrophoneError(f"Audio backend unavailable: {exc}") from exc

    duration = max(0.5, float(seconds))
    try:
        log.info("mic_record_start", seconds=duration, sample_rate=sample_rate)
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )
        sd.wait()
    except Exception as exc:  # sounddevice raises plain Exceptions for device errors
        raise MicrophoneError(
            f"Microphone recording failed: {exc}. "
            "Check that an input device is connected and permitted."
        ) from exc

    mono = recording.mean(axis=1) if recording.ndim == 2 else recording
    log.info("mic_record_done", samples=int(mono.shape[0]))
    return np.asarray(mono, dtype=np.float32)


def save_wav(waveform: NDArray, path: Path, sample_rate: int = 16_000) -> Path:
    """Write a mono float32 waveform to a WAV file (for debugging/repro).

    Args:
        waveform: ``(n,)`` float32 array in [-1, 1].
        path: Destination ``.wav`` path; parents are created.
        sample_rate: Sample rate to embed.

    Returns:
        The path written.
    """
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate)
    log.info("wav_saved", path=str(path))
    return path
