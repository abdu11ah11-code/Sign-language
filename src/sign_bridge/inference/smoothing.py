"""Temporal smoothing for live predictions.

A raw per-window classifier is jittery: adjacent windows can flip between
classes. :class:`PredictionSmoother` keeps a short history of (class,
confidence) pairs and only reports a "stable" sign once the same class has
dominated the recent window *and* cleared the confidence floor — the
mechanism that turns noisy frame-by-frame output into a single locked sign.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class SmoothedPrediction:
    """Result of feeding one raw prediction into the smoother.

    Attributes:
        label: Majority label across the current buffer (or None if empty).
        confidence: Mean confidence of the majority label's samples.
        is_stable: True when the majority label has met the agreement and
            confidence criteria across ``lock_frames`` recent predictions.
    """

    label: str | None
    confidence: float
    is_stable: bool


class PredictionSmoother:
    """Sliding-window majority vote with a confidence gate.

    Args:
        lock_frames: History length; also the number of recent predictions
            considered for stability.
        confidence_threshold: Minimum mean confidence for the majority label.
        agreement: Fraction of the buffer that must share the majority label
            (0.7 = 70% of recent frames agree).
    """

    def __init__(
        self,
        lock_frames: int = 10,
        confidence_threshold: float = 0.80,
        agreement: float = 0.7,
    ) -> None:
        self._buf: deque[tuple[str, float]] = deque(maxlen=lock_frames)
        self._lock_frames = lock_frames
        self._threshold = confidence_threshold
        self._agreement = agreement

    def update(self, label: str, confidence: float) -> SmoothedPrediction:
        """Add one raw prediction and recompute the smoothed result."""
        self._buf.append((label, confidence))
        if not self._buf:
            return SmoothedPrediction(None, 0.0, False)

        counts = Counter(lbl for lbl, _ in self._buf)
        top_label, top_count = counts.most_common(1)[0]
        confs = [c for lbl, c in self._buf if lbl == top_label]
        mean_conf = sum(confs) / len(confs)

        is_stable = (
            len(self._buf) >= self._lock_frames
            and top_count / len(self._buf) >= self._agreement
            and mean_conf >= self._threshold
        )
        return SmoothedPrediction(top_label, mean_conf, is_stable)

    @property
    def latest(self) -> tuple[str | None, float]:
        """The most recent raw (label, confidence), or (None, 0.0)."""
        return self._buf[-1] if self._buf else (None, 0.0)

    def reset(self) -> None:
        """Clear history (call after locking a sign or losing the hands)."""
        self._buf.clear()
