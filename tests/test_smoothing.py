"""Tests for temporal smoothing / sign locking."""

from __future__ import annotations

from sign_bridge.inference.smoothing import PredictionSmoother


def test_locks_after_consistent_high_confidence() -> None:
    sm = PredictionSmoother(lock_frames=5, confidence_threshold=0.8, agreement=0.7)
    result = None
    for _ in range(5):
        result = sm.update("HELLO", 0.9)
    assert result is not None and result.is_stable
    assert result.label == "HELLO"


def test_does_not_lock_below_confidence() -> None:
    sm = PredictionSmoother(lock_frames=5, confidence_threshold=0.8, agreement=0.7)
    result = None
    for _ in range(6):
        result = sm.update("HELLO", 0.5)  # consistently low
    assert result is not None and not result.is_stable


def test_does_not_lock_when_jittery() -> None:
    sm = PredictionSmoother(lock_frames=6, confidence_threshold=0.8, agreement=0.7)
    labels = ["A", "B", "A", "C", "A", "B"]  # no 70% majority
    result = None
    for lbl in labels:
        result = sm.update(lbl, 0.95)
    assert result is not None and not result.is_stable


def test_reset_clears_history() -> None:
    sm = PredictionSmoother(lock_frames=3, confidence_threshold=0.8, agreement=0.7)
    for _ in range(3):
        sm.update("YES", 0.9)
    sm.reset()
    assert sm.latest == (None, 0.0)
    # One post-reset update cannot meet lock_frames=3 yet.
    assert not sm.update("YES", 0.9).is_stable
