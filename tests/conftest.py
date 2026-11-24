"""Shared pytest fixtures.

These let the suite run on CI without a camera, microphone, or the heavy
MediaPipe/TensorFlow stack: MediaPipe result objects are faked, and feature
windows are synthesized directly in the 162-dim base layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sign_bridge.capture.holistic import FEATURE_DIM, HAND_POINTS, POSE_LANDMARK_INDICES


@dataclass
class _FakeLandmark:
    x: float
    y: float
    z: float


class _FakeLandmarkList:
    """Mimics MediaPipe's ``landmark`` container."""

    def __init__(self, points: list[tuple[float, float, float]]) -> None:
        self.landmark = [_FakeLandmark(*p) for p in points]


class FakeHolisticResults:
    """Stand-in for MediaPipe Holistic results with controllable presence."""

    def __init__(self, left: bool = True, right: bool = True, pose: bool = True,
                 face: bool = True) -> None:
        rng = np.random.default_rng(0)
        self.left_hand_landmarks = (
            _FakeLandmarkList([(float(rng.random()), float(rng.random()), float(rng.random()))
                               for _ in range(HAND_POINTS)]) if left else None
        )
        self.right_hand_landmarks = (
            _FakeLandmarkList([(float(rng.random()), float(rng.random()), float(rng.random()))
                               for _ in range(HAND_POINTS)]) if right else None
        )
        # Pose needs the max index we read (33 covers all POSE_LANDMARK_INDICES).
        n_pose = max(POSE_LANDMARK_INDICES) + 1
        self.pose_landmarks = (
            _FakeLandmarkList([(0.5 + 0.01 * i, 0.5, 0.0) for i in range(n_pose)])
            if pose else None
        )
        # Face mesh has 468 points; lips are read from a subset of them.
        self.face_landmarks = (
            _FakeLandmarkList([(0.5, 0.2 + 0.0001 * i, 0.0) for i in range(468)])
            if face else None
        )


@pytest.fixture
def fake_results() -> FakeHolisticResults:
    """A fully-detected fake Holistic result."""
    return FakeHolisticResults()


@pytest.fixture
def synthetic_window() -> np.ndarray:
    """A ``(30, 162)`` window with a plausible shoulder reference set.

    Shoulders are placed at fixed, separated positions so normalization has a
    non-degenerate reference; other points are small random offsets.
    """
    from sign_bridge.capture.holistic import LEFT_SHOULDER_OFFSET, RIGHT_SHOULDER_OFFSET

    rng = np.random.default_rng(42)
    window = rng.normal(0.5, 0.05, size=(30, FEATURE_DIM)).astype(np.float32)
    window[:, LEFT_SHOULDER_OFFSET : LEFT_SHOULDER_OFFSET + 3] = (0.4, 0.5, 0.0)
    window[:, RIGHT_SHOULDER_OFFSET : RIGHT_SHOULDER_OFFSET + 3] = (0.6, 0.5, 0.0)
    return window
