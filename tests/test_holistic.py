"""Tests for landmark feature extraction (no real MediaPipe needed)."""

from __future__ import annotations

import numpy as np

from sign_bridge.capture.holistic import (
    FEATURE_DIM,
    LEFT_HAND_SLICE,
    LIP_LANDMARK_INDICES,
    LIPS_SLICE,
    RAW_DIM,
    RIGHT_HAND_SLICE,
    extract_features,
)
from tests.conftest import FakeHolisticResults


def test_layout_sizes() -> None:
    assert FEATURE_DIM == 162          # hands + pose (the original layout)
    assert len(LIP_LANDMARK_INDICES) == 40
    assert RAW_DIM == 282              # + 40 lip points


def test_extract_full_detection_shape(fake_results: FakeHolisticResults) -> None:
    feats = extract_features(fake_results)
    assert feats.shape == (RAW_DIM,)
    assert feats.dtype == np.float32
    assert np.any(feats[LIPS_SLICE] != 0.0)


def test_missing_left_hand_is_zero_filled() -> None:
    results = FakeHolisticResults(left=False, right=True, pose=True)
    feats = extract_features(results)
    assert np.all(feats[LEFT_HAND_SLICE] == 0.0)
    assert np.any(feats[RIGHT_HAND_SLICE] != 0.0)


def test_missing_pose_zero_fills_pose_block() -> None:
    results = FakeHolisticResults(left=True, right=True, pose=False)
    feats = extract_features(results)
    assert np.all(feats[126:162] == 0.0)


def test_missing_face_zero_fills_lips() -> None:
    feats = extract_features(FakeHolisticResults(face=False))
    assert np.all(feats[LIPS_SLICE] == 0.0)
    assert np.any(feats[:FEATURE_DIM] != 0.0)


def test_no_detection_all_zero() -> None:
    results = FakeHolisticResults(left=False, right=False, pose=False, face=False)
    feats = extract_features(results)
    assert np.all(feats == 0.0)
