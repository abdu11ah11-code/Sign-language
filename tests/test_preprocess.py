"""Tests for preprocessing: normalization, resampling, augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from sign_bridge.capture.holistic import FEATURE_DIM
from sign_bridge.data.augment import augment_dataset, mirror, rotate
from sign_bridge.data.preprocess import normalize_frame, normalize_window, resample_clip


def test_resample_up_and_down() -> None:
    short = np.ones((10, FEATURE_DIM), dtype=np.float32)
    long = np.ones((60, FEATURE_DIM), dtype=np.float32)
    assert resample_clip(short, 30).shape == (30, FEATURE_DIM)
    assert resample_clip(long, 30).shape == (30, FEATURE_DIM)


def test_resample_identity() -> None:
    clip = np.random.rand(30, FEATURE_DIM).astype(np.float32)
    assert np.array_equal(resample_clip(clip, 30), clip)


def test_normalization_is_translation_invariant(synthetic_window: np.ndarray) -> None:
    base = normalize_window(synthetic_window)
    # Apply a rigid translation in x and y to every point (a real camera pan).
    # z is left untouched: MediaPipe z is on a different scale and the shoulder
    # reference has z=0, so only x/y participate in the centering reference.
    shifted = synthetic_window.copy().reshape(synthetic_window.shape[0], -1, 3)
    shifted[..., 0] += 0.1
    shifted[..., 1] += 0.07
    shifted = shifted.reshape(synthetic_window.shape)
    shifted_norm = normalize_window(shifted)
    # Centering removes the global x/y translation → identical normalized output.
    assert np.allclose(base, shifted_norm, atol=1e-4)


def test_normalize_frame_without_shoulders_returns_copy() -> None:
    frame = np.zeros(FEATURE_DIM, dtype=np.float32)
    frame[0:3] = (0.5, 0.5, 0.0)  # a hand point but no shoulders
    out = normalize_frame(frame)
    assert np.array_equal(out, frame)
    assert out is not frame  # must be a copy


def test_augment_grows_dataset_and_preserves_shape(synthetic_window: np.ndarray) -> None:
    x = np.stack([normalize_window(synthetic_window) for _ in range(4)])
    y = np.array([0, 1, 0, 1])
    x_aug, y_aug = augment_dataset(x, y, copies_per_sample=2, seed=1)
    assert x_aug.shape == (12, 30, FEATURE_DIM)  # 4 originals + 2*4 copies
    assert y_aug.shape == (12,)
    assert np.array_equal(x_aug[:4], x)  # originals preserved at the front


def test_rotation_keeps_zero_points_zero() -> None:
    window = np.zeros((5, FEATURE_DIM), dtype=np.float32)
    window[:, 0:3] = (0.3, 0.2, 0.0)  # one detected point
    rng = np.random.default_rng(0)
    out = rotate(window, 15.0, rng)
    # Untouched (zero) points remain zero.
    assert np.all(out[:, 3:] == 0.0)


def test_mirror_swaps_hands() -> None:
    window = np.zeros((2, FEATURE_DIM), dtype=np.float32)
    window[:, 0:3] = (0.2, 0.5, 0.0)    # left-hand point
    window[:, 63:66] = (0.8, 0.5, 0.0)  # right-hand point
    out = mirror(window)
    # After mirroring, left block should carry the (negated-x) right point.
    assert out[0, 0] == pytest.approx(-0.8, abs=1e-5)
