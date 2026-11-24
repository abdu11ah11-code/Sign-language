"""Landmark-space data augmentation.

All transforms operate on *normalized* windows (chest at the origin,
shoulder width = 1), which makes rotation/scale trivially correct: they act
around the origin. Zero-filled (undetected) landmarks are left at exactly
zero by every transform.

Windows come in two layouts (see :data:`LAYOUTS`):

* ``holistic`` / ``hand_shape``: ``(T, 162)``, 54 points × (x, y, z)
* ``holistic_lips``: ``(T, 188)``, 94 points × (x, y), lips last

Transforms (config §augmentation):
    rotate   — ± in-plane rotation of (x, y); simulates camera tilt
    scale    — ± uniform scale; simulates distance variation
    mirror   — horizontal flip + left/right swap of hands, symmetric pose
               points and symmetric lip points; valid because the lexicons
               here are not handedness-sensitive (documented limitation in
               README — disable for signs where it matters)
    jitter   — random frame drop/duplicate then resample; simulates
               signing-speed variation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sign_bridge.capture.holistic import (
    LIP_LANDMARK_INDICES,
    LIP_MIRROR_PAIRS,
    POSE_LANDMARK_INDICES,
)
from sign_bridge.data.preprocess import resample_clip
from sign_bridge.typing import NDArray

_N_HAND = 21
_N_POSE = len(POSE_LANDMARK_INDICES)
_N_LIPS = len(LIP_LANDMARK_INDICES)


@dataclass(frozen=True)
class Layout:
    """Where each body part sits in a normalized window, in *points*."""

    coords: int                    # values per point: 3 (x, y, z) or 2 (x, y)
    left: slice
    right: slice
    pose: slice
    lips: slice | None = None


_BASE = Layout(3, slice(0, 21), slice(21, 42), slice(42, 54))
#: Layout per feature mode.
LAYOUTS: dict[str, Layout] = {
    "holistic": _BASE,
    "hand_shape": _BASE,
    "holistic_lips": Layout(2, slice(0, 21), slice(21, 42), slice(42, 54), slice(54, 94)),
    "holistic_lips_clip": Layout(2, slice(0, 21), slice(21, 42), slice(42, 54), slice(54, 94)),
}

# Symmetric pose pairs (MediaPipe indices) → positions inside the pose block.
_POSE_POS = {idx: i for i, idx in enumerate(POSE_LANDMARK_INDICES)}
_SYMMETRIC_POSE_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (_POSE_POS[a], _POSE_POS[b])
    for a, b in ((2, 5), (11, 12), (13, 14), (15, 16), (23, 24))
)
# Symmetric lip pairs → positions inside the lip block.
_LIP_POS = {idx: i for i, idx in enumerate(LIP_LANDMARK_INDICES)}
_SYMMETRIC_LIP_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (_LIP_POS[a], _LIP_POS[b]) for a, b in LIP_MIRROR_PAIRS
)


def mirror_permutation(layout: Layout) -> NDArray:
    """Point order after a horizontal mirror (hands swapped, pairs swapped)."""
    n = (layout.lips.stop if layout.lips else layout.pose.stop)
    perm = np.arange(n)
    perm[layout.left] = np.arange(layout.right.start, layout.right.stop)
    perm[layout.right] = np.arange(layout.left.start, layout.left.stop)
    blocks = [(layout.pose.start, _SYMMETRIC_POSE_PAIRS)]
    if layout.lips is not None:
        blocks.append((layout.lips.start, _SYMMETRIC_LIP_PAIRS))
    for start, pairs in blocks:
        for a, b in pairs:
            perm[start + a], perm[start + b] = start + b, start + a
    return perm


def _points(window: NDArray, layout: Layout) -> NDArray:
    return window.reshape(window.shape[0], -1, layout.coords)


def _detected_mask(pts: NDArray) -> NDArray:
    """``(T, P)`` bool mask of points that are not zero-filled."""
    return np.any(pts != 0.0, axis=2)  # type: ignore[no-any-return]


def rotate(
    window: NDArray, degrees: float, rng: np.random.Generator, layout: Layout = _BASE
) -> NDArray:
    """Rotate (x, y) of all detected points by a random angle in ±degrees."""
    theta = np.deg2rad(rng.uniform(-degrees, degrees))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)

    pts = _points(window, layout).copy()
    mask = _detected_mask(pts)
    xy = pts[..., :2]
    pts[..., :2] = np.where(mask[..., None], xy @ rot.T, xy)
    return pts.reshape(window.shape).astype(np.float32)


def scale(
    window: NDArray, pct: float, rng: np.random.Generator, layout: Layout = _BASE
) -> NDArray:
    """Uniformly scale all detected points by a random factor in ±pct%."""
    factor = 1.0 + rng.uniform(-pct, pct) / 100.0
    pts = _points(window, layout).copy()
    mask = _detected_mask(pts)
    pts = np.where(mask[..., None], pts * factor, pts)
    return pts.reshape(window.shape).astype(np.float32)


def mirror(window: NDArray, layout: Layout = _BASE) -> NDArray:
    """Horizontally mirror a window: negate x and swap left/right parts."""
    pts = _points(window, layout)[:, mirror_permutation(layout)].copy()
    mask = _detected_mask(pts)
    pts[..., 0] = np.where(mask, -pts[..., 0], pts[..., 0])
    return pts.reshape(window.shape).astype(np.float32)


def temporal_jitter(
    window: NDArray, max_frames: int, rng: np.random.Generator
) -> NDArray:
    """Randomly drop and duplicate up to ``max_frames``, then resample back."""
    if max_frames <= 0:
        return window.astype(np.float32)
    t = window.shape[0]
    k = int(rng.integers(0, max_frames + 1))
    if k == 0:
        return window.astype(np.float32)

    keep = np.sort(rng.choice(t, size=t - k, replace=False))
    jittered = window[keep]
    dup_idx = np.sort(rng.integers(0, jittered.shape[0], size=k))
    jittered = np.insert(jittered, dup_idx, jittered[dup_idx], axis=0)
    return resample_clip(jittered, t)


def augment_dataset(
    x: NDArray,
    y: NDArray,
    copies_per_sample: int = 2,
    rotation_deg: float = 15.0,
    scale_pct: float = 10.0,
    use_mirror: bool = True,
    jitter_frames: int = 3,
    seed: int = 42,
    feature_mode: str = "holistic",
) -> tuple[NDArray, NDArray]:
    """Expand a dataset with randomized augmented copies.

    Args:
        x: ``(N, T, F)`` normalized windows in the layout of ``feature_mode``.
        y: ``(N,)`` labels.
        copies_per_sample: Augmented copies per original (0 = no-op).
        rotation_deg / scale_pct / use_mirror / jitter_frames: Transform
            magnitudes; see module docstring.
        seed: RNG seed for reproducible training sets.
        feature_mode: Selects the window layout (:data:`LAYOUTS`).

    Returns:
        ``(x_aug, y_aug)`` — originals first, copies appended.
    """
    if copies_per_sample <= 0:
        return x, y
    layout = LAYOUTS[feature_mode]
    rng = np.random.default_rng(seed)
    new_x: list[NDArray] = [x]
    new_y: list[NDArray] = [y]

    for _ in range(copies_per_sample):
        batch = []
        for window in x:
            w = rotate(window, rotation_deg, rng, layout)
            w = scale(w, scale_pct, rng, layout)
            if use_mirror and rng.random() < 0.5:
                w = mirror(w, layout)
            w = temporal_jitter(w, jitter_frames, rng)
            batch.append(w)
        new_x.append(np.stack(batch))
        new_y.append(y.copy())

    return np.concatenate(new_x), np.concatenate(new_y)
