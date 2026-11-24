"""Landmark preprocessing: normalization, resampling, dataset assembly.

Pipeline (mirrors README architecture diagram):

    raw CSV clips ──resample──▶ (30, 162) windows ──normalize──▶ model input

Normalization (per frame):
    1. Center: subtract the chest point = midpoint of the two shoulders.
       Signs become translation-invariant (signer can stand anywhere).
    2. Scale: divide by shoulder width. Signs become distance-invariant
       (signer can be near or far from the camera).
    Zero-filled landmarks (undetected hand) stay exactly zero so the model
    can learn "hand absent" as signal rather than seeing noise.

Feature modes (chosen at training time, saved with the model):

    holistic    the scheme above: both hands + upper body, relative to the
                chest. Needs the signer's shoulders in view. Default; used
                for clips recorded with sign-collect.
    hand_shape  one hand only, relative to its own wrist and scaled by its
                own size; body ignored. Works for close-up hand photos
                (e.g. the UAlpha40 PSL alphabet dataset) and for static
                hand shapes such as fingerspelled letters. The most raised
                hand is the signing hand; it always occupies the right-hand
                slot so left- and right-handed signers look the same.
    holistic_lips
                both hands + upper body + 40 lip points, centred on the
                chest and scaled by shoulder width like ``holistic``, kept
                in 2D (x, y): 94 points × 2 = 188 values per frame. The
                representation used by the top solutions of Google's
                Isolated Sign Language Recognition competition (ASL Signs
                dataset); mouth shapes carry meaning in ASL. Depth (z) is
                dropped because MediaPipe's z is noisy.

``holistic`` and ``hand_shape`` read only the first 162 raw values, so
models and recordings made before lips were added keep working.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sign_bridge.capture.holistic import (
    FEATURE_DIM,
    LEFT_SHOULDER_OFFSET,
    RAW_DIM,
    RIGHT_SHOULDER_OFFSET,
)
from sign_bridge.logging_conf import get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)

_EPS = 1e-6

#: Supported feature modes (see module docstring).
FEATURE_MODES = ("holistic", "hand_shape", "holistic_lips", "holistic_lips_clip")
#: Values per normalized frame for each mode (= the model's input width).
MODE_DIMS = {"holistic": FEATURE_DIM, "hand_shape": FEATURE_DIM, "holistic_lips": 188,
             "holistic_lips_clip": 188}
#: Modes normalized per clip (statistics over the whole clip, not per frame).
CLIP_MODES = ("holistic_lips_clip",)
#: Vertical nose → shoulder-line distance, in shoulder widths, that
#: ``holistic_lips_clip`` rescales every clip to (see that function).
CANONICAL_NECK = 0.6
#: Shoulder width assumed when a clip has no tracked shoulders at all.
_DEFAULT_SHOULDER_WIDTH = 0.25
#: Offset of the nose (first of the 12 pose points) in the raw layout.
NOSE_OFFSET = FEATURE_DIM - 36
#: Accepted raw clip widths: 162 (recorded before lips existed) or 282.
RAW_WIDTHS = (FEATURE_DIM, RAW_DIM)
_HAND = 21 * 3          # values per hand block
_LEFT = slice(0, _HAND)
_RIGHT = slice(_HAND, 2 * _HAND)


# ── geometric normalization ──────────────────────────────────────────────
def mode_dim(mode: str) -> int:
    """Width of one normalized frame (the model's input width) for ``mode``."""
    if mode not in MODE_DIMS:
        raise ValueError(f"Unknown feature mode {mode!r}; expected one of {FEATURE_MODES}")
    return MODE_DIMS[mode]


def pad_raw(clip: NDArray) -> NDArray:
    """Pad raw frames recorded before lips existed (162 wide) to 282 wide."""
    if clip.shape[-1] == RAW_DIM:
        return clip.astype(np.float32, copy=False)
    if clip.shape[-1] != FEATURE_DIM:
        raise ValueError(f"Raw frames must be {FEATURE_DIM} or {RAW_DIM} wide, got {clip.shape}")
    pad = np.zeros((*clip.shape[:-1], RAW_DIM - FEATURE_DIM), dtype=np.float32)
    return np.concatenate([clip.astype(np.float32), pad], axis=-1)


def holistic_lips_batch(raw: NDArray) -> NDArray:
    """``holistic_lips`` features for any batch of raw frames.

    Vectorized so the same code normalizes one live frame or a whole
    dataset (the Kaggle training notebook calls it on ``(N, T, 282)``).

    Args:
        raw: ``(..., 162 or 282)`` raw frames.

    Returns:
        ``(..., 188)`` float32: (x, y) of 94 points centred on the chest and
        divided by shoulder width; undetected points stay exactly 0. Frames
        without shoulders keep raw (x, y), as :func:`normalize_frame` does.
    """
    raw = pad_raw(np.asarray(raw))
    pts = raw.reshape(*raw.shape[:-1], -1, 3)[..., :2]       # (..., 94, 2)
    ls = raw[..., LEFT_SHOULDER_OFFSET : LEFT_SHOULDER_OFFSET + 2]
    rs = raw[..., RIGHT_SHOULDER_OFFSET : RIGHT_SHOULDER_OFFSET + 2]
    has_ref = np.any(ls != 0.0, axis=-1) | np.any(rs != 0.0, axis=-1)
    center = np.where(has_ref[..., None], (ls + rs) / 2.0, 0.0)
    scale = np.linalg.norm(ls - rs, axis=-1)
    scale = np.where(has_ref & (scale > _EPS), scale, 1.0)
    detected = np.any(pts != 0.0, axis=-1, keepdims=True)
    out = np.where(detected, (pts - center[..., None, :]) / scale[..., None, None], 0.0)
    return out.reshape(*raw.shape[:-1], -1).astype(np.float32)


def holistic_lips_clip(raw: NDArray) -> NDArray:
    """``holistic_lips_clip`` features: per-clip, camera-shape-invariant.

    MediaPipe divides x by the image *width* and y by the image *height*, so
    the same body looks taller or flatter depending on the camera's shape
    (a portrait phone video vs. a 4:3 laptop webcam differ by more than 2×
    vertically). This mode undoes that and steadies the coordinates:

    1. centre = median shoulder midpoint over the clip; scale = median
       shoulder width (per clip, so the signer's own body motion is kept);
    2. y is additionally rescaled so the median vertical nose → shoulder
       distance equals :data:`CANONICAL_NECK` shoulder widths — a body
       proportion that is roughly constant across people, so what remains
       is the camera's distortion, which this removes.

    Args:
        raw: ``(T, 162|282)`` clip or ``(N, T, 162|282)`` batch of clips.

    Returns:
        ``(T, 188)`` / ``(N, T, 188)`` float32; undetected points stay 0.
    """
    raw = pad_raw(np.asarray(raw))
    single = raw.ndim == 2
    if single:
        raw = raw[None]
    pts = raw.reshape(*raw.shape[:-1], -1, 3)[..., :2]               # (N, T, 94, 2)
    ls = raw[..., LEFT_SHOULDER_OFFSET : LEFT_SHOULDER_OFFSET + 2]
    rs = raw[..., RIGHT_SHOULDER_OFFSET : RIGHT_SHOULDER_OFFSET + 2]
    nose = raw[..., NOSE_OFFSET : NOSE_OFFSET + 2]
    has_sh = np.any(ls != 0.0, axis=-1) & np.any(rs != 0.0, axis=-1)  # (N, T)
    mid = (ls + rs) / 2.0
    width_f = np.linalg.norm(ls - rs, axis=-1)
    has_sh &= width_f > _EPS
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # all-NaN clips
        center = np.nanmedian(np.where(has_sh[..., None], mid, np.nan), axis=1)   # (N, 2)
        width = np.nanmedian(np.where(has_sh, width_f, np.nan), axis=1)           # (N,)
        has_nose = has_sh & np.any(nose != 0.0, axis=-1)
        neck_f = np.where(has_nose, (mid[..., 1] - nose[..., 1]) / np.where(has_sh, width_f, 1.0),
                          np.nan)
        neck = np.nanmedian(neck_f, axis=1)                                       # (N,)
    center = np.where(np.isfinite(center), center, 0.5)
    width = np.where(np.isfinite(width) & (width > _EPS), width, _DEFAULT_SHOULDER_WIDTH)
    k_y = np.where(np.isfinite(neck) & (neck > 0.05), CANONICAL_NECK / np.maximum(neck, 0.05), 1.0)
    k_y = np.clip(k_y, 0.4, 2.5)
    scale = np.stack([1.0 / width, k_y / width], axis=-1)               # (N, 2)
    detected = np.any(pts != 0.0, axis=-1, keepdims=True)
    out = (pts - center[:, None, None, :]) * scale[:, None, None, :]
    out = np.where(detected, out, 0.0).reshape(*raw.shape[:-1], -1).astype(np.float32)
    return out[0] if single else out


def normalize_frame(frame: NDArray) -> NDArray:
    """Center one 162-dim frame on the chest and scale by shoulder width.

    Args:
        frame: ``(162,)`` raw feature vector.

    Returns:
        ``(162,)`` normalized copy. If shoulders were not detected the
        frame is returned unchanged (cannot normalize without a reference).
        A 282-wide raw frame is accepted; its lip block is ignored.
    """
    frame = frame[:FEATURE_DIM]
    ls = frame[LEFT_SHOULDER_OFFSET : LEFT_SHOULDER_OFFSET + 3]
    rs = frame[RIGHT_SHOULDER_OFFSET : RIGHT_SHOULDER_OFFSET + 3]
    if not (np.any(ls) or np.any(rs)):
        return frame.astype(np.float32, copy=True)

    center = (ls + rs) / 2.0
    scale = float(np.linalg.norm(ls[:2] - rs[:2]))
    scale = scale if scale > _EPS else 1.0

    pts = frame.reshape(-1, 3)
    detected = np.any(pts != 0.0, axis=1)  # keep zero-filled points at zero
    out = pts.copy()
    out[detected] = (pts[detected] - center) / scale
    return out.reshape(-1).astype(np.float32)


def hand_shape_frame(frame: NDArray) -> NDArray:
    """Hand-shape features: the signing hand relative to its own wrist.

    Picks the most raised visible hand (smallest wrist ``y`` in image
    coordinates), translates it so the wrist is the origin and divides by
    the hand's own extent. The result sits in the right-hand slot; the left
    slot and the pose block are zero.

    Args:
        frame: ``(162,)`` raw feature vector.

    Returns:
        ``(162,)`` hand-shape vector (all zeros when no hand is visible).
    """
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    frame = frame[:FEATURE_DIM]
    hands = [frame[sl].reshape(21, 3) for sl in (_LEFT, _RIGHT)]
    visible = [h for h in hands if np.any(h != 0.0)]
    if not visible:
        return out
    hand = min(visible, key=lambda h: float(h[0, 1]))
    rel = hand - hand[0]
    extent = float(np.max(np.linalg.norm(rel[:, :2], axis=1)))
    if extent < _EPS:
        return out
    out[_RIGHT] = (rel / extent).reshape(-1)
    return out


def canonical_hand_window(window: NDArray) -> NDArray:
    """Move a lone left-slot hand into the right slot (after mirroring).

    Mirror augmentation swaps the hand slots; in ``hand_shape`` mode the
    signing hand must always sit in the right slot, as it does live.
    """
    out = window.copy()
    right_empty = ~np.any(out[:, _RIGHT] != 0.0, axis=1)
    left_present = np.any(out[:, _LEFT] != 0.0, axis=1)
    move = right_empty & left_present
    out[move, _RIGHT] = out[move, _LEFT]
    out[move, _LEFT] = 0.0
    return out


def normalize_frame_mode(frame: NDArray, mode: str = "holistic") -> NDArray:
    """Normalize one frame with the given feature mode."""
    if mode == "hand_shape":
        return hand_shape_frame(frame)
    if mode == "holistic":
        return normalize_frame(frame)
    if mode == "holistic_lips":
        return holistic_lips_batch(frame[None, :])[0]  # type: ignore[no-any-return]
    if mode == "holistic_lips_clip":
        return holistic_lips_clip(frame[None, :])[0]  # type: ignore[no-any-return]
    raise ValueError(f"Unknown feature mode {mode!r}; expected one of {FEATURE_MODES}")


def normalize_window(window: NDArray, mode: str = "holistic") -> NDArray:
    """Normalize every frame of a ``(T, 162 or 282)`` raw window with ``mode``."""
    if mode == "holistic_lips":
        return holistic_lips_batch(window)
    if mode == "holistic_lips_clip":
        return holistic_lips_clip(window)
    return np.stack([normalize_frame_mode(f, mode) for f in window]).astype(np.float32)


# ── temporal resampling ──────────────────────────────────────────────────
def resample_clip(clip: NDArray, target_frames: int) -> NDArray:
    """Uniformly resample a clip to a fixed number of frames.

    Collector clips are ~60 frames (2 s); models consume 30. Uniform index
    selection preserves motion shape; clips shorter than the target are
    padded by repeating the final frame (hold), matching how a signer
    holds the final posture of a sign.

    Args:
        clip: ``(T, F)`` array, any T ≥ 1.
        target_frames: Output length.

    Returns:
        ``(target_frames, F)`` float32 array.
    """
    t = clip.shape[0]
    if t == target_frames:
        return clip.astype(np.float32)
    if t > target_frames:
        idx = np.linspace(0, t - 1, target_frames).round().astype(int)
        return clip[idx].astype(np.float32)  # type: ignore[no-any-return]
    pad = np.repeat(clip[-1:], target_frames - t, axis=0)
    return np.concatenate([clip, pad]).astype(np.float32)  # type: ignore[no-any-return]


# ── dataset loading ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Dataset:
    """A loaded, normalized training set.

    Attributes:
        x: ``(N, window, mode_dim(mode))`` float32 inputs.
        y: ``(N,)`` int class indices.
        labels: Sorted gloss names; index = class id.
    """

    x: NDArray
    y: NDArray
    labels: list[str]


def load_dataset(
    language_dir: Path, window_size: int = 30, feature_mode: str = "holistic"
) -> Dataset:
    """Load all landmark CSVs under ``data/<lang>/<LABEL>/*.csv``.

    Each CSV holds one recorded clip: rows = frames, 282 columns (or 162
    for clips recorded before lip landmarks were added).
    Clips are resampled to ``window_size`` frames and normalized.

    Args:
        language_dir: e.g. ``Path("data/asl")``.
        window_size: Frames per training window.
        feature_mode: one of :data:`FEATURE_MODES` (see module docs).

    Returns:
        :class:`Dataset` with stable, sorted label ordering.

    Raises:
        FileNotFoundError: If the directory has no labeled CSVs at all.
    """
    label_dirs = (
        sorted(d for d in language_dir.iterdir() if d.is_dir())
        if language_dir.is_dir()
        else []
    )
    xs: list[NDArray] = []
    ys: list[int] = []
    labels = [d.name for d in label_dirs if any(d.glob("*.csv"))]
    if not labels:
        raise FileNotFoundError(
            f"No training data under {language_dir}. "
            "Record some with: sign-collect --label HELLO --mode "
            f"{language_dir.name} --samples 50"
        )

    for class_id, label in enumerate(labels):
        files = sorted((language_dir / label).glob("*.csv"))
        for csv_path in files:
            clip = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float32)
            if clip.ndim != 2 or clip.shape[1] not in RAW_WIDTHS:
                log.warning("skipping_malformed_clip", path=str(csv_path), shape=clip.shape)
                continue
            if clip.shape[0] < 2:
                log.warning("skipping_empty_clip", path=str(csv_path))
                continue
            window = normalize_window(resample_clip(clip, window_size), feature_mode)
            xs.append(window)
            ys.append(class_id)
        log.info("loaded_label", label=label, clips=len(files))

    if not xs:
        raise FileNotFoundError(
            f"Found CSV files under {language_dir} but none had {RAW_DIM} columns. "
            "Re-record them with sign-collect."
        )
    present = sorted(set(ys))
    if len(present) != len(labels):
        # A label folder held only malformed clips: drop it and re-index.
        remap = {old: new for new, old in enumerate(present)}
        labels = [labels[i] for i in present]
        ys = [remap[v] for v in ys]
    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    log.info("dataset_loaded", samples=int(x.shape[0]), classes=len(labels))
    return Dataset(x=x, y=y, labels=labels)


# ── optional flattened scaler (second-stage standardization) ─────────────
#: Smallest per-feature std the scaler divides by. Chest-centred,
#: shoulder-scaled features include values that are (almost) constant by
#: construction — the shoulders always land at (±0.5, 0). Dividing by their
#: ~0 spread turns any small change (augmentation, tracking noise) into
#: values of millions and swamps every other feature.
MIN_FEATURE_SCALE = 1e-2


def fit_scaler(x: NDArray, min_scale: float = MIN_FEATURE_SCALE) -> StandardScaler:
    """Fit a StandardScaler over all frames of all windows.

    Geometric normalization already does the heavy lifting; this final
    zero-mean/unit-variance pass helps optimization slightly and is kept
    for parity with the saved ``scalers.joblib`` artifact contract. Per-
    feature scales are floored at ``min_scale`` (see
    :data:`MIN_FEATURE_SCALE`).
    """
    scaler = StandardScaler()
    scaler.fit(x.reshape(-1, x.shape[-1]))
    return floor_scaler(scaler, min_scale)


def floor_scaler(scaler: StandardScaler, min_scale: float = MIN_FEATURE_SCALE) -> StandardScaler:
    """Clamp a fitted scaler's per-feature std to at least ``min_scale``."""
    scaler.scale_ = np.maximum(scaler.scale_, min_scale)
    scaler.var_ = scaler.scale_ ** 2
    return scaler


def apply_scaler(x: NDArray, scaler: StandardScaler) -> NDArray:
    """Transform ``(N, T, F)`` windows with a fitted scaler, preserving shape."""
    n, t, f = x.shape
    return scaler.transform(x.reshape(-1, f)).reshape(n, t, f).astype(np.float32)  # type: ignore[no-any-return]


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    """Persist a scaler with joblib (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> StandardScaler:
    """Load a scaler saved by :func:`save_scaler`."""
    loaded: StandardScaler = joblib.load(path)
    return loaded


# ── label persistence ────────────────────────────────────────────────────
def save_labels(labels: list[str], path: Path) -> None:
    """Write the class-index→gloss mapping next to a model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_labels(path: Path) -> list[str]:
    """Read labels written by :func:`save_labels`."""
    return list(json.loads(path.read_text(encoding="utf-8")))
