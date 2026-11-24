"""Tests for the lips feature mode, mirror permutation and new model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sign_bridge.capture.holistic import FEATURE_DIM, RAW_DIM
from sign_bridge.data.augment import LAYOUTS, augment_dataset, mirror, mirror_permutation
from sign_bridge.data.preprocess import (
    holistic_lips_batch,
    mode_dim,
    normalize_frame,
    normalize_window,
    pad_raw,
)


def _raw_frame() -> np.ndarray:
    raw = np.zeros(RAW_DIM, dtype=np.float32)
    raw[126 + 4 * 3 : 126 + 4 * 3 + 3] = (0.6, 0.5, 0.0)   # left shoulder
    raw[126 + 5 * 3 : 126 + 5 * 3 + 3] = (0.4, 0.5, 0.0)   # right shoulder
    raw[63:66] = (0.5, 0.3, 0.1)                           # right wrist
    raw[162:165] = (0.52, 0.2, 0.0)                        # first lip point
    return raw


def test_mode_dims() -> None:
    assert mode_dim("holistic") == FEATURE_DIM
    assert mode_dim("hand_shape") == FEATURE_DIM
    assert mode_dim("holistic_lips") == 188
    with pytest.raises(ValueError):
        mode_dim("nope")


def test_holistic_lips_centres_and_scales() -> None:
    out = holistic_lips_batch(_raw_frame()[None])[0]
    assert out.shape == (188,)
    assert np.allclose(out[42:44], (0.0, -1.0))           # wrist: 0.2 above chest / 0.2
    assert np.allclose(out[108:110], (0.1, -1.5))         # lip point
    assert np.all(out[:42] == 0.0)                        # left hand absent stays 0


def test_old_162_wide_clips_still_work() -> None:
    raw = _raw_frame()
    old = raw[:FEATURE_DIM]
    assert pad_raw(old[None]).shape == (1, RAW_DIM)
    assert np.allclose(normalize_frame(raw), normalize_frame(old))
    lips = normalize_window(old[None], "holistic_lips")[0]
    assert np.all(lips[108:] == 0.0)


@pytest.mark.parametrize("mode", ["holistic", "holistic_lips"])
def test_mirror_is_an_involution(mode: str) -> None:
    layout = LAYOUTS[mode]
    perm = mirror_permutation(layout)
    assert sorted(perm.tolist()) == list(range(len(perm)))
    assert np.all(perm[perm] == np.arange(len(perm)))
    w = np.random.default_rng(0).normal(size=(10, mode_dim(mode))).astype(np.float32)
    assert np.allclose(mirror(mirror(w, layout), layout), w)


def test_augment_lips_layout() -> None:
    x = np.random.default_rng(1).normal(size=(3, 20, 188)).astype(np.float32)
    xa, ya = augment_dataset(x, np.arange(3), copies_per_sample=1, feature_mode="holistic_lips")
    assert xa.shape == (6, 20, 188) and ya.shape == (6,)


def test_conv_transformer_builds_and_roundtrips(tmp_path: Path) -> None:
    tf = pytest.importorskip("tensorflow")
    from sign_bridge.models.registry import build_from_spec, make_spec

    spec = make_spec("conv_transformer", 5, 16, 188,
                     {"dim": 32, "kernel_size": 5, "num_stacks": 1, "convs_per_stack": 1,
                      "num_heads": 2})
    model = build_from_spec(spec)
    x = np.random.default_rng(2).normal(size=(4, 16, 188)).astype(np.float32)
    model.fit(x, np.arange(4) % 5, epochs=1, verbose=0)
    path = tmp_path / "m.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False)
    assert np.allclose(model.predict_on_batch(x), loaded.predict_on_batch(x), atol=1e-5)


def test_fit_scaler_floors_constant_features() -> None:
    from sign_bridge.data.preprocess import MIN_FEATURE_SCALE, fit_scaler

    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 10, 4)).astype(np.float32)
    # Nearly constant column, e.g. a shoulder after normalization (tiny jitter).
    x[..., 0] = 0.5 + rng.normal(scale=1e-4, size=x.shape[:-1])
    scaler = fit_scaler(x)
    assert scaler.scale_[0] == MIN_FEATURE_SCALE
    out = scaler.transform(np.array([[0.52, 0.0, 0.0, 0.0]], dtype=np.float32))
    assert abs(out[0, 0]) < 5.0  # a small wobble stays small


def _body_clip(t: int = 20, y_squash: float = 1.0, x_shift: float = 0.0) -> np.ndarray:
    """Raw clip of a moving right hand; ``y_squash`` mimics another camera shape."""
    from sign_bridge.capture.holistic import RAW_DIM

    clip = np.zeros((t, RAW_DIM), np.float32)
    for i in range(t):
        pose = clip[i, 126:162].reshape(12, 3)
        pose[0] = (0.5, 0.30, 0)                       # nose
        pose[4], pose[5] = (0.65, 0.48, 0), (0.35, 0.48, 0)   # shoulders
        hand = clip[i, 63:126].reshape(21, 3)
        hand[:, 0] = 0.40 + 0.01 * np.arange(21) + 0.005 * i
        hand[:, 1] = 0.35 + 0.004 * np.arange(21) - 0.003 * i
        lips = clip[i, 162:282].reshape(40, 3)
        lips[:, 0], lips[:, 1] = np.linspace(0.47, 0.53, 40), 0.34
    nz = clip != 0
    xy = clip.reshape(t, -1, 3)
    xy[..., 1] = np.where(xy[..., 1] != 0, 0.5 + (xy[..., 1] - 0.5) * y_squash, 0)
    xy[..., 0] = np.where(xy[..., 0] != 0, xy[..., 0] + x_shift, 0)
    assert np.array_equal(clip != 0, nz)
    return clip


def test_clip_mode_cancels_camera_shape() -> None:
    from sign_bridge.data.preprocess import holistic_lips_batch, holistic_lips_clip

    a = holistic_lips_clip(_body_clip())
    b = holistic_lips_clip(_body_clip(y_squash=0.45, x_shift=0.07))  # portrait-like squash
    assert a.shape == (20, 188)
    assert np.allclose(a, b, atol=1e-4)
    # the per-frame mode is NOT invariant to camera shape
    pa, pb = holistic_lips_batch(_body_clip()), holistic_lips_batch(_body_clip(y_squash=0.45))
    assert not np.allclose(pa, pb, atol=1e-2)
    # batch form matches single-clip form; missing points stay zero
    batch = holistic_lips_clip(np.stack([_body_clip(), _body_clip(y_squash=0.45)]))
    assert np.allclose(batch[0], a) and np.all(a[:, :42] == 0.0)   # left hand absent


def test_clip_mode_without_body_is_finite() -> None:
    from sign_bridge.capture.holistic import RAW_DIM
    from sign_bridge.data.preprocess import holistic_lips_clip

    clip = np.zeros((10, RAW_DIM), np.float32)
    clip[:, 63:66] = (0.5, 0.5, 0.0)          # only one hand point, no pose
    out = holistic_lips_clip(clip)
    assert np.all(np.isfinite(out))
