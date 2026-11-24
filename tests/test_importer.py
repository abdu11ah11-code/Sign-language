"""Tests for the dataset importer (no MediaPipe needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sign_bridge.capture.holistic import FEATURE_DIM, RIGHT_HAND_SLICE
from sign_bridge.data.importer import discover, label_from_folder, static_clip, to_frame_units
from sign_bridge.data.preprocess import hand_shape_frame
from sign_bridge.language.psl_lexicon import URDU_LETTERS


def test_label_from_folder_ualpha40_names() -> None:
    assert label_from_folder("Alif-Original") == ("ALIF", False)
    assert label_from_folder("1-Hay-Augmented") == ("1_HAY", True)
    assert label_from_folder("Jeem") == ("JEEM", False)
    assert label_from_folder("Nuungh-Original") == ("NUUNGH", False)


def test_ualpha40_labels_have_urdu_letters() -> None:
    for folder in ("Alif-Original", "1-Hay-Original", "2-Hay", "Alifmad", "Byeh-Original"):
        assert label_from_folder(folder)[0] in URDU_LETTERS


def test_discover_merges_and_skips_augmented(tmp_path: Path) -> None:
    for folder, name in [("Alif-Original", "a.jpg"), ("Alif-Augmented", "b.jpg"),
                         ("Jeem", "c.mp4"), ("Notes", "readme.txt")]:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_bytes(b"x")
    groups = discover(tmp_path)
    assert sorted(groups) == ["ALIF", "JEEM"]
    assert len(groups["ALIF"]) == 1
    assert len(discover(tmp_path, include_augmented=True)["ALIF"]) == 2


def test_to_frame_units_undoes_photo_aspect() -> None:
    # A square in a 2:1 (wide) photo is normalized to half its height in x.
    pts = np.array([[0.25, 0.5, 0.0], [0.75, 0.5, 0.1]], dtype=np.float32)
    out = to_frame_units(pts, width=200, height=100, aspect=4 / 3)
    assert np.isclose(out[1, 0] - out[0, 0], 0.5 * 2.0 / (4 / 3))
    assert np.isclose(out[0, 1], 0.5)


def test_static_clip_is_a_held_hand_shape() -> None:
    rng = np.random.default_rng(0)
    hand = rng.random((21, 3)).astype(np.float32)
    clip = static_clip(hand, 30, rng)
    assert clip.shape == (30, FEATURE_DIM)
    assert np.all(clip[:, :63] == 0) and np.all(clip[:, 126:] == 0)
    shapes = np.stack([hand_shape_frame(f)[RIGHT_HAND_SLICE] for f in clip])
    assert np.max(np.std(shapes, axis=0)) < 0.05  # small tremor only
