"""Tests for trainer helpers that guard small or imbalanced datasets."""

from __future__ import annotations

import numpy as np
import pytest

from sign_bridge.data.preprocess import Dataset
from sign_bridge.models.trainer import _check_dataset, _class_weights, _split


def test_split_small_dataset_stays_stratified() -> None:
    x = np.zeros((12, 30, 162), dtype=np.float32)
    y = np.repeat(np.arange(4), 3)  # 4 signs x 3 clips: sklearn's default split fails here
    x_a, x_b, y_a, y_b = _split(x, y, 0.15, seed=0)
    assert sorted(set(y_b.tolist())) == [0, 1, 2, 3]
    assert len(y_a) + len(y_b) == 12


def test_class_weights_cover_every_class() -> None:
    weights = _class_weights(np.array([0, 0, 2]), num_classes=3)
    assert set(weights) == {0, 1, 2}
    assert weights[1] == 1.0


def test_check_dataset_rejects_too_few_clips() -> None:
    ds = Dataset(x=np.zeros((4, 30, 162)), y=np.array([0, 0, 0, 1]), labels=["A", "B"])
    with pytest.raises(SystemExit, match="B"):
        _check_dataset(ds)
