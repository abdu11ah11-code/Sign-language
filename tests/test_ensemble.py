"""Tests for several models answering together (models/ensemble.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sign_bridge.config import load_config
from sign_bridge.inference.sign_to_text import SignToTextPipeline
from sign_bridge.models.ensemble import AveragedModel, IdentityScaler


class _Fixed:
    """A stand-in model: always the same probabilities; remembers what it was given."""

    def __init__(self, probs: list[float], frames: int = 8, features: int = 188) -> None:
        self.probs = np.asarray(probs, dtype=np.float32)
        self.input_shape = (None, frames, features)
        self.seen: list[np.ndarray] = []

    def predict_on_batch(self, x: np.ndarray) -> np.ndarray:
        self.seen.append(np.asarray(x))
        return np.tile(self.probs, (len(x), 1))

    def predict(self, x: np.ndarray, batch_size: int = 256, verbose: int = 0) -> np.ndarray:
        return self.predict_on_batch(x)


class _Shift:
    """A stand-in scaler that adds a constant, so each member's scaling is visible."""

    def __init__(self, by: float) -> None:
        self.by = by

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x + self.by


def test_answer_is_the_weighted_average() -> None:
    a, b = _Fixed([0.8, 0.2, 0.0]), _Fixed([0.2, 0.2, 0.6])
    x = np.zeros((2, 8, 188), np.float32)
    even = AveragedModel([(a, _Shift(0)), (b, _Shift(0))]).predict_on_batch(x)
    assert np.allclose(even, [[0.5, 0.2, 0.3]] * 2)
    leaning = AveragedModel([(a, _Shift(0)), (b, _Shift(0))], weights=[3, 1]).predict(x)
    assert np.allclose(leaning, [[0.65, 0.2, 0.15]] * 2)


def test_each_member_scales_its_own_input() -> None:
    a, b = _Fixed([1, 0]), _Fixed([0, 1])
    model = AveragedModel([(a, _Shift(1.0)), (b, _Shift(-2.0))])
    model.predict_on_batch(np.zeros((1, 8, 188), np.float32))
    assert np.allclose(a.seen[0], 1.0) and np.allclose(b.seen[0], -2.0)
    assert a.seen[0].shape == (1, 8, 188) and a.seen[0].dtype == np.float32
    x = np.ones((3, 4))
    assert IdentityScaler().transform(x) is x


def test_bad_ensembles_are_refused() -> None:
    with pytest.raises(ValueError):
        AveragedModel([])
    with pytest.raises(ValueError):
        AveragedModel([(_Fixed([1, 0], frames=8), _Shift(0)),
                       (_Fixed([1, 0], frames=16), _Shift(0))])
    with pytest.raises(ValueError):
        AveragedModel([(_Fixed([1, 0]), _Shift(0))], weights=[1, 1])
    with pytest.raises(ValueError):
        AveragedModel([(_Fixed([1, 0]), _Shift(0)), (_Fixed([0, 1]), _Shift(0))], weights=[0, 0])


def test_live_recognizer_runs_on_an_ensemble() -> None:
    model = AveragedModel([(_Fixed([0.7, 0.3], frames=12), _Shift(0)),
                           (_Fixed([0.5, 0.5], frames=12), _Shift(0))])
    pipe = SignToTextPipeline(model, ["MOM", "DAD"], IdentityScaler(), load_config(None),
                              feature_mode="holistic_lips_clip", mode="window")
    assert pipe._window_size == 12                      # the members' input length wins
    assert pipe.model_count == 2
    pipe._frames.extend((i / 10, np.zeros(188, np.float32)) for i in range(12))
    label, conf, _ = pipe._predict(1.2)
    assert label == "MOM" and conf == pytest.approx(0.6)


def _save_tiny_model(folder: Path, stem: str, seed: int) -> Any:
    pytest.importorskip("tensorflow")
    from sign_bridge.data.preprocess import fit_scaler, save_labels, save_scaler
    from sign_bridge.models.registry import build_from_spec, make_spec

    spec = make_spec("conv_transformer", 3, 8, 188,
                     {"dim": 16, "kernel_size": 3, "num_stacks": 1, "convs_per_stack": 1,
                      "num_heads": 2})
    model = build_from_spec(spec)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(6, 8, 188)).astype(np.float32)
    model.fit(x, np.arange(6) % 3, epochs=1, verbose=0)
    model.save(folder / f"{stem}.keras")
    save_labels(["MOM", "DAD", "YES"], folder / f"{stem}.labels.json")
    scaler = fit_scaler(x + seed)
    save_scaler(scaler, folder / f"{stem}.scaler.joblib")
    return model, scaler


def test_pointer_can_name_an_ensemble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sign_bridge.models.evaluator import load_latest_model

    models = tmp_path / "models"
    models.mkdir()
    first = _save_tiny_model(models, "asl_a", 1)
    second = _save_tiny_model(models, "asl_b", 2)
    config = tmp_path / "config.yaml"
    config.write_text("paths:\n  models_dir: models\n", encoding="utf-8")
    monkeypatch.setenv("SIGN_BRIDGE_CONFIG", str(config))      # relative paths resolve from it
    pointer = {"stem": "asl_b", "ensemble": ["asl_a", "asl_b"],
               "feature_mode": "holistic_lips_clip"}
    (models / "asl_latest.json").write_text(json.dumps(pointer), encoding="utf-8")

    model, labels, scaler = load_latest_model("asl", str(config))
    assert isinstance(model, AveragedModel) and len(model) == 2
    assert isinstance(scaler, IdentityScaler) and labels == ["MOM", "DAD", "YES"]
    x = np.random.default_rng(5).normal(size=(2, 8, 188)).astype(np.float32)

    def alone(member: Any) -> np.ndarray:
        m, sc = member
        return np.asarray(m.predict_on_batch(sc.transform(x.reshape(-1, 188)).reshape(x.shape)))

    assert np.allclose(model.predict_on_batch(x), (alone(first) + alone(second)) / 2, atol=1e-5)

    # one model named in "stem" alone still loads as before
    (models / "asl_latest.json").write_text(json.dumps({"stem": "asl_a"}), encoding="utf-8")
    single, _, single_scaler = load_latest_model("asl", str(config))
    assert not isinstance(single, AveragedModel) and not isinstance(single_scaler, IdentityScaler)


def test_ensemble_of_different_signs_is_refused(tmp_path: Path) -> None:
    from sign_bridge.data.preprocess import save_labels
    from sign_bridge.models.evaluator import load_ensemble

    _save_tiny_model(tmp_path, "asl_a", 1)
    _save_tiny_model(tmp_path, "asl_b", 2)
    save_labels(["MOM", "DAD", "NO"], tmp_path / "asl_b.labels.json")
    with pytest.raises(ValueError, match="different signs"):
        load_ensemble([tmp_path / "asl_a", tmp_path / "asl_b"])
