"""Tests for the live sign→text recognizer (fake model, no TensorFlow)."""

from __future__ import annotations

import numpy as np

from sign_bridge.capture.holistic import FEATURE_DIM, HolisticResult
from sign_bridge.config import AppConfig
from sign_bridge.inference.sign_to_text import SignToTextPipeline


class _IdentityScaler:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class _FixedModel:
    """Always predicts class 0 with 95% confidence."""

    def __init__(self, n_classes: int = 3) -> None:
        self.calls = 0
        self._probs = np.full((1, n_classes), 0.05 / (n_classes - 1), dtype=np.float32)
        self._probs[0, 0] = 0.95

    def predict_on_batch(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[1:] == (30, FEATURE_DIM)
        self.calls += 1
        return self._probs


def _frame(hands: bool) -> HolisticResult:
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    feats[126:] = 0.5
    if hands:
        feats[63:126] = 0.6
    return HolisticResult(features=feats, hands_visible=hands, left_hand_visible=False,
                          right_hand_visible=hands, raw=None)


def _pipeline() -> tuple[SignToTextPipeline, _FixedModel]:
    model = _FixedModel()
    return SignToTextPipeline(model, ["HELLO", "YES", "NO"], _IdentityScaler(), AppConfig()), model


def _feed(p: SignToTextPipeline, seconds: float, fps: float, start: float = 0.0,
          drop_every: int = 0, hands: bool = True) -> tuple[list[str], float, list]:
    locks: list[str] = []
    outcomes = []
    t = start
    for i in range(int(seconds * fps)):
        t += 1.0 / fps
        visible = hands and not (drop_every and i % drop_every == 0)
        out = p.process_frame(_frame(visible), now=t)
        outcomes.append(out)
        if out.locked_sign:
            locks.append(out.locked_sign)
    return locks, t, outcomes


def test_no_prediction_until_window_is_filled() -> None:
    p, model = _pipeline()
    _feed(p, seconds=1.0, fps=30)  # < 75% of the 2 s window
    assert model.calls == 0


def test_locks_at_low_fps() -> None:
    p, _ = _pipeline()
    locks, _, _ = _feed(p, seconds=3.0, fps=12)
    assert locks == ["HELLO"]


def test_brief_hand_dropouts_do_not_block_locking() -> None:
    p, _ = _pipeline()
    locks, _, _ = _feed(p, seconds=3.0, fps=30, drop_every=4)  # 1 frame in 4 lost
    assert locks == ["HELLO"]


def test_sentence_finalized_after_pause() -> None:
    p, _ = _pipeline()
    locks, t, _ = _feed(p, seconds=3.0, fps=30)
    assert locks
    _, _, outcomes = _feed(p, seconds=2.0, fps=30, start=t, hands=False)
    finalized = [o.finalized_sentence for o in outcomes if o.finalized_sentence]
    assert finalized == ["Hello"]


def test_top_predictions_ranked() -> None:
    p, _ = _pipeline()
    _, _, outcomes = _feed(p, seconds=2.0, fps=30)
    top = next(o.top_predictions for o in reversed(outcomes) if o.top_predictions)
    assert top[0][0] == "HELLO"
    assert len(top) == 3
    assert top[0][1] >= top[1][1] >= top[2][1]
