"""Tests for the ASL Citizen name-match check, with a stand-in model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sign_bridge.capture.holistic import RAW_DIM
from sign_bridge.data import asl_citizen as ac
from sign_bridge.data import asl_citizen_check as chk

LABELS = ["DAD", "MOM", "DRINK", "WEUS"] + [f"FILLER{c}" for c in "ABCDEFGH"]
#: enough signs that "in the top 5" can fail, as with the real 250


def _row(gloss: str, sign: str, split: str = "train") -> dict[str, str]:
    return {"gloss": gloss, "sign": sign, "split": split}


def _probs(predicted: list[int]) -> np.ndarray:
    p = np.full((len(predicted), len(LABELS)), 0.01)
    p[np.arange(len(predicted)), predicted] = 0.97
    return p


@pytest.mark.parametrize(("gloss", "sign", "kind"), [
    ("MOM", "MOM", "exact"), ("CAT1", "CAT", "variant"), ("FATHER", "DAD", "alias"),
    ("ice cream", "ICECREAM", "exact")])
def test_gloss_kind(gloss: str, sign: str, kind: str) -> None:
    assert chk.gloss_kind(gloss, sign) == kind


def test_an_alias_the_model_recognises_is_kept_one_it_does_not_is_dropped() -> None:
    rows = ([_row("MOM", "MOM")] * 10 + [_row("DRINK", "DRINK")] * 10
            + [_row("FATHER", "DAD")] * 10 + [_row("US", "WEUS")] * 10)
    # exact signs: right 9/10 and 8/10. FATHER → DAD 9/10. US → predicted MOM every time.
    predicted = [1] * 9 + [0] + [2] * 8 + [0, 0] + [0] * 9 + [1] + [1] * 10
    results = chk.score(rows, _probs(predicted), LABELS)
    threshold = chk.decide(results)
    by = {r.gloss: r for r in results}
    assert by["FATHER"].kind == "alias" and by["FATHER"].keep
    assert by["FATHER"].top1 == pytest.approx(0.9)
    assert not by["US"].keep and by["US"].most_predicted == "MOM"
    assert by["MOM"].keep and by["DRINK"].keep          # exact matches always kept
    assert 0 < threshold <= 1


def test_exact_matches_are_never_dropped_even_when_weak() -> None:
    rows = [_row("MOM", "MOM")] * 6 + [_row("DRINK", "DRINK")] * 6
    results = chk.score(rows, _probs([1] * 6 + [0] * 6), LABELS)   # DRINK never right
    chk.decide(results)
    assert all(r.keep for r in results)


def test_too_few_clips_is_not_enough_to_keep() -> None:
    rows = [_row("MOM", "MOM")] * 6 + [_row("FATHER", "DAD")] * 2
    results = chk.score(rows, _probs([1] * 6 + [0] * 2), LABELS)
    chk.decide(results)
    assert not {r.gloss: r for r in results}["FATHER"].keep


def test_split_accuracy_counts_exact_matches_only() -> None:
    rows = [_row("MOM", "MOM", "test"), _row("MOM", "MOM", "test"),
            _row("FATHER", "DAD", "test")]
    acc = chk.split_accuracy(rows, _probs([1, 0, 0]), LABELS)
    assert acc["test"]["clips"] == 2 and acc["test"]["top1"] == pytest.approx(0.5)


class _Scaler:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class _Model:
    """Answers DAD for every clip."""

    def predict(self, x: np.ndarray, batch_size: int = 0, verbose: int = 0) -> np.ndarray:
        return _probs([0] * len(x))       # always DAD


def test_check_end_to_end_writes_decisions(tmp_path: Path) -> None:
    out = tmp_path / "aslc"
    out.mkdir()
    clips = np.random.default_rng(0).random((24, 8, RAW_DIM)).astype(np.float16)
    np.savez_compressed(out / "clips_0000.npz", clips=clips)
    gloss_sign = [("DAD", "DAD")] * 8 + [("MOM", "MOM")] * 8 + [("FATHER", "DAD")] * 8
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("shard,row,gloss,sign,participant,split,source\n")
        for i, (g, s) in enumerate(gloss_sign):
            split = "test" if i % 4 == 0 else "train"
            handle.write(f"clips_0000.npz,{i},{g},{s},P{i % 3},{split},v{i}.mp4\n")
    report = chk.check(out, loaded=(_Model(), LABELS, _Scaler()), meta=("holistic_lips_clip", 8))
    saved = json.loads((out / "check.json").read_text())
    assert saved == report
    assert "FATHER" in report["keep_glosses"]          # predicted DAD every time
    assert report["accuracy_exact_matches"]["test"]["clips"] == 4
    assert {r["gloss"] for r in report["glosses"]} == {"DAD", "MOM", "FATHER"}
    # decisions used train signers only
    assert all(r["clips"] == 6 for r in report["glosses"])


def test_aliases_are_real_asl_citizen_glosses_by_shape() -> None:
    """Every candidate is upper-case letters only, as normalize_gloss produces."""
    for sign, glosses in ac.ALIAS_CANDIDATES.items():
        assert sign == ac.normalize_gloss(sign)
        assert all(g == ac.normalize_gloss(g) for g in glosses)
