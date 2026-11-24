"""Tests for motion-based sign segmentation and segment-mode recognition."""

from __future__ import annotations

from typing import Any

import numpy as np

from sign_bridge.capture.holistic import RAW_DIM, HolisticResult
from sign_bridge.config import load_config
from sign_bridge.inference.segmenter import SegmenterSettings, SignSegmenter, hand_speed
from sign_bridge.inference.sign_to_text import SignToTextPipeline
from sign_bridge.inference.word_builder import WordBuilder
from sign_bridge.language.asl_lexicon import lookup_url
from sign_bridge.language.psl_lexicon import display_name

FPS = 30.0


def _frame(hand_offset: tuple[float, float] | None) -> np.ndarray:
    """Raw frame with shoulders 0.3 apart and the right hand at an offset."""
    f = np.zeros(RAW_DIM, np.float32)
    pose = f[126:162].reshape(12, 3)
    pose[4] = (0.65, 0.6, 0)   # left shoulder
    pose[5] = (0.35, 0.6, 0)   # right shoulder
    pose[0] = (0.5, 0.35, 0)   # nose
    lips = f[162:282].reshape(40, 3)
    lips[:, :2] = np.linspace(0.45, 0.55, 40)[:, None]
    if hand_offset is not None:
        hand = f[63:126].reshape(21, 3)
        base = np.stack([np.linspace(0, 0.05, 21), np.linspace(0, 0.08, 21)], 1)
        hand[:, :2] = base + np.array(hand_offset)
    return f


def _script() -> list[tuple[float, np.ndarray, bool]]:
    """0.5 s no hands, 0.8 s moving, 0.6 s still, 2.0 s no hands."""
    frames: list[tuple[np.ndarray, bool]] = []
    frames += [(_frame(None), False)] * int(0.5 * FPS)
    for i in range(int(0.8 * FPS)):
        x = 0.4 + 0.15 * np.sin(2 * np.pi * i / FPS * 1.5)
        frames.append((_frame((x, 0.4)), True))
    still = frames[-1][0]
    frames += [(still.copy(), True) for _ in range(int(0.6 * FPS))]
    frames += [(_frame(None), False)] * int(2.0 * FPS)
    return [(i / FPS, raw, vis) for i, (raw, vis) in enumerate(frames)]


def test_hand_speed_units() -> None:
    a, b = _frame((0.40, 0.4)), _frame((0.43, 0.4))  # 0.03 = 0.1 shoulder widths
    assert abs(hand_speed(a, b, 0.1) - 1.0) < 1e-4   # 0.1 widths in 0.1 s
    assert hand_speed(None, b, 0.1) > 1.0            # appearing hand = movement


def test_one_segment_per_sign() -> None:
    seg = SignSegmenter()
    segments = [s for t, raw, vis in _script() if (s := seg.update(t, raw, vis, t)) is not None]
    assert len(segments) == 1
    s = segments[0]
    assert 0.6 <= s.duration <= 1.4
    assert seg.phase == "idle"


def test_still_hands_never_start_a_sign() -> None:
    seg = SignSegmenter()
    still = _frame((0.4, 0.4))
    out = [seg.update(i / FPS, still, True, i) for i in range(1, 90)]
    # the first frame "appears" → one short segment at most, then silence
    assert sum(o is not None for o in out) <= 1
    assert seg.phase == "idle"


class _FakeModel:
    """Always answers class 1 with p=0.9 (or a flat distribution)."""

    input_shape = (None, 64, 188)

    def __init__(self, confident: bool = True) -> None:
        self.calls = 0
        self.confident = confident

    def predict_on_batch(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        assert x.shape == (1, 64, 188)
        p = np.full(4, 0.1 / 3) if self.confident else np.full(4, 0.25)
        if self.confident:
            p[1] = 0.9
        return p[None, :]


class _Identity:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


def _run(model: Any, vocabulary: list[str] | None = None,
         live: bool = False) -> tuple[list[Any], SignToTextPipeline]:
    pipe = SignToTextPipeline(model, ["HELLO", "THANKYOU", "YES", "NO"], _Identity(),
                              load_config(), feature_mode="holistic_lips", vocabulary=vocabulary)
    if not live:
        pipe.LIVE_EVERY = float("inf")       # count only the guesses made when a sign ends
    outs = []
    for t, raw, vis in _script():
        outs.append(pipe.process_frame(
            HolisticResult(features=raw, hands_visible=vis, left_hand_visible=False,
                           right_hand_visible=vis, raw=None), now=t))
    return outs, pipe


def test_segment_mode_predicts_once_per_sign() -> None:
    model = _FakeModel()
    outs, pipe = _run(model)
    assert pipe.mode == "segment"
    assert model.calls == 1                       # one guess, not one per frame
    locked = [o.locked_sign for o in outs if o.is_locked]
    assert locked == ["THANKYOU"]
    assert any(o.phase == "signing" for o in outs)
    assert any(o.finalized_sentence == "Thank You" for o in outs)


def test_guess_while_the_sign_is_still_moving() -> None:
    """While signing, the movement so far is guessed and shown, but never written down."""
    model = _FakeModel()
    outs, _ = _run(model, live=True)
    live = [o for o in outs if o.live_sign]
    assert live and all(o.phase == "signing" for o in live)
    assert all(o.live_sign == "THANKYOU" and o.live_confidence > 0.5 for o in live)
    assert [o.locked_sign for o in outs if o.is_locked] == ["THANKYOU"]   # still one word
    assert model.calls > 1                                  # the extra guesses ran
    done = outs[-1]
    assert done.phase == "idle" and done.live_sign is None


def test_unsure_when_not_confident() -> None:
    outs, _ = _run(_FakeModel(confident=False))
    assert not any(o.is_locked for o in outs)
    assert any(o.unsure and o.top_predictions for o in outs)


def test_vocabulary_restricts_choices() -> None:
    outs, pipe = _run(_FakeModel(confident=False), vocabulary=["yes"])
    assert pipe.vocabulary == ["YES"]
    assert [o.locked_sign for o in outs if o.is_locked] == ["YES"]


def test_segment_retimed_to_fixed_fps() -> None:
    """The same 1-second sign tracked at 10 or 30 fps gives the same input length."""
    _, pipe = _run(_FakeModel())
    from sign_bridge.inference.segmenter import Segment

    def seg(fps: float) -> Segment:
        n = int(fps)
        frames = [(i / fps, _frame((0.3 + 0.2 * i / n, 0.4))) for i in range(n + 1)]
        return Segment(frames=frames, start=0.0, end=1.0)

    a, b = pipe.segment_input(seg(10)), pipe.segment_input(seg(30))
    assert a.shape == b.shape == (64, 188)
    assert np.allclose(a[:31], b[:31], atol=0.08)


def test_word_builder_allows_repeats_for_segments() -> None:
    b = WordBuilder()
    assert b.commit_sign("YES", allow_repeat=True)
    assert b.commit_sign("YES", allow_repeat=True)
    assert b.end_sentence() == "Yes Yes"


def test_asl_names_and_links() -> None:
    assert display_name("GLASSWINDOW") == "Glass Window"
    assert display_name("TV") == "TV"
    assert "glass+window" in lookup_url("GLASSWINDOW")
    assert lookup_url("HELLO").startswith("https://www.youtube.com/results?search_query=")


def test_tuning_notebook_builds(tmp_path: Any) -> None:
    """The segmenter-tuning notebook generator produces valid, compilable cells."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    import build_tuning_notebook as builder

    data = json.loads(builder.build(tmp_path / "nb.ipynb").read_text(encoding="utf-8"))
    assert data["nbformat"] == 4 and len(data["cells"]) >= 7
    for i, src in enumerate(builder.code_cells()):
        compile(src, f"<cell {i}>", "exec")
    joined = "\n".join(builder.code_cells())
    assert "segmenter_tuning.csv" in joined and "segmenter_full_pipeline.csv" in joined


def test_precomputed_speed_matches_measured() -> None:
    """Passing a measured speed in gives exactly the same segments as measuring inline."""
    script = _script()
    live = SignSegmenter()
    live_out = [live.update(t, raw, vis, i) for i, (t, raw, vis) in enumerate(script)]

    # measure every frame's raw speed once, the way an offline sweep would
    speeds, prev_raw, prev_t = [], None, None
    for t, raw, vis in script:
        dt = 0.0 if prev_t is None else t - prev_t
        speeds.append(hand_speed(prev_raw, raw, dt) if (vis and dt > 0) else 0.0)
        prev_raw, prev_t = (raw if vis else None), t

    offline = SignSegmenter()
    offline_out = [offline.update(t, raw, vis, i, speed=speeds[i])
                   for i, (t, raw, vis) in enumerate(script)]

    assert [o is not None for o in live_out] == [o is not None for o in offline_out]
    pairs = [(a, b) for a, b in zip(live_out, offline_out, strict=True) if a is not None]
    assert len(pairs) == 1
    assert pairs[0][0].frames == pairs[0][1].frames


def _moving(n: int, start: float = 0.0) -> list[tuple[float, np.ndarray, bool]]:
    """n frames of a moving right hand, starting at time ``start``."""
    return [(start + i / FPS,
             _frame((0.4 + 0.15 * np.sin(2 * np.pi * i / FPS * 1.5), 0.4)), True)
            for i in range(n)]


def _still(n: int, start: float) -> list[tuple[float, np.ndarray, bool]]:
    frame = _frame((0.4, 0.4))
    return [(start + i / FPS, frame.copy(), True) for i in range(n)]


def _paused_sign() -> list[tuple[float, np.ndarray, bool]]:
    """One sign with a 0.2 s pause in the middle, then hands away."""
    out = [(i / FPS, _frame(None), False) for i in range(int(0.5 * FPS))]
    t = len(out) / FPS
    out += _moving(int(0.5 * FPS), t)
    t = out[-1][0] + 1 / FPS
    out += _still(int(0.5 * FPS), t)   # the signer pauses mid-sign
    t = out[-1][0] + 1 / FPS
    out += _moving(int(0.5 * FPS), t)
    t = out[-1][0] + 1 / FPS
    out += [(t + i / FPS, _frame(None), False) for i in range(int(2.0 * FPS))]
    return out


def test_pause_mid_sign_does_not_split_it() -> None:
    """A short pause inside a sign is merged, not reported as two signs."""
    merged = SignSegmenter(SegmenterSettings(hold_seconds=0.15, merge_gap_seconds=0.4))
    got = [s for t, raw, vis in _paused_sign() if (s := merged.update(t, raw, vis, t))]
    assert len(got) == 1
    assert got[0].duration > 1.0                      # both halves, pause included

    split = SignSegmenter(SegmenterSettings(hold_seconds=0.15, merge_gap_seconds=0.0))
    got2 = [s for t, raw, vis in _paused_sign() if (s := split.update(t, raw, vis, t))]
    assert len(got2) == 2                             # merging off → two signs


def test_long_pause_still_gives_two_signs() -> None:
    """Two signs with a real gap between them stay two signs."""
    seg = SignSegmenter(SegmenterSettings(hold_seconds=0.15, merge_gap_seconds=0.3))
    script = [(i / FPS, _frame(None), False) for i in range(int(0.3 * FPS))]
    t = len(script) / FPS
    script += _moving(int(0.5 * FPS), t)
    t = script[-1][0] + 1 / FPS
    script += _still(int(1.2 * FPS), t)   # long pause
    t = script[-1][0] + 1 / FPS
    script += _moving(int(0.5 * FPS), t)
    t = script[-1][0] + 1 / FPS
    script += [(t + i / FPS, _frame(None), False) for i in range(int(2.0 * FPS))]
    got = [s for tt, raw, vis in script if (s := seg.update(tt, raw, vis, tt))]
    assert len(got) == 2


def test_hands_leaving_reports_the_sign_at_once() -> None:
    """Dropping the hands ends a sign immediately — no merge delay."""
    seg = SignSegmenter(SegmenterSettings(merge_gap_seconds=0.5))
    got = [s for t, raw, vis in _script() if (s := seg.update(t, raw, vis, t))]
    assert len(got) == 1


def _two_bursts(gap_seconds: float) -> list[tuple[float, np.ndarray, bool]]:
    """Two separate signing bursts with ``gap_seconds`` of stillness between."""
    out = [(i / FPS, _frame(None), False) for i in range(int(0.3 * FPS))]
    t = len(out) / FPS
    out += _moving(int(0.5 * FPS), t)
    t = out[-1][0] + 1 / FPS
    out += _still(int(gap_seconds * FPS), t)
    t = out[-1][0] + 1 / FPS
    out += _moving(int(0.5 * FPS), t)
    t = out[-1][0] + 1 / FPS
    out += [(t + i / FPS, _frame(None), False) for i in range(int(3.0 * FPS))]
    return out


def _words(script: list[tuple[float, np.ndarray, bool]], model: Any) -> list[str]:
    """Words the pipeline writes down for a script."""
    pipe = SignToTextPipeline(model, ["HELLO", "THANKYOU", "YES", "NO"], _Identity(),
                              load_config(), feature_mode="holistic_lips")
    pipe.LIVE_EVERY = float("inf")           # count only the guesses made when a sign ends
    locked = []
    for t, raw, vis in script:
        out = pipe.process_frame(
            HolisticResult(features=raw, hands_visible=vis, left_hand_visible=False,
                           right_hand_visible=vis, raw=None), now=t)
        if out.is_locked:
            locked.append(out.locked_sign)
    return locked


def test_one_sign_split_in_two_is_written_once() -> None:
    """The same sign seen twice after a short pause counts as one word."""
    model = _FakeModel()
    assert _words(_two_bursts(1.2), model) == ["THANKYOU"]      # short pause → one word
    assert model.calls == 2                                      # (both halves were classified)


def test_a_genuinely_repeated_sign_is_kept() -> None:
    """After a real pause, signing the same word again counts twice."""
    assert _words(_two_bursts(2.5), _FakeModel()) == ["THANKYOU", "THANKYOU"]


def test_live_settings_are_the_measured_ones() -> None:
    """Chosen on ASL Citizen val signers, checked on its test signers (see config.yaml)."""
    inf = load_config().inference
    assert (inf.segment_confidence, inf.segment_margin, inf.hold_seconds) == (0.35, 0.05, 0.4)


def test_readout_shows_the_live_guess_faded() -> None:
    from sign_bridge.ui.components.readout import segment_sign_html

    html_ = segment_sign_html(None, False, "signing", True, "asl", live="MOM", live_confidence=0.62)
    assert "Mom…" in html_ and "62%" in html_ and "fg-sign idle" in html_
    assert "watching your sign" in segment_sign_html(None, False, "signing", True, "asl")
    done = segment_sign_html("MOM", False, "idle", True, "asl", live=None)
    assert "fg-sign locked" in done
