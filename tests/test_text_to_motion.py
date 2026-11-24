"""Tests for planning sentences for the 3D person and joining signs into one clip."""

from __future__ import annotations

import io

import numpy as np

from sign_bridge.inference.text_to_motion import plan_motion

VOCAB = {"HELLO", "MOM", "DAD", "DRINK", "WATER", "DOG", "SLEEP", "THANKYOU", "WHERE", "CAT",
         "NOW", "TOMORROW", "MINEMY", "ICECREAM", "CRY", "GO", "HUNGRY", "NOT"}


def test_words_reach_their_signs() -> None:
    plan = plan_motion("Hello, my mother is drinking water.", VOCAB)
    assert plan.glosses == ["HELLO", "MINEMY", "MOM", "DRINK", "WATER"]
    assert plan.missing == []                              # "is" is dropped, as in ASL


def test_phrases_plurals_and_endings() -> None:
    assert plan_motion("Thank you", VOCAB).glosses == ["THANKYOU"]
    assert plan_motion("ice cream", VOCAB).glosses == ["ICECREAM"]
    assert plan_motion("dogs", VOCAB).glosses == ["DOG"]
    assert plan_motion("sleeping", VOCAB).glosses == ["SLEEP"]
    assert plan_motion("cried", VOCAB).glosses == ["CRY"]
    assert plan_motion("father went", VOCAB).glosses == ["DAD", "GO"]
    assert plan_motion("don't", VOCAB).glosses == ["NOT"]


def test_asl_word_order() -> None:
    # time words first, question words last
    assert plan_motion("my dog is sleeping now", VOCAB).glosses == ["NOW", "MINEMY", "DOG", "SLEEP"]
    assert plan_motion("where is the cat?", VOCAB).glosses == ["CAT", "WHERE"]


def test_words_without_a_sign_are_kept_and_marked() -> None:
    plan = plan_motion("hello I am a human", VOCAB)
    assert plan.glosses == ["HELLO"]
    assert plan.missing == ["i", "human"]
    assert plan.gloss_line == "HELLO (i) (human)"


def test_phrases_only_when_the_sign_exists() -> None:
    assert plan_motion("thank you", {"HELLO"}).glosses == []


def _clip(gloss: str, n: int, value: float) -> object:
    from sign_bridge.ui.animation import LAYOUT_V2, POINTS_V2, SignClip

    frames = [[value + i] * (POINTS_V2 * 3) for i in range(n)]
    return SignClip(gloss, frames, 12.0, LAYOUT_V2)


def test_join_clips_blends_and_rests() -> None:
    from sign_bridge.ui.figure3d import join_clips

    a, b = _clip("MOM", 5, 10.0), _clip("DAD", 6, 50.0)
    joined, segs = join_clips([a, b], ["MOM", "DAD"], blend=3, rest=4)  # type: ignore[list-item]
    assert [(s.word, s.start, s.end) for s in segs] == [("MOM", 0, 4), ("DAD", 8, 13)]
    assert len(joined.frames) == 5 + 3 + 6 + 4 and joined.has_depth
    mids = [f[0] for f in joined.frames[5:8]]                  # 14 → 50, in between
    assert all(14.0 < m < 50.0 for m in mids) and mids == sorted(mids)
    assert joined.frames[-1] == joined.frames[13]              # holds the last pose


def test_sentence_page_has_words_and_touches_per_sign() -> None:
    import json
    import re

    from sign_bridge.ui.figure3d import join_clips, sign_3d_html
    from tests.test_figure3d import _touching_clip

    touch, plain = _touching_clip(True), _touching_clip(False)
    joined, segs = join_clips([plain, touch], ["YES", "MOM"])
    page = sign_3d_html(joined, segments=segs)
    data = json.loads(re.search(r"const D = (\{.*?\});\n", page).group(1))  # type: ignore[union-attr]
    assert [w["word"] for w in data["words"]] == ["YES", "MOM"]
    assert data["touch"] is None and len(data["touches"]) == 1
    t = data["touches"][0]
    assert (t["start"], t["end"]) == (segs[1].start, segs[1].end)
    assert len(t["weight"]) == len(joined.frames)
    assert max(t["weight"][: segs[1].start]) == 0.0              # nothing before MOM
    assert 'id="word"' in page


def test_browser_recording_becomes_whisper_audio() -> None:
    import soundfile as sf

    from sign_bridge.ui.components.sign_player import wav_to_whisper

    rate = 48_000
    t = np.arange(rate) / rate                                   # one second, stereo
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], 1) * 0.5
    buf = io.BytesIO()
    sf.write(buf, stereo, rate, format="WAV")
    audio = wav_to_whisper(buf.getvalue())
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) - 16_000) <= 2 and 0.4 < np.abs(audio).max() < 0.6


def test_two_frame_motion_and_old_models() -> None:
    from sign_bridge.models.conv_transformer import TemporalDelta

    x = np.random.default_rng(0).random((2, 6, 3)).astype("float32")
    y = TemporalDelta((1, 2))(x).numpy()
    assert y.shape == (2, 6, 9)
    assert np.allclose(y[:, 4, 3:6], x[:, 4] - x[:, 3])
    assert np.allclose(y[:, 4, 6:], x[:, 4] - x[:, 2])
    assert np.allclose(y[:, 1, 6:], x[:, 1] - x[:, 0])          # first frames repeat frame 0
    old = TemporalDelta.from_config({"name": "motion"})          # config saved before lags existed
    assert old.lags == (1,) and old(x).shape == (2, 6, 6)
