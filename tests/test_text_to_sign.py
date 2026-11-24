"""Tests for the text→sign planner and video stitching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sign_bridge.data.video_library import VideoLibrary
from sign_bridge.inference.text_to_sign import TextToSignPipeline
from sign_bridge.language.grammar import reorder
from sign_bridge.language.tokenizer import Tokenizer, TokenKind


def _write_clip(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (320, 240)
) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, size)
    for _ in range(12):
        writer.write(np.full((size[1], size[0], 3), color, dtype=np.uint8))
    writer.release()


@pytest.fixture
def library(tmp_path: Path) -> VideoLibrary:
    root = tmp_path / "sign_videos"
    _write_clip(root / "asl" / "HELLO.mp4", (0, 0, 120))
    _write_clip(root / "asl" / "THANK_YOU.mp4", (0, 120, 0))
    _write_clip(root / "asl" / "GOOD_MORNING.mp4", (120, 0, 0))
    for ch in "ALI":
        _write_clip(root / "asl" / "letters" / f"{ch}.mp4", (90, 90, 90))
    return VideoLibrary(root, "asl")


def test_tokenizer_matches_underscore_glosses() -> None:
    tokens = Tokenizer({"THANK_YOU"}).tokenize("thank you")
    assert [t.kind for t in tokens] == [TokenKind.SIGN]


def test_phrase_clip_used_not_fingerspelled(library: VideoLibrary) -> None:
    seq = TextToSignPipeline(library).plan("Thank you")
    assert [c.caption for c in seq.clips] == ["Thank You"]
    assert not seq.missing


def test_grammar_keeps_phrase_intact(library: VideoLibrary) -> None:
    seq = TextToSignPipeline(library).plan("Good morning")
    assert seq.gloss == "GOOD MORNING"
    assert seq.clips[0].caption == "Good Morning"


def test_clause_punctuation_becomes_pause(library: VideoLibrary) -> None:
    seq = TextToSignPipeline(library).plan("Hello. Ali!")
    kinds = ["pause" if c.is_pause else "clip" for c in seq.clips]
    assert kinds == ["clip", "pause", "clip", "clip", "clip"]  # no trailing pause


def test_reorder_generic_key() -> None:
    items = [("the", 1), ("water", 2), ("today", 3)]
    assert reorder(items, key=lambda it: it[0]) == [("today", 3), ("water", 2)]


def test_render_with_captions_needs_no_imagemagick(library: VideoLibrary, tmp_path: Path) -> None:
    pytest.importorskip("moviepy")
    pipeline = TextToSignPipeline(library)
    out = pipeline.render(pipeline.plan("Hello Ali"), tmp_path / "out.mp4", add_captions=True)
    assert out.is_file() and out.stat().st_size > 0
