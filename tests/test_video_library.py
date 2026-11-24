"""Tests for the video library and config validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from sign_bridge.data.video_library import VideoLibrary


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")  # placeholder clip


def test_library_indexes_signs_and_letters(tmp_path: Path) -> None:
    root = tmp_path / "sign_videos"
    _touch(root / "asl" / "HELLO.mp4")
    _touch(root / "asl" / "THANK_YOU.mp4")
    _touch(root / "asl" / "letters" / "A.mp4")
    lib = VideoLibrary(root, "asl")

    assert lib.has("HELLO")
    assert lib.has("thank you")            # case/space-insensitive
    assert lib.path_for_letter("A") is not None
    assert "HELLO" in lib.available_signs()
    assert lib.available_letters() == ["A"]


def test_library_fingerspell_check(tmp_path: Path) -> None:
    root = tmp_path / "sign_videos"
    for ch in "CAT":
        _touch(root / "asl" / "letters" / f"{ch}.mp4")
    lib = VideoLibrary(root, "asl")
    assert lib.can_fingerspell("cat")
    assert not lib.can_fingerspell("dog")  # D, O, G missing


def test_library_missing_dir_is_empty(tmp_path: Path) -> None:
    lib = VideoLibrary(tmp_path / "nope", "psl")
    assert lib.available_signs() == []
    assert not lib.has("HELLO")


def test_unknown_letter_returns_none(tmp_path: Path) -> None:
    lib = VideoLibrary(tmp_path, "asl")
    assert lib.path_for_letter("1") is None
    assert lib.path_for_letter("AB") is None


# ── config validation ────────────────────────────────────────────────────
def test_config_defaults_load() -> None:
    from sign_bridge.config import AppConfig

    cfg = AppConfig()
    assert cfg.landmarks.feature_dim == 162
    assert cfg.inference.confidence_threshold == 0.80


def test_config_rejects_out_of_range() -> None:
    from pydantic import ValidationError

    from sign_bridge.config import InferenceSection

    with pytest.raises(ValidationError):
        InferenceSection(confidence_threshold=1.5)  # > 1.0


def test_config_rejects_unknown_key() -> None:
    from pydantic import ValidationError

    from sign_bridge.config import AppSection

    with pytest.raises(ValidationError):
        AppSection(unknown_field="x")  # extra=forbid
