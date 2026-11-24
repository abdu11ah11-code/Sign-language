"""Tests for word/sentence assembly."""

from __future__ import annotations

from sign_bridge.inference.word_builder import WordBuilder


def test_fingerspelling_builds_word() -> None:
    wb = WordBuilder(cooldown_frames=0)
    for ch in "ALI":
        wb.commit_sign(ch)
    assert wb.current_word == "ALI"


def test_whole_word_flushes_spelled_word() -> None:
    wb = WordBuilder(cooldown_frames=0)
    for ch in "ALI":
        wb.commit_sign(ch)
    wb.commit_sign("HELLO")
    assert wb.current_word == ""
    assert wb.pending_sentence == "Ali Hello"


def test_end_sentence_returns_and_archives() -> None:
    wb = WordBuilder(cooldown_frames=0)
    wb.commit_sign("HELLO")
    wb.commit_sign("FRIEND")
    sentence = wb.end_sentence()
    assert sentence == "Hello Friend"
    assert wb.sentences == ["Hello Friend"]
    assert wb.transcript == "Hello Friend"


def test_delete_removes_last_character() -> None:
    wb = WordBuilder(cooldown_frames=0)
    for ch in "CAT":
        wb.commit_sign(ch)
    wb.commit_sign("DELETE")
    assert wb.current_word == "CA"


def test_cooldown_blocks_immediate_repeat() -> None:
    wb = WordBuilder(cooldown_frames=3)
    assert wb.commit_sign("A") is True
    # Same frame region: cooldown active, different sign still blocked.
    assert wb.commit_sign("B") is False
    for _ in range(3):
        wb.tick()
    assert wb.commit_sign("B") is True


def test_same_sign_not_double_counted() -> None:
    wb = WordBuilder(cooldown_frames=0)
    assert wb.commit_sign("A") is True
    assert wb.commit_sign("A") is False  # exact repeat ignored
    wb.hands_lost()                       # boundary clears the guard
    assert wb.commit_sign("A") is True


def test_empty_sentence_returns_none() -> None:
    wb = WordBuilder(cooldown_frames=0)
    assert wb.end_sentence() is None
