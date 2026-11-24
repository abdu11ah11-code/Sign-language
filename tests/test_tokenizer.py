"""Tests for tokenization and grammar adaptation."""

from __future__ import annotations

from sign_bridge.language.asl_lexicon import known_signs
from sign_bridge.language.grammar import adapt_to_gloss
from sign_bridge.language.tokenizer import Tokenizer, TokenKind


def test_phrase_matched_as_single_sign() -> None:
    tok = Tokenizer(known_signs())
    tokens = tok.tokenize("thank you")
    assert len(tokens) == 1
    assert tokens[0].kind is TokenKind.SIGN
    assert tokens[0].gloss == "THANK YOU"


def test_unknown_word_fingerspelled() -> None:
    tok = Tokenizer(known_signs())
    tokens = tok.tokenize("Zara")
    assert tokens[0].kind is TokenKind.FINGERSPELL
    assert tokens[0].letters == ("Z", "A", "R", "A")


def test_known_single_word_is_sign() -> None:
    tok = Tokenizer(known_signs())
    tokens = tok.tokenize("water")
    assert tokens[0].kind is TokenKind.SIGN
    assert tokens[0].gloss == "WATER"


def test_punctuation_becomes_pause() -> None:
    tok = Tokenizer(known_signs())
    tokens = tok.tokenize("yes.")
    assert tokens[-1].kind is TokenKind.PUNCT


def test_mixed_sentence() -> None:
    tok = Tokenizer(known_signs())
    tokens = tok.tokenize("hello Ali please")
    kinds = [t.kind for t in tokens]
    assert kinds == [TokenKind.SIGN, TokenKind.FINGERSPELL, TokenKind.SIGN]


def test_grammar_drops_function_words() -> None:
    assert adapt_to_gloss(["the", "water", "is", "good"]) == ["water", "good"]


def test_grammar_moves_time_to_front() -> None:
    assert adapt_to_gloss(["I", "eat", "today"]) == ["today", "I", "eat"]


def test_grammar_moves_wh_to_end() -> None:
    assert adapt_to_gloss(["you", "what", "want"]) == ["you", "want", "what"]
