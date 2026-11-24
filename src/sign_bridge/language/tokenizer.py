"""Tokenize English text into signable units.

Turns a free-text string into an ordered list of :class:`SignToken`, each
either a whole-sign gloss (found in the lexicon) or a fingerspelled word
(spelled letter by letter when no whole sign exists). Multi-word phrases
such as "thank you" are matched greedily as a single sign before falling
back to single words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TokenKind(Enum):
    """How a token should be rendered to sign video."""

    SIGN = "sign"            # a whole-word sign exists
    FINGERSPELL = "spell"    # render letter by letter
    PUNCT = "punct"          # sentence boundary / pause (no video)


@dataclass(frozen=True)
class SignToken:
    """One signable unit.

    Attributes:
        kind: SIGN, FINGERSPELL, or PUNCT.
        text: Display text (e.g. "THANK YOU", "ALI", ".").
        gloss: Lexicon gloss for SIGN tokens; None otherwise.
        letters: Uppercase letters for FINGERSPELL tokens; () otherwise.
    """

    kind: TokenKind
    text: str
    gloss: str | None = None
    letters: tuple[str, ...] = ()


_WORD_RE = re.compile(r"[A-Za-z']+|[.?!,]")
_MAX_PHRASE_WORDS = 3  # longest multi-word phrase to test against the lexicon


class Tokenizer:
    """Greedy phrase-aware tokenizer over a known sign vocabulary.

    Args:
        known_signs: Set of glosses available as whole signs. Case and
            ``_``/space are interchangeable (``THANK_YOU`` == ``thank you``).
    """

    def __init__(self, known_signs: set[str]) -> None:
        # On-disk glosses use underscores (THANK_YOU.mp4); text uses spaces.
        self._known = {s.upper().replace("_", " ") for s in known_signs}

    def tokenize(self, text: str) -> list[SignToken]:
        """Convert text to an ordered list of signable tokens.

        Args:
            text: Free-form English input.

        Returns:
            Ordered tokens; phrases collapse to one SIGN, unknown words
            become FINGERSPELL, punctuation becomes PUNCT pauses.
        """
        words = _WORD_RE.findall(text)
        tokens: list[SignToken] = []
        i = 0
        while i < len(words):
            w = words[i]
            if w in ".?!,":
                tokens.append(SignToken(TokenKind.PUNCT, w))
                i += 1
                continue

            matched = self._match_phrase(words, i)
            if matched is not None:
                phrase, span = matched
                tokens.append(SignToken(TokenKind.SIGN, phrase.title(), gloss=phrase))
                i += span
                continue

            upper = w.upper()
            if upper in self._known:
                tokens.append(SignToken(TokenKind.SIGN, w.title(), gloss=upper))
            else:
                letters = tuple(ch for ch in upper if ch.isalpha())
                tokens.append(SignToken(TokenKind.FINGERSPELL, w.upper(), letters=letters))
            i += 1
        return tokens

    def _match_phrase(self, words: list[str], start: int) -> tuple[str, int] | None:
        """Try the longest multi-word lexicon phrase starting at ``start``."""
        upper = max_span = None
        for span in range(min(_MAX_PHRASE_WORDS, len(words) - start), 1, -1):
            chunk = " ".join(words[start : start + span]).upper()
            if chunk in self._known:
                upper, max_span = chunk, span
                break
        return (upper, max_span) if upper is not None and max_span is not None else None
