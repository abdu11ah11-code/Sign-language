"""Lightweight English→sign-gloss grammar adaptation.

Signed languages are NOT word-for-word English. ASL and PSL both tend to
use topic–comment ordering, place time markers first, and omit English
function words (articles, the copula "to be"). This module applies a small,
transparent set of rules to reorder/clean an English token stream before it
is sent to the video sequencer, producing more natural sign output.

This is deliberately simple and rule-based — a documented limitation. Full
sign-language MT is an open research problem; the goal here is "better than
raw English word order", not linguistic perfection. Real deployments should
involve Deaf signers in evaluating output (see README §Ethics).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

# English function words that are typically dropped in ASL/PSL glossing.
_DROP = {"a", "an", "the", "is", "are", "am", "be", "to", "of", "do", "does"}

# Time words conventionally move to the front of the clause.
_TIME = {
    "today", "tomorrow", "yesterday", "now", "later", "tonight",
    "morning", "evening", "night", "week", "month", "year",
}

# WH-words conventionally move to the end in ASL question structure.
_WH = {"what", "where", "who", "when", "why", "which", "how"}


def reorder(items: Sequence[T], key: Callable[[T], str]) -> list[T]:
    """Apply the gloss rules to any token type.

    ``key`` returns the English text of an item. Multi-word sign phrases
    (e.g. a single "good morning" token) never match a rule, so phrases that
    already have their own sign are kept intact and in place.
    """
    kept = [it for it in items if key(it).lower() not in _DROP]

    time_items = [it for it in kept if key(it).lower() in _TIME]
    rest = [it for it in kept if key(it).lower() not in _TIME]

    wh_items = [it for it in rest if key(it).lower() in _WH]
    non_wh = [it for it in rest if key(it).lower() not in _WH]

    return [*time_items, *non_wh, *wh_items]


def adapt_to_gloss(words: list[str]) -> list[str]:
    """Reorder/clean an English word list toward sign-gloss order.

    Rules, applied in order:
        1. Drop English function words that have no sign equivalent.
        2. Move time words to the front (ASL/PSL time-topic ordering).
        3. Move a WH-word to the end (common question structure).

    Args:
        words: English words (no punctuation).

    Returns:
        A new, reordered list of glossable words. Casing is preserved.
    """
    return reorder(words, key=lambda w: w)
