"""American Sign Language (ASL) lexicon.

Defines the whole-word vocabulary and the fingerspelling alphabet for ASL.
Each entry carries light metadata (gloss, English label, whether it is a
movement sign) so the UI and the text→sign player can decide between a
whole-sign clip and letter-by-letter fingerspelling.

Vocabulary per project spec:
    A–Z fingerspelling (one-handed ASL manual alphabet) +
    HELLO, GOODBYE, THANK YOU, PLEASE, SORRY, YES, NO, HELP, NAME, LOVE,
    FRIEND, FAMILY, EAT, DRINK, BATHROOM, WATER, GOOD, BAD.

References (see README for full citations):
    * Start ASL — Top 25 basic signs for beginners.
    * Lifeprint / ASL University (Vicars, W.) — sign descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus


@dataclass(frozen=True)
class LexiconEntry:
    """One vocabulary item.

    Attributes:
        gloss: Canonical UPPERCASE gloss (key); spaces allowed for phrases.
        english: Human-readable English label.
        is_movement: True if the sign involves motion (affects clip length
            hints and whether a single still frame can represent it).
    """

    gloss: str
    english: str
    is_movement: bool = True


#: One-handed ASL manual alphabet.
ASL_ALPHABET: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

#: Whole-word signs available in the ASL model vocabulary.
ASL_WORDS: dict[str, LexiconEntry] = {
    e.gloss: e
    for e in (
        LexiconEntry("HELLO", "hello"),
        LexiconEntry("GOODBYE", "goodbye"),
        LexiconEntry("THANK YOU", "thank you"),
        LexiconEntry("PLEASE", "please"),
        LexiconEntry("SORRY", "sorry"),
        LexiconEntry("YES", "yes"),
        LexiconEntry("NO", "no"),
        LexiconEntry("HELP", "help"),
        LexiconEntry("NAME", "name"),
        LexiconEntry("LOVE", "love"),
        LexiconEntry("FRIEND", "friend"),
        LexiconEntry("FAMILY", "family"),
        LexiconEntry("EAT", "eat"),
        LexiconEntry("DRINK", "drink"),
        LexiconEntry("BATHROOM", "bathroom"),
        LexiconEntry("WATER", "water"),
        LexiconEntry("GOOD", "good"),
        LexiconEntry("BAD", "bad"),
    )
}

#: Manual-alphabet style: ASL is one-handed (contrast with PSL).
ALPHABET_STYLE = "one-handed"


def known_signs() -> set[str]:
    """Return the set of whole-word ASL glosses (phrases included)."""
    return set(ASL_WORDS)


def all_labels() -> list[str]:
    """All trainable class labels: alphabet letters + whole-word glosses.

    Whole-word glosses use underscores to match on-disk filenames /
    data directory names (e.g. ``THANK_YOU``).
    """
    words = [g.replace(" ", "_") for g in ASL_WORDS]
    return [*ASL_ALPHABET, *words]


# ── Google ASL Signs vocabulary (Kaggle-trained models) ──────────────────
#: Multi-word glosses in the ASL Signs vocabulary, written there as one token
#: (GLASSWINDOW) → readable words, used for display, for the words a
#: sentence is built from, and for looking the sign up.
ASL_COMPOUNDS: dict[str, str] = {
    "CALLONPHONE": "call on phone",
    "FRENCHFRIES": "french fries",
    "GLASSWINDOW": "glass window",
    "HAVETO": "have to",
    "HESHEIT": "he / she / it",
    "ICECREAM": "ice cream",
    "MINEMY": "mine / my",
    "THANKYOU": "thank you",
    "WEUS": "we / us",
}

#: Glosses kept exactly as written (acronyms).
_KEEP_CASE = {"TV"}


def readable(gloss: str) -> str | None:
    """Readable English for a compound/acronym gloss, else None."""
    g = gloss.strip().upper()
    if g in _KEEP_CASE:
        return g
    return ASL_COMPOUNDS.get(g)


def lookup_url(gloss: str) -> str:
    """A video search for how to sign gloss in ASL."""
    words = readable(gloss) or gloss.replace("_", " ").lower()
    words = words.split(" / ")[0]
    return "https://www.youtube.com/results?search_query=" + quote_plus(f"ASL sign {words}")
