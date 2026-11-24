"""Pakistan Sign Language (PSL) lexicon — the project's novel contribution.

PSL is severely under-resourced in computational tooling. This lexicon
documents a starter vocabulary of common signs drawn from the Deaf Reach /
FESF Pakistan Sign Language resources, each annotated with an Urdu gloss
(transliterated) alongside its English meaning.

Important linguistic note — alphabet handedness:
    PSL predominantly uses a TWO-HANDED manual alphabet, whereas ASL uses a
    one-handed alphabet. (FESF/PSL documentation also notes that one-handed
    fingerspelling is increasingly recognized nationwide, and Urdu
    fingerspelling is not in common use.) Because two-handed fingerspelling
    occupies both hands, the 162-dim feature (both hands + pose) is
    well-suited to it — but mirror augmentation must be used cautiously for
    handedness-sensitive PSL letters (see data/augment.py and README).

Dialect note:
    PSL has regional dialects (Karachi, Lahore, etc.). FESF documents the
    "most commonly used" sign per word; this lexicon follows that choice and
    flags that deployed systems should validate signs with local Deaf
    communities.

References (full citations in README):
    * Family Educational Services Foundation (FESF), Pakistan Sign Language
      Dictionary (7,000+ words), https://psl.org.pk/
    * FESF / Deaf Reach, "Signs to Learn" and "1000 Basic Signs", 2017.
    * National Institute of Special Education, PSL Book I & II, Govt. of
      Pakistan, 1991.
    * Khan, N. S. et al., "Speak Pakistan: Challenges in developing Pakistan
      Sign Language using information technology," South Asian Studies 30(2),
      2014.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PslEntry:
    """One PSL vocabulary item.

    Attributes:
        gloss: Canonical UPPERCASE English gloss (key).
        english: English label.
        urdu: Transliterated Urdu word.
        two_handed: True if the sign uses both hands (most PSL signs do).
    """

    gloss: str
    english: str
    urdu: str
    two_handed: bool = True


#: PSL uses a two-handed manual alphabet (key difference from ASL).
ALPHABET_STYLE = "two-handed"
PSL_ALPHABET: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

#: Common PSL signs (English gloss → entry). Urdu transliterations provided
#: for the bilingual UI; validate against psl.org.pk for production use.
PSL_WORDS: dict[str, PslEntry] = {
    e.gloss: e
    for e in (
        PslEntry("HELLO", "hello", "salaam"),
        PslEntry("GOODBYE", "goodbye", "khuda hafiz"),
        PslEntry("THANK YOU", "thank you", "shukriya"),
        PslEntry("PLEASE", "please", "meherbani"),
        PslEntry("SORRY", "sorry", "maaf kijiye"),
        PslEntry("YES", "yes", "haan"),
        PslEntry("NO", "no", "nahi"),
        PslEntry("HELP", "help", "madad"),
        PslEntry("NAME", "name", "naam"),
        PslEntry("WATER", "water", "paani"),
        PslEntry("EAT", "eat", "khana"),
        PslEntry("DRINK", "drink", "peena"),
        PslEntry("MOTHER", "mother", "ammi"),
        PslEntry("FATHER", "father", "abbu"),
        PslEntry("FRIEND", "friend", "dost"),
        PslEntry("GOOD", "good", "acha"),
        PslEntry("BAD", "bad", "bura"),
        PslEntry("SCHOOL", "school", "school"),
        PslEntry("HOME", "home", "ghar"),
        PslEntry("LOVE", "love", "mohabbat"),
    )
}


def known_signs() -> set[str]:
    """Return the set of whole-word PSL glosses (phrases included)."""
    return set(PSL_WORDS)


def urdu_for(gloss: str) -> str | None:
    """Return the Urdu transliteration for a gloss, if known."""
    entry = PSL_WORDS.get(gloss.upper())
    return entry.urdu if entry else None


def all_labels() -> list[str]:
    """All trainable class labels: alphabet letters + whole-word glosses."""
    words = [g.replace(" ", "_") for g in PSL_WORDS]
    return [*PSL_ALPHABET, *words]


# ── Urdu alphabet (UAlpha40 dataset) ─────────────────────────────────────
#: Gloss → Urdu letter for the 40 signs of the UAlpha40 PSL alphabet
#: dataset. Glosses are the dataset's folder names as ``sign-import`` writes
#: them (upper case, ``-`` → ``_``). The letter for each name was checked
#: against the alphabet position in the dataset's file names
#: (e.g. ``s0251-28kaf`` → 28th letter → ق, ``s0251-40byeh`` → 40th → ے).
URDU_LETTERS: dict[str, str] = {
    "ALIFMAD": "آ", "ALIF": "ا", "BAY": "ب", "PAY": "پ", "TAY": "ت",
    "TAAY": "ٹ", "SAY": "ث", "JEEM": "ج", "CHAY": "چ", "1_HAY": "ح",
    "KHAY": "خ", "DAL": "د", "DAAL": "ڈ", "ZAAL": "ذ", "RAY": "ر",
    "ARAY": "ڑ", "ZAY": "ز", "ZAEY": "ژ", "SEEN": "س", "SHEEN": "ش",
    "SUAD": "ص", "ZUAD": "ض", "TUEY": "ط", "ZUEY": "ظ", "AIN": "ع",
    "GHAIN": "غ", "FAY": "ف", "KAF": "ق", "KIAF": "ک", "GAAF": "گ",
    "LAM": "ل", "MEEM": "م", "NUUN": "ن", "NUUNGH": "ں", "WAO": "و",
    "2_HAY": "ہ", "DOCHAHAY": "ھ", "HAMZA": "ء", "CYEH": "ی", "BYEH": "ے",
}


def display_name(gloss: str) -> str:
    """Human-readable label, with the Urdu letter for alphabet signs.

    ``"KAF"`` → ``"Kaf · ق"``, ``"1_HAY"`` → ``"1 Hay · ح"``,
    ``"THANK_YOU"`` → ``"Thank You"``.
    """
    from sign_bridge.language.asl_lexicon import readable

    asl = readable(gloss)
    if asl is not None:
        return asl if asl.isupper() else asl.title()
    name = gloss.replace("_", " ").title()
    letter = URDU_LETTERS.get(gloss.upper())
    return f"{name} · {letter}" if letter else name
