"""English text → the signs the 3D person can perform.

The Speech → Sign tab used to need recorded sign *videos*, which a fresh
install does not have. The "How to sign" animation library already holds a
recorded performance of every sign the recognizer knows (250 for ASL), so
this module plans a sentence against *that* vocabulary instead: the 3D
person then signs it.

Words reach a sign in four ways, tried in order:

1. the gloss itself (``dog`` → DOG);
2. a multi-word phrase that is one sign (``thank you`` → THANKYOU);
3. an everyday synonym (``mother`` → MOM, ``television`` → TV);
4. a plain English ending taken off (``dogs`` → DOG, ``drinking`` → DRINK).

ASL word order is applied as in :mod:`sign_bridge.inference.text_to_sign`
(time words first, question words last, articles and "be" dropped). Words
with no sign are kept in the plan, marked missing, so the reader can see
what was left out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sign_bridge.language.grammar import reorder
from sign_bridge.language.tokenizer import Tokenizer, TokenKind

_CLAUSE_RE = re.compile(r"[^.?!,]+|[.?!,]")

#: Several words that are one sign in the library (spaces are matched loosely).
PHRASES = {
    "thank you": "THANKYOU", "thanks": "THANKYOU", "ice cream": "ICECREAM",
    "french fries": "FRENCHFRIES", "fries": "FRENCHFRIES", "have to": "HAVETO",
    "has to": "HAVETO", "must": "HAVETO", "phone call": "CALLONPHONE",
    "call me": "CALLONPHONE", "glass window": "GLASSWINDOW", "good night": "NIGHT",
    "good morning": "MORNING", "fire fighter": "FIREMAN", "firefighter": "FIREMAN",
}

#: Everyday words for signs whose gloss is a different word.
SYNONYMS = {
    "mother": "MOM", "mum": "MOM", "mommy": "MOM", "mummy": "MOM", "mama": "MOM",
    "father": "DAD", "daddy": "DAD", "papa": "DAD",
    "grandmother": "GRANDMA", "granny": "GRANDMA", "nana": "GRANDMA",
    "grandfather": "GRANDPA", "grandad": "GRANDPA",
    "hi": "HELLO", "hey": "HELLO", "goodbye": "BYE", "bye-bye": "BYE",
    "television": "TV", "telly": "TV", "kitten": "KITTY", "puppies": "PUPPY",
    "my": "MINEMY", "mine": "MINEMY", "we": "WEUS", "us": "WEUS",
    "he": "HESHEIT", "she": "HESHEIT", "it": "HESHEIT", "him": "HESHEIT", "her": "HESHEIT",
    "phone": "CALLONPHONE", "call": "CALLONPHONE", "telephone": "CALLONPHONE",
    "window": "GLASSWINDOW", "toilet": "POTTY", "bathroom": "POTTY",
    "fridge": "REFRIGERATOR", "eyes": "EYE", "foot": "FEET", "teeth": "TOOTH",
    "angry": "MAD", "glad": "HAPPY", "unwell": "SICK", "ill": "SICK",
    "kid": "CHILD", "kids": "CHILD", "children": "CHILD", "people": "PERSON",
    "watch": "LOOK", "looking": "LOOK", "saw": "SEE", "seen": "SEE",
    "ate": "FOOD", "eat": "FOOD", "eating": "FOOD", "said": "SAY", "talking": "TALK",
    "went": "GO", "going": "GO", "gone": "GO", "made": "MAKE", "gave": "GIVE",
    "found": "FIND", "slept": "SLEEP", "drank": "DRINK", "fell": "FALL",
    "finished": "FINISH", "done": "FINISH", "hello": "HELLO", "yes": "YES", "no": "NO",
    "not": "NOT", "don't": "NOT", "dont": "NOT", "can't": "NOT", "cant": "NOT",
    "house": "HOME", "candy": "CHOCOLATE", "trash": "GARBAGE", "rubbish": "GARBAGE",
}


@dataclass(frozen=True)
class PlannedWord:
    """One word (or phrase) of the input and the sign that shows it.

    Attributes:
        word: The input word(s), as typed or spoken.
        gloss: The sign in the library, or None when there is none.
    """

    word: str
    gloss: str | None


@dataclass
class MotionPlan:
    """A sentence planned for the 3D person."""

    words: list[PlannedWord] = field(default_factory=list)

    @property
    def glosses(self) -> list[str]:
        """The signs to perform, in signing order."""
        return [w.gloss for w in self.words if w.gloss]

    @property
    def missing(self) -> list[str]:
        """Input words with no sign in the library."""
        return [w.word for w in self.words if w.gloss is None]

    @property
    def gloss_line(self) -> str:
        """The signs in order, with missing words in brackets: ``MOM DRINK (coffee)``."""
        return " ".join(w.gloss or f"({w.word})" for w in self.words)


def _unwrap(word: str, known: set[str]) -> str | None:
    """Take plain English endings off a word until it is a known sign."""
    w = word.lower()
    if w in SYNONYMS and SYNONYMS[w] in known:
        return SYNONYMS[w]
    tries = [w]
    if w.endswith("ies") and len(w) > 4:
        tries.append(w[:-3] + "y")
    if w.endswith("ied") and len(w) > 4:
        tries.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        tries.append(w[:-2])
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        tries.append(w[:-1])
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        tries += [stem, stem + "e"]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            tries.append(stem[:-1])                  # running → run
    if w.endswith("ed") and len(w) > 4:
        stem = w[:-2]
        tries += [stem, w[:-1]]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            tries.append(stem[:-1])                  # stopped → stop
    for t in tries:
        if t.upper() in known:
            return t.upper()
        if t in SYNONYMS and SYNONYMS[t] in known:
            return SYNONYMS[t]
    return None


def plan_motion(text: str, known: set[str] | list[str], use_grammar: bool = True) -> MotionPlan:
    """Plan English text as signs from the animation library.

    Args:
        text: Typed or transcribed English.
        known: Glosses the library can perform (upper case).
        use_grammar: Apply ASL word order (time first, questions last, no "the"/"is").

    Returns:
        The words in signing order, each with its sign or None.
    """
    vocab = {g.upper() for g in known}
    lowered = f" {text.lower()} "
    for phrase, gloss in sorted(PHRASES.items(), key=lambda kv: -len(kv[0])):
        if gloss in vocab:
            lowered = re.sub(rf"(?<![a-z']){re.escape(phrase)}(?![a-z'])",
                             f" {gloss.lower()} ", lowered)
    tokenizer = Tokenizer(vocab)
    plan = MotionPlan()
    for chunk in _CLAUSE_RE.findall(lowered):
        if chunk in ".?!,":
            continue
        tokens = [t for t in tokenizer.tokenize(chunk) if t.kind is not TokenKind.PUNCT]
        if use_grammar:
            tokens = reorder(tokens, key=lambda t: t.text)
        for tok in tokens:
            if tok.kind is TokenKind.SIGN and tok.gloss:
                plan.words.append(PlannedWord(tok.text.lower(), tok.gloss.replace(" ", "")))
            else:
                plan.words.append(PlannedWord(tok.text.lower(), _unwrap(tok.text, vocab)))
    return plan
