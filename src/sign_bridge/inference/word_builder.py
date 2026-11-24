"""Word and sentence assembly from locked signs.

Bridges discrete locked signs into readable text. Two sign kinds:

    * letter signs (single A-Z): appended to the current word buffer;
    * whole-word signs (HELLO, THANK_YOU): flushed as their own token.

Special control glosses, if present in a model's vocabulary, are honored:
    SPACE   → finalize the current word
    DELETE  → remove the last character / token
    NOTHING → ignored (idle / transition class)

A sentence is finalized when the hands disappear for longer than the
configured pause (handled by the caller, which calls :meth:`end_sentence`).
A cooldown prevents the same physical hold from registering twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sign_bridge.language.asl_lexicon import readable

_CONTROL = {"SPACE", "DELETE", "DEL", "NOTHING", "BLANK"}


@dataclass
class BuilderState:
    """Mutable assembly state.

    Attributes:
        current_word: Letters accumulated since the last word break.
        tokens: Finalized words/glosses in the sentence so far.
        sentences: Completed sentences this session.
    """

    current_word: str = ""
    tokens: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)


class WordBuilder:
    """Assembles locked signs into words, sentences, and a transcript.

    Args:
        cooldown_frames: Frames to ignore new locks after accepting one,
            so a single sustained hold is not counted repeatedly.
        letter_map: Extra fingerspelling letters: gloss → character, e.g.
            ``{"ALIF": "ا"}`` so PSL Urdu-alphabet signs spell Urdu words.
    """

    def __init__(self, cooldown_frames: int = 15, letter_map: dict[str, str] | None = None) -> None:
        self._state = BuilderState()
        self._letters = {k.upper(): v for k, v in (letter_map or {}).items()}
        self._cooldown_frames = cooldown_frames
        self._cooldown = 0
        self._last_committed: str | None = None
        #: Words of the most recently finished sentence (for the AI writer).
        self.last_tokens: list[str] = []

    # ── frame tick ───────────────────────────────────────────────────────
    def tick(self) -> None:
        """Advance one frame (decrement the cooldown). Call every frame."""
        if self._cooldown > 0:
            self._cooldown -= 1

    def commit_sign(self, gloss: str, allow_repeat: bool = False) -> bool:
        """Accept a freshly locked sign.

        Args:
            gloss: The locked sign label.
            allow_repeat: Skip the cooldown and the no-immediate-repeat rule.
                Used when each call is already one separately performed sign
                (motion-segmented recognition), so ``YES YES`` stays two words.

        Returns:
            True if the sign changed the buffer; False if ignored (cooldown,
            idle class, or exact repeat of the just-committed sign).
        """
        if self._cooldown > 0 and not allow_repeat:
            return False
        upper = gloss.strip().upper()
        if upper in {"NOTHING", "BLANK"}:
            return False
        if upper == self._last_committed and not allow_repeat:
            return False

        self._apply(upper)
        self._last_committed = upper
        self._cooldown = self._cooldown_frames
        return True

    def _apply(self, gloss: str) -> None:
        if gloss in {"SPACE"}:
            self._flush_word()
        elif gloss in {"DELETE", "DEL"}:
            self._backspace()
        elif gloss in self._letters:
            self._state.current_word += self._letters[gloss]
        elif len(gloss) == 1 and gloss.isalpha():
            self._state.current_word += gloss
        else:
            # Whole-word sign: flush any in-progress fingerspelling first.
            self._flush_word()
            asl = readable(gloss)
            if asl:
                token = asl if asl.isupper() else asl.title()
            else:
                token = gloss.replace("_", " ").title()
            self._state.tokens.append(token)

    def _backspace(self) -> None:
        if self._state.current_word:
            self._state.current_word = self._state.current_word[:-1]
        elif self._state.tokens:
            self._state.tokens.pop()

    def _flush_word(self) -> None:
        if self._state.current_word:
            self._state.tokens.append(self._state.current_word.title())
            self._state.current_word = ""

    # ── boundaries ───────────────────────────────────────────────────────
    def hands_lost(self) -> None:
        """Signal that hands left the frame: allow the next sign to repeat."""
        self._last_committed = None

    def end_sentence(self) -> str | None:
        """Finalize the current sentence (called after a long hands-absent pause).

        Returns:
            The completed sentence text, or None if nothing was buffered.
        """
        self._flush_word()
        if not self._state.tokens:
            return None
        sentence = " ".join(self._state.tokens).strip()
        self._state.sentences.append(sentence)
        self.last_tokens = list(self._state.tokens)
        self._state.tokens = []
        self._last_committed = None
        return sentence

    # ── views ────────────────────────────────────────────────────────────
    @property
    def current_word(self) -> str:
        """Letters accumulated since the last word break."""
        return self._state.current_word

    @property
    def pending_sentence(self) -> str:
        """The in-progress sentence including the current word."""
        parts = [*self._state.tokens]
        if self._state.current_word:
            parts.append(self._state.current_word.title())
        return " ".join(parts)

    @property
    def sentences(self) -> list[str]:
        """All finalized sentences this session."""
        return list(self._state.sentences)

    @property
    def transcript(self) -> str:
        """All finalized sentences joined into one copyable block."""
        return " ".join(self._state.sentences)

    def reset(self) -> None:
        """Clear all state (new session)."""
        self._state = BuilderState()
        self._cooldown = 0
        self._last_committed = None
