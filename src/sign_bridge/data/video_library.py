"""Video library: maps sign glosses to pre-recorded clips on disk.

Directory contract (created by ``sign-collect``, extendable by hand):

    assets/sign_videos/
        asl/
            HELLO.mp4            whole-word signs, gloss-named, UPPERCASE
            THANK_YOU.mp4        spaces in glosses become underscores
            letters/
                A.mp4 … Z.mp4    fingerspelling alphabet
        psl/
            SALAAM.mp4
            letters/ …

Lookups are case-insensitive and tolerate ``.mp4``/``.mov``/``.webm``.
"""

from __future__ import annotations

from pathlib import Path

from sign_bridge.logging_conf import get_logger

log = get_logger(__name__)

_EXTENSIONS = (".mp4", ".mov", ".webm")


class VideoLibrary:
    """Read-only index over one language's sign clips.

    Args:
        root: The ``sign_videos`` directory (containing ``asl/``, ``psl/``).
        language: ``"asl"`` or ``"psl"``.
    """

    def __init__(self, root: Path, language: str) -> None:
        self._dir = Path(root) / language
        self._letters_dir = self._dir / "letters"
        self._index: dict[str, Path] = {}
        self._letter_index: dict[str, Path] = {}
        self.refresh()

    def refresh(self) -> None:
        """Re-scan the directory (call after adding clips at runtime)."""
        self._index.clear()
        self._letter_index.clear()
        if self._dir.is_dir():
            for p in self._dir.iterdir():
                if p.is_file() and p.suffix.lower() in _EXTENSIONS:
                    self._index[p.stem.upper()] = p
        if self._letters_dir.is_dir():
            for p in self._letters_dir.iterdir():
                if p.is_file() and p.suffix.lower() in _EXTENSIONS:
                    self._letter_index[p.stem.upper()] = p
        log.info(
            "video_library_indexed",
            dir=str(self._dir),
            signs=len(self._index),
            letters=len(self._letter_index),
        )

    # ── lookups ──────────────────────────────────────────────────────────
    def path_for(self, gloss: str) -> Path | None:
        """Return the clip for a whole-sign gloss, or None."""
        return self._index.get(gloss.strip().upper().replace(" ", "_"))

    def path_for_letter(self, letter: str) -> Path | None:
        """Return the fingerspelling clip for a single letter, or None."""
        if len(letter) != 1 or not letter.isalpha():
            return None
        return self._letter_index.get(letter.upper())

    def has(self, gloss: str) -> bool:
        """True when a whole-sign clip exists for this gloss."""
        return self.path_for(gloss) is not None

    def can_fingerspell(self, word: str) -> bool:
        """True when every letter of ``word`` has a letter clip."""
        return all(self.path_for_letter(ch) is not None for ch in word if ch.isalpha())

    def available_signs(self) -> list[str]:
        """Sorted list of whole-sign glosses present on disk."""
        return sorted(self._index)

    def available_letters(self) -> list[str]:
        """Sorted list of letters present in the fingerspelling set."""
        return sorted(self._letter_index)
