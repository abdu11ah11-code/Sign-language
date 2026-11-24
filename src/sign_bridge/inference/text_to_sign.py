"""Direction 2 — text → sign pipeline.

Converts English text (typed, or transcribed from speech via Whisper) into
a playable sequence of sign clips:

    text ─▶ grammar adapt ─▶ tokenize ─▶ resolve each token to a clip
         ─▶ (optional) moviepy concat with caption overlay

Resolution per token:
    SIGN          → whole-sign clip from the video library
    FINGERSPELL   → sequence of letter clips (skips letters with no clip)
    PUNCT         → a short pause (no clip)

If a whole-sign clip is missing, the word falls back to fingerspelling so
the system degrades gracefully instead of dropping content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sign_bridge.data.video_library import VideoLibrary
from sign_bridge.language.grammar import reorder
from sign_bridge.language.tokenizer import SignToken, Tokenizer, TokenKind
from sign_bridge.logging_conf import get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)

_CLAUSE_RE = re.compile(r"[^.?!,]+|[.?!,]")
_OUT_SIZE = (640, 480)


def _caption_frame(frame: NDArray, text: str) -> NDArray:
    """Burn a caption band into an RGB frame with OpenCV.

    Replaces moviepy's ``TextClip``, which needs ImageMagick installed and
    therefore failed on most Windows machines (the whole video then fell
    back to the text storyboard).
    """
    import cv2

    out = np.ascontiguousarray(frame).copy()
    h, w = out.shape[:2]
    band = max(40, h // 9)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - band), (w, h), (8, 11, 13), thickness=-1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    safe = text.replace("\u00b7", "-").encode("ascii", "replace").decode("ascii")
    scale = 1.0
    while scale > 0.4 and cv2.getTextSize(safe, font, scale, 2)[0][0] > w - 24:
        scale -= 0.05
    (tw, th), _ = cv2.getTextSize(safe, font, scale, 2)
    org = ((w - tw) // 2, h - (band - th) // 2)
    cv2.putText(out, safe, org, font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


@dataclass(frozen=True)
class SignClip:
    """One playable element of the output sequence.

    Attributes:
        caption: Text shown while this element plays.
        video_path: Clip to play, or None for a pause.
        is_pause: True for punctuation gaps.
        is_fallback: True when a word was fingerspelled for lack of a sign.
    """

    caption: str
    video_path: Path | None
    is_pause: bool = False
    is_fallback: bool = False


@dataclass
class SignSequence:
    """A resolved text→sign plan.

    Attributes:
        clips: Ordered playable elements.
        missing: Letters/words that could not be rendered at all.
        gloss: The adapted gloss string (for the caption track / debugging).
    """

    clips: list[SignClip] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    gloss: str = ""


class TextToSignPipeline:
    """Resolves English text to a sign-clip sequence for a given language.

    Args:
        library: The :class:`VideoLibrary` for the active language.
        use_grammar: Apply ASL/PSL gloss reordering before tokenizing.
    """

    def __init__(self, library: VideoLibrary, use_grammar: bool = True) -> None:
        self._library = library
        self._use_grammar = use_grammar
        self._tokenizer = Tokenizer(set(library.available_signs()))

    def refresh(self) -> None:
        """Re-scan the library and rebuild the tokenizer (after new clips)."""
        self._library.refresh()
        self._tokenizer = Tokenizer(set(self._library.available_signs()))

    def plan(self, text: str) -> SignSequence:
        """Build a :class:`SignSequence` from input text (no rendering).

        Phrases are matched *before* grammar reordering, so a sign such as
        GOOD MORNING stays one sign instead of being split and reordered,
        and each clause (``.``, ``?``, ``!``, ``,``) is glossed separately
        with a short pause between clauses.
        """
        seq = SignSequence()
        gloss_parts: list[str] = []
        for chunk in _CLAUSE_RE.findall(text):
            if chunk in ".?!,":
                if seq.clips and not seq.clips[-1].is_pause:
                    seq.clips.append(SignClip(caption=chunk, video_path=None, is_pause=True))
                continue
            tokens = [t for t in self._tokenizer.tokenize(chunk) if t.kind is not TokenKind.PUNCT]
            if self._use_grammar:
                tokens = reorder(tokens, key=lambda t: t.text)
            for tok in tokens:
                self._resolve_token(tok, seq)
                gloss_parts.append((tok.gloss or tok.text).upper())
        while seq.clips and seq.clips[-1].is_pause:
            seq.clips.pop()
        seq.gloss = " ".join(gloss_parts)
        log.info("text_to_sign_planned", clips=len(seq.clips), missing=len(seq.missing))
        return seq

    def _resolve_token(self, tok: SignToken, seq: SignSequence) -> None:
        if tok.kind is TokenKind.PUNCT:
            seq.clips.append(SignClip(caption=tok.text, video_path=None, is_pause=True))
            return

        if tok.kind is TokenKind.SIGN and tok.gloss is not None:
            path = self._library.path_for(tok.gloss)
            if path is not None:
                seq.clips.append(SignClip(caption=tok.text, video_path=path))
                return
            # Whole-sign clip missing → fall back to fingerspelling.
            self._fingerspell(tok.text, seq)
            return

        # FINGERSPELL token.
        self._fingerspell(tok.text, seq)

    def _fingerspell(self, word: str, seq: SignSequence) -> None:
        for ch in word.upper():
            if not ch.isalpha():
                continue
            path = self._library.path_for_letter(ch)
            if path is not None:
                seq.clips.append(
                    SignClip(caption=f"{word} · {ch}", video_path=path, is_fallback=True)
                )
            else:
                seq.missing.append(ch)

    # ── optional rendering (requires moviepy) ────────────────────────────
    def render(
        self,
        sequence: SignSequence,
        output_path: Path,
        pause_seconds: float = 0.4,
        add_captions: bool = True,
    ) -> Path:
        """Stitch a sequence into one MP4 with caption overlays.

        Args:
            sequence: A plan from :meth:`plan`.
            output_path: Destination ``.mp4``.
            pause_seconds: Gap inserted for punctuation tokens.
            add_captions: Burn the caption text into each segment.

        Returns:
            The path written.

        Raises:
            ValueError: If the sequence has no playable clips.
        """
        import PIL.Image

        # moviepy 1.0.3 references Image.ANTIALIAS, removed in Pillow 10.
        if not hasattr(PIL.Image, "ANTIALIAS"):
            PIL.Image.ANTIALIAS = PIL.Image.LANCZOS  # type: ignore[attr-defined]

        from moviepy.editor import (  # local import: heavy, optional at runtime
            ColorClip,
            VideoFileClip,
            concatenate_videoclips,
        )

        sources: list[Any] = []
        segments: list[Any] = []
        try:
            for clip in sequence.clips:
                if clip.is_pause or clip.video_path is None:
                    segments.append(
                        ColorClip(size=_OUT_SIZE, color=(13, 11, 8), duration=pause_seconds)
                    )
                    continue
                source = VideoFileClip(str(clip.video_path), audio=False)
                sources.append(source)
                video = source.resize(newsize=_OUT_SIZE)
                if add_captions:
                    caption = clip.caption
                    video = video.fl_image(lambda f, c=caption: _caption_frame(f, c))
                segments.append(video)

            if not any(not c.is_pause and c.video_path is not None for c in sequence.clips):
                raise ValueError("Nothing to render: no playable clips in sequence.")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            final = concatenate_videoclips(segments, method="compose")
            final.write_videofile(str(output_path), fps=24, codec="libx264",
                                  audio=False, logger=None)
            final.close()
        finally:
            for source in sources:
                source.close()
        log.info("sequence_rendered", path=str(output_path))
        return output_path
