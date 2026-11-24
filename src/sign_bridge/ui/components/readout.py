"""Reusable Field Guide readout fragments.

Each fragment exists in two forms:

* ``*_html(...)`` returns the themed HTML string, so the live camera loop can
  repaint a placeholder in place (``slot.markdown(html)``) many times per
  second without creating new widgets;
* the plain function (``focal_sign(...)`` etc.) writes it once with
  ``st.markdown`` for static parts of the page.

Classes used here are defined in :mod:`sign_bridge.ui.theme`.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

import streamlit as st

from sign_bridge.language.psl_lexicon import display_name

#: A transcript entry: ``(time, text)`` or ``(time, text, speaker)`` where
#: speaker is ``"signer"`` (recognized signs) or ``"speaker"`` (typed/spoken).
Entry = tuple[str, ...]


def _entry_parts(entry: Entry) -> tuple[str, str, str]:
    ts, text = entry[0], entry[1]
    who = entry[2] if len(entry) > 2 else "signer"
    return ts, text, who


# ── focal sign ───────────────────────────────────────────────────────────
def focal_sign_html(label: str | None, locked: bool, hands_visible: bool = True) -> str:
    """Large current-sign display."""
    if not label:
        idle = "waiting for hands…" if not hands_visible else "reading the sign…"
        return ('<div class="fg-label">Current sign</div>'
                f'<div class="fg-sign idle">{idle}</div>')
    cls = "fg-sign locked" if locked else "fg-sign"
    return (f'<div class="fg-label">Current sign</div>'
            f'<div class="{cls}">{html.escape(display_name(label))}</div>')


def segment_sign_html(
    label: str | None, unsure: bool, phase: str, hands_visible: bool, language: str = "asl",
    live: str | None = None, live_confidence: float = 0.0,
) -> str:
    """Current-sign display for one-guess-per-sign (segment) recognition.

    While a sign is still moving, ``live`` is the model's guess from the
    movement so far; it is shown faded, and the answer is written down only
    when the sign ends.
    """
    head = '<div class="fg-label">Current sign</div>'
    if phase == "signing":
        if live:
            return (head + f'<div class="fg-sign idle">{html.escape(display_name(live))}…</div>'
                    f'<div class="fg-hint">watching the movement — looks like '
                    f'{html.escape(display_name(live))} ({live_confidence:.0%}) so far</div>')
        return head + '<div class="fg-sign idle">watching your sign…</div>'
    if unsure:
        return (head + '<div class="fg-sign unsure">not sure</div>'
                '<div class="fg-hint">Best guesses below — try the sign again.</div>')
    if label:
        link = ""
        if language == "asl":
            from sign_bridge.language.asl_lexicon import lookup_url

            link = (f'<div class="fg-howto"><a href="{html.escape(lookup_url(label))}" '
                    f'target="_blank">How to sign “{html.escape(display_name(label))}” ↗</a></div>')
        return head + f'<div class="fg-sign locked">{html.escape(display_name(label))}</div>' + link
    idle = "ready — sign now" if hands_visible else "show your hands to start"
    return head + f'<div class="fg-sign idle">{idle}</div>'


def focal_sign(label: str | None, locked: bool) -> None:
    """Render the large current-sign display."""
    st.markdown(focal_sign_html(label, locked, hands_visible=False), unsafe_allow_html=True)


# ── confidence ───────────────────────────────────────────────────────────
def confidence_gauge_html(confidence: float) -> str:
    """Thin confidence line with a marker dot and caption."""
    pct = max(0.0, min(1.0, confidence)) * 100.0
    return (
        f'<div class="fg-gauge"><div class="fill" style="width:{pct:.0f}%"></div>'
        f'<div class="dot" style="left:{pct:.0f}%"></div></div>'
        f'<div class="fg-gauge-cap">Recognized with <em>{pct:.0f}% confidence</em>'
        f' over the last window</div>'
    )


def confidence_gauge(confidence: float) -> None:
    """Render the confidence gauge."""
    st.markdown(confidence_gauge_html(confidence), unsafe_allow_html=True)


def top_predictions_html(
    top: Sequence[tuple[str, float]], threshold: float, title: str = "Top candidates",
) -> str:
    """Ranked candidate signs with probability bars (live model insight).

    Args:
        top: ``(label, probability)`` pairs, best first.
        threshold: Lock threshold, drawn as a tick on every bar.
    """
    if not top:
        return ""
    rows = []
    tick = max(0.0, min(1.0, threshold)) * 100
    for label, prob in top:
        pct = max(0.0, min(1.0, prob)) * 100
        rows.append(
            '<div class="fg-cand">'
            f'<span class="name">{html.escape(display_name(label))}</span>'
            f'<span class="bar"><span class="fill" style="width:{pct:.0f}%"></span>'
            f'<span class="tick" style="left:{tick:.0f}%"></span></span>'
            f'<span class="pct">{pct:.0f}%</span></div>'
        )
    return (f'<div class="fg-label" style="margin-top:.6rem">{html.escape(title)}</div>'
            + "".join(rows))


# ── word / sentence ──────────────────────────────────────────────────────
def word_html(word: str, pending_sentence: str) -> str:
    """Letters being fingerspelled plus the pending sentence."""
    out = ('<div class="fg-label">Word in progress</div>'
           f'<div class="fg-building">{html.escape(word)}'
           '<span class="pending">▏</span></div>')
    if pending_sentence:
        out += ('<div class="fg-label" style="margin-top:.8rem">Sentence</div>'
                f'<div class="fg-tagline" style="font-size:1em">'
                f'{html.escape(pending_sentence)}</div>')
    return out


def ai_bar_html(ai_last: dict[str, Any] | None, enabled: bool) -> str:
    """The "AI sentence" block: signed words → English reading(s)."""
    head = '<div class="fg-label" style="margin-top:.8rem">AI sentence</div>'
    if not enabled:
        return head + ('<div class="fg-hint">Off — switch on "Write sentences with AI" '
                       'in the sidebar.</div>')
    if not ai_last:
        return head + ('<div class="fg-hint">Sign a few words, then lower your hands '
                       '(or press Finish) — the AI writes them as an English sentence.</div>')
    signs = " · ".join(html.escape(s.upper()) for s in ai_last.get("signs", []))
    out = head + f'<div class="fg-hint">🤟 {signs}</div>'
    if ai_last.get("pending"):
        return out + '<div class="fg-tagline" style="font-size:1em">🤖 writing…</div>'
    sentences = list(ai_last.get("sentences") or [])
    if sentences:
        out += f'<div class="fg-ai">🤖 {html.escape(sentences[0])}</div>'
    if len(sentences) > 1:
        alts = " / ".join(f"“{html.escape(x)}”" for x in sentences[1:])
        out += f'<div class="fg-hint">or: {alts}</div>'
    if ai_last.get("error"):
        out += f'<div class="fg-hint">AI not used: {html.escape(str(ai_last["error"]))}</div>'
    return out


def word_in_progress(word: str, pending_sentence: str) -> None:
    """Render the word-in-progress block."""
    st.markdown(word_html(word, pending_sentence), unsafe_allow_html=True)


# ── transcript ───────────────────────────────────────────────────────────
def transcript_html(entries: Sequence[Entry], title: str = "Transcript") -> str:
    """Numbered transcript; entries from the hearing side are tagged."""
    head = f'<div class="fg-label">{html.escape(title)}</div>'
    if not entries:
        return head + ('<div class="fg-tagline">Nothing yet — signed sentences '
                       'will appear here.</div>')
    rows = []
    for i, entry in enumerate(entries, start=1):
        ts, text, who = _entry_parts(entry)
        tag = "Speaker" if who == "speaker" else "Signer"
        rows.append(
            f'<div class="fg-entry {who}"><span class="idx">{i:02d}</span>'
            f'<span class="who">{tag}</span>'
            f'<span style="flex:1">{html.escape(text)}</span>'
            f'<span class="time">{html.escape(ts)}</span></div>'
        )
    return head + "".join(rows)


def transcript(entries: Sequence[Entry], context: str = "main") -> None:
    """Render the transcript plus a copy box.

    ``st.code`` carries a built-in copy button and, unlike a keyed
    ``text_area``, never goes stale or collides on widget ids when the
    transcript is shown on more than one tab.
    """
    del context  # kept for API compatibility
    st.markdown(transcript_html(entries), unsafe_allow_html=True)
    if entries:
        st.code("\n".join(_entry_parts(e)[1] for e in entries), language=None)


def last_signed(entries: Sequence[Entry]) -> str | None:
    """Most recent sentence produced by the signer, if any."""
    for entry in reversed(entries):
        _, text, who = _entry_parts(entry)
        if who == "signer":
            return text
    return None


# ── status line ──────────────────────────────────────────────────────────
def status_html(items: Sequence[tuple[str, bool]]) -> str:
    """Status footer; ok items render in green."""
    parts: list[str] = []
    for idx, (text, ok) in enumerate(items):
        cls = "on" if ok else ""
        parts.append(f'<span class="{cls}">{html.escape(text)}</span>')
        if idx < len(items) - 1:
            parts.append('<span class="sep">·</span>')
    return f'<div class="fg-status">{"".join(parts)}</div>'


def status_line(items: Sequence[tuple[str, bool]]) -> None:
    """Render the status footer."""
    st.markdown(status_html(items), unsafe_allow_html=True)
