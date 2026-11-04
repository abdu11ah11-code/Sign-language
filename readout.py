"""Reusable Field Guide readout fragments.

Small render helpers shared by the tabs so the editorial styling stays
consistent: the focal detected sign, the hairline confidence gauge, the
word-in-progress line, the numbered transcript, and the status footer.
Each emits themed HTML defined in :mod:`sign_bridge.ui.theme`.
"""

from __future__ import annotations

import html

import streamlit as st


def focal_sign(label: str | None, locked: bool) -> None:
    """Render the large current-sign display.

    Args:
        label: Current sign text, or None when idle.
        locked: True when the sign has locked (adds the green full stop).
    """
    if not label:
        st.markdown('<div class="fg-label">Current sign</div>'
                    '<div class="fg-sign idle">waiting for hands…</div>',
                    unsafe_allow_html=True)
        return
    cls = "fg-sign locked" if locked else "fg-sign"
    st.markdown(f'<div class="fg-label">Current sign</div>'
                f'<div class="{cls}">{html.escape(label.title())}</div>',
                unsafe_allow_html=True)


def confidence_gauge(confidence: float) -> None:
    """Render the thin confidence line with a marker dot and caption.

    Args:
        confidence: Value in [0, 1].
    """
    pct = max(0.0, min(1.0, confidence)) * 100.0
    st.markdown(
        f'<div class="fg-gauge"><div class="fill" style="width:{pct:.0f}%"></div>'
        f'<div class="dot" style="left:{pct:.0f}%"></div></div>'
        f'<div class="fg-gauge-cap">Recognized with <em>{pct:.0f}% confidence</em>'
        f' over the last window</div>',
        unsafe_allow_html=True,
    )


def word_in_progress(word: str, pending_sentence: str) -> None:
    """Render the letters being fingerspelled and the pending sentence.

    Args:
        word: Current word buffer (fingerspelling).
        pending_sentence: The sentence assembled so far.
    """
    spelled = html.escape(word) if word else ""
    st.markdown('<div class="fg-label">Word in progress</div>'
                f'<div class="fg-building">{spelled}'
                '<span class="pending">▏</span></div>',
                unsafe_allow_html=True)
    if pending_sentence:
        st.markdown('<div class="fg-label" style="margin-top:.8rem">Sentence</div>'
                    f'<div class="fg-tagline" style="font-size:1em">'
                    f'{html.escape(pending_sentence)}</div>',
                    unsafe_allow_html=True)


def transcript(entries: list[tuple[str, str]], context: str = "main") -> None:
    """Render the numbered, copyable transcript.

    Args:
        entries: List of ``(timestamp, sentence)`` newest-last.
        context: Unique prefix for the copy box's key (avoids duplicate-id
            errors when the transcript is shown on more than one tab).
    """
    st.markdown('<div class="fg-label">Transcript</div>', unsafe_allow_html=True)
    if not entries:
        st.markdown('<div class="fg-tagline">Nothing yet — signed sentences '
                    'will appear here.</div>', unsafe_allow_html=True)
        return
    for i, (ts, sentence) in enumerate(entries, start=1):
        st.markdown(
            f'<div class="fg-entry"><span class="idx">{i:02d}</span>'
            f'<span style="flex:1">{html.escape(sentence)}</span>'
            f'<span class="time">{html.escape(ts)}</span></div>',
            unsafe_allow_html=True,
        )
    st.text_area(
        "Copy transcript",
        value="\n".join(s for _, s in entries),
        height=90,
        label_visibility="collapsed",
        key=f"{context}_copy_transcript",
    )


def status_line(items: list[tuple[str, bool]]) -> None:
    """Render the status footer.

    Args:
        items: List of ``(text, is_ok)`` pairs; ok items render in green.
    """
    parts: list[str] = []
    for idx, (text, ok) in enumerate(items):
        cls = "on" if ok else ""
        parts.append(f'<span class="{cls}">{html.escape(text)}</span>')
        if idx < len(items) - 1:
            parts.append('<span class="sep">·</span>')
    st.markdown(f'<div class="fg-status">{"".join(parts)}</div>', unsafe_allow_html=True)
