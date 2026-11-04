"""Sign player — the Speech → Sign tab body.

Takes typed text or a microphone recording (transcribed locally with
Whisper), plans an English→sign clip sequence, and plays it back. When the
full sequence is renderable (moviepy + clips present) it stitches one
captioned MP4; otherwise it falls back to a clear, accessible storyboard of
the planned clips and captions so the feature degrades gracefully on a
fresh install with no videos yet.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from sign_bridge.config import AppConfig


@st.cache_resource(show_spinner=False)
def _library(language: str, root: str) -> Any:
    """Build (and cache) the video library for a language."""
    from pathlib import Path

    from sign_bridge.data.video_library import VideoLibrary

    return VideoLibrary(Path(root), language)


def _transcribe(config: AppConfig) -> str | None:
    """Record from the mic and transcribe with Whisper. Returns text or None."""
    try:
        from sign_bridge.capture.microphone import record_audio
        from sign_bridge.inference.stt import SpeechToText
    except Exception as exc:  # noqa: BLE001
        st.error(f"Speech input unavailable: {exc}")
        return None

    with st.spinner(f"Recording {config.audio.max_record_seconds:.0f}s — speak now…"):
        try:
            audio = record_audio(config.audio.max_record_seconds,
                                 config.audio.sample_rate, config.audio.channels)
        except Exception as exc:  # noqa: BLE001 — no mic / permission
            st.error(f"Could not record audio: {exc}")
            return None
    with st.spinner("Transcribing locally with Whisper…"):
        stt = SpeechToText(config.stt.whisper_model, config.stt.language)
        return stt.transcribe(audio)


def render(config: AppConfig, context: str = "speech") -> None:
    """Render the Speech → Sign tab.

    Args:
        config: Loaded application config.
        context: Unique prefix for widget keys, so the same component can be
            rendered on more than one tab (e.g. the Conversation tab) without
            Streamlit duplicate-element-id collisions.
    """
    language = st.session_state["language"]
    library = _library(language, str(config.resolve(config.paths.video_dir)))

    st.markdown('<div class="fg-label">Type or speak</div>', unsafe_allow_html=True)
    text = st.text_input("Text to translate into signs",
                         placeholder="e.g. Hello, thank you",
                         label_visibility="collapsed",
                         value=st.session_state.get("_tts_text", ""),
                         key=f"{context}_tts_input")
    c1, c2, _ = st.columns([1, 1, 2])
    translate = c1.button("Translate", use_container_width=True, key=f"{context}_translate")
    if c2.button("Record voice", use_container_width=True, key=f"{context}_record"):
        spoken = _transcribe(config)
        if spoken:
            st.session_state["_tts_text"] = spoken
            st.rerun()

    available = library.available_signs()
    if not available:
        st.info(
            "No sign videos installed yet for this language.\n\n"
            "Record signs with `sign-collect` (it seeds the video library "
            "automatically), or drop clips into "
            f"`{config.paths.video_dir}/{language}/`. The translator below will "
            "show the planned sequence regardless.")

    if not (translate and text.strip()):
        if available:
            st.markdown('<div class="fg-figcap"><span>Available signs: '
                        f'{", ".join(available[:20])}</span></div>',
                        unsafe_allow_html=True)
        return

    # Plan the sequence (always possible — rendering is best-effort).
    from sign_bridge.inference.text_to_sign import TextToSignPipeline

    pipeline = TextToSignPipeline(library, use_grammar=True)
    sequence = pipeline.plan(text)

    st.markdown(f'<div class="fg-label" style="margin-top:1rem">Sign gloss</div>'
                f'<div class="fg-sign" style="font-size:calc(var(--base)*1.8)">'
                f'{html.escape(sequence.gloss)}</div>', unsafe_allow_html=True)

    playable = [c for c in sequence.clips if c.video_path is not None]
    rendered = _try_render(pipeline, sequence, config, language) if playable else None

    if rendered is not None:
        st.video(str(rendered))
    else:
        _storyboard(sequence)

    if st.session_state.get("captions_on", True):
        caption_track = " · ".join(c.caption for c in sequence.clips if not c.is_pause)
        st.markdown('<div class="fg-label" style="margin-top:1rem">Captions</div>'
                    f'<div class="fg-tagline" style="font-size:1em">'
                    f'{html.escape(caption_track)}</div>', unsafe_allow_html=True)

    if sequence.missing:
        st.warning("No clip for: " + ", ".join(sorted(set(sequence.missing)))
                   + " — add them with sign-collect to complete the sequence.")


def _try_render(pipeline: Any, sequence: Any, config: AppConfig, language: str) -> Any:
    """Attempt to stitch the sequence to one MP4; return path or None."""
    from pathlib import Path

    try:
        out = config.resolve(config.paths.recordings_dir) / f"_preview_{language}.mp4"
        return pipeline.render(sequence, Path(out),
                               add_captions=st.session_state.get("captions_on", True))
    except Exception as exc:  # noqa: BLE001 — moviepy/codec/clip issues
        st.info(f"Showing storyboard (video stitching unavailable: {exc}).")
        return None


def _storyboard(sequence: Any) -> None:
    """Accessible fallback: render the planned clips as a captioned list."""
    st.markdown('<div class="fg-label" style="margin-top:1rem">Planned sequence'
                '</div>', unsafe_allow_html=True)
    for i, clip in enumerate(sequence.clips, start=1):
        if clip.is_pause:
            label, note = "—", "pause"
        elif clip.video_path is None:
            label, note = clip.caption, "missing clip"
        elif clip.is_fallback:
            label, note = clip.caption, "fingerspelled"
        else:
            label, note = clip.caption, "sign"
        st.markdown(
            f'<div class="fg-entry"><span class="idx">{i:02d}</span>'
            f'<span style="flex:1">{html.escape(label)}</span>'
            f'<span class="time">{note}</span></div>', unsafe_allow_html=True)
