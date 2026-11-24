"""Sign player — the Speech → Sign tab body.

Takes typed text or speech, recorded in the browser (``st.audio_input``: press
the mic, speak, press stop) and transcribed locally with Whisper. Then:

* **3D person** (when the version-2 animation library and the 3D person are
  installed): the sentence is planned against the library's signs
  (:mod:`sign_bridge.inference.text_to_motion`) and the person signs it, one
  sign after another, with the current word shown above and a red spot
  wherever a finger touches the face;
* otherwise the older path: an English→sign plan over recorded sign
  *videos*, stitched into one captioned MP4, or a storyboard when there are
  no videos yet.

State per context (the component appears on two tabs):

    <ctx>_tts_input      the text box (a keyed widget)
    <ctx>_last_text      text of the most recent translation (persists across
                         reruns, so a sidebar click no longer wipes the result)
    <ctx>_render_key     what the cached MP4 was rendered from
    <ctx>_mic            the browser recorder (a keyed widget)
    <ctx>_mic_done       which recording was already transcribed
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from sign_bridge.config import AppConfig


@st.cache_resource(show_spinner=False)
def _library(language: str, root: str) -> Any:
    """Build (and cache) the video library for a language."""
    from sign_bridge.data.video_library import VideoLibrary

    return VideoLibrary(Path(root), language)


def wav_to_whisper(data: bytes, rate: int = 16_000) -> Any:
    """Decode a recorded WAV into what Whisper takes: mono float32 at 16 kHz."""
    import io

    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != rate and len(mono):
        from math import gcd

        from scipy.signal import resample_poly  # type: ignore[import-untyped]

        g = gcd(int(sr), rate)
        mono = resample_poly(mono, rate // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _transcribe(config: AppConfig, recording: bytes) -> str | None:
    """Transcribe a browser recording with Whisper. Returns text or None."""
    try:
        from sign_bridge.inference.stt import SpeechToText

        audio = wav_to_whisper(recording, config.audio.sample_rate)
    except Exception as exc:  # noqa: BLE001 — whisper/soundfile missing, bad file
        st.error(f"Speech input unavailable: {exc}")
        return None
    if len(audio) < config.audio.sample_rate * 0.3:
        st.warning("That recording was too short — press the mic, speak, then press stop.")
        return None
    with st.spinner("Writing down what you said (Whisper, on this computer)…"):
        try:
            stt = SpeechToText(config.stt.whisper_model, config.stt.language)
            return stt.transcribe(audio) or None
        except Exception as exc:  # noqa: BLE001 — whisper/torch missing, etc.
            st.error(f"Transcription failed: {exc}")
            return None


# ── callbacks ────────────────────────────────────────────────────────────
def _submit(context: str) -> None:
    """Translate the current text (Translate button or Enter in the box)."""
    text = st.session_state.get(f"{context}_tts_input", "").strip()
    if not text:
        return
    st.session_state[f"{context}_last_text"] = text
    if context.startswith("conv"):
        log = st.session_state["transcript"]
        # Enter + Translate in one go fires both callbacks: log it once.
        if not (log and log[-1][1:] == (text, "speaker")):
            log.append((datetime.now().strftime("%H:%M"), text, "speaker"))




# ── render ───────────────────────────────────────────────────────────────
def render(config: AppConfig, context: str = "speech") -> None:
    """Render the Speech → Sign tab.

    Args:
        config: Loaded application config.
        context: Unique prefix for widget keys, so the same component can be
            rendered on more than one tab without duplicate-id collisions.
    """
    language = st.session_state["language"]
    library = _library(language, str(config.resolve(config.paths.video_dir)))
    input_key = f"{context}_tts_input"

    st.markdown('<div class="fg-label">Type or speak</div>', unsafe_allow_html=True)

    # Speech first: the recorder sits in the browser (press the mic, speak, press
    # stop). A new recording is transcribed *before* the text box is created —
    # Streamlit only lets us set a widget's value before it is instantiated.
    recording = st.audio_input("Speak — press the mic, talk, then press stop",
                               key=f"{context}_mic")
    if recording is not None:
        data = recording.getvalue()
        stamp = f"{len(data)}:{hash(data)}"
        if st.session_state.get(f"{context}_mic_done") != stamp:
            st.session_state[f"{context}_mic_done"] = stamp
            spoken = _transcribe(config, data)
            if spoken:
                st.session_state[input_key] = spoken
                _submit(context)

    st.text_input("Or type it",
                  placeholder="e.g. Hello, my mother is drinking water — press Enter",
                  key=input_key, on_change=_submit, args=(context,))
    c1, _ = st.columns([1, 3])
    c1.button("Translate", width="stretch", key=f"{context}_translate",
              on_click=_submit, args=(context,), type="primary")

    if _person_signs(config, language, context):
        return

    available = library.available_signs()
    if not available:
        st.info(
            "No sign videos installed yet for this language.\n\n"
            "Record signs with `sign-collect` (it seeds the video library "
            "automatically), or drop clips into "
            f"`{config.paths.video_dir}/{language}/`. The translator below will "
            "show the planned sequence regardless.")

    text = st.session_state.get(f"{context}_last_text", "")
    if not text:
        if available:
            names = ", ".join(s.replace("_", " ") for s in available[:30])
            st.markdown(f'<div class="fg-figcap"><span>Available signs: {html.escape(names)}'
                        '</span></div>', unsafe_allow_html=True)
        return

    from sign_bridge.inference.text_to_sign import TextToSignPipeline

    pipeline = TextToSignPipeline(library, use_grammar=True)
    sequence = pipeline.plan(text)

    st.markdown(f'<div class="fg-label" style="margin-top:1rem">Sign gloss</div>'
                f'<div class="fg-sign" style="font-size:calc(var(--base)*1.8)">'
                f'{html.escape(sequence.gloss)}</div>', unsafe_allow_html=True)

    playable = [c for c in sequence.clips if c.video_path is not None]
    rendered = (
        _try_render(pipeline, sequence, config, language, context, text) if playable else None
    )

    if rendered is not None:
        st.video(str(rendered), autoplay=True)
    else:
        _storyboard(sequence)

    if st.session_state.get("captions_on", True):
        caption_track = _caption_track(sequence)
        st.markdown('<div class="fg-label" style="margin-top:1rem">Captions</div>'
                    f'<div class="fg-tagline" style="font-size:1em">'
                    f'{html.escape(caption_track)}</div>', unsafe_allow_html=True)

    if sequence.missing:
        st.warning("No clip for: " + ", ".join(sorted(set(sequence.missing)))
                   + " — add them with sign-collect to complete the sequence.")


def _person_signs(config: AppConfig, language: str, context: str) -> bool:
    """Show the 3D person signing the last text. False when it is not installed."""
    from sign_bridge.ui.components.camera_view import _sign_library
    from sign_bridge.ui.figure3d import person_model

    library = _sign_library(language, id(config))
    if not (len(library) and library.has_depth and person_model()):
        return False
    text = st.session_state.get(f"{context}_last_text", "")
    if not text:
        st.markdown('<div class="fg-figcap"><span>The 3D person signs what you say or type. '
                    f'It knows {len(library)} signs.</span></div>', unsafe_allow_html=True)
        return True

    from sign_bridge.inference.text_to_motion import plan_motion
    from sign_bridge.language.psl_lexicon import display_name
    from sign_bridge.ui.figure3d import join_clips, sign_3d_html

    plan = plan_motion(text, library.glosses())
    st.markdown(f'<div class="fg-label" style="margin-top:1rem">You said</div>'
                f'<div class="fg-tagline" style="font-size:1.15em">{html.escape(text)}</div>',
                unsafe_allow_html=True)
    shown = " ".join(
        f'<span>{html.escape(display_name(w.gloss).upper())}</span>' if w.gloss else
        f'<span style="opacity:.45;text-decoration:line-through">{html.escape(w.word)}</span>'
        for w in plan.words)
    st.markdown(f'<div class="fg-label" style="margin-top:.8rem">In signs</div>'
                f'<div class="fg-sign" style="font-size:calc(var(--base)*1.5)">{shown}</div>',
                unsafe_allow_html=True)
    clips = [c for c in (library.get(g) for g in plan.glosses) if c is not None]
    if not clips:
        st.info("None of these words has a sign yet. Try words like: hello, mother, "
                "drink, water, thank you, happy, dog.")
        return True
    left, right = st.columns([1, 1])
    slow = left.toggle("Slow", value=False, key=f"{context}_slow")
    mirror = right.toggle("Mirror", value=False, key=f"{context}_mirror",
                          help="Off: as if the person signs to you. On: copy it like a mirror.")
    joined, segments = join_clips(clips, [display_name(c.gloss).upper() for c in clips])
    page = sign_3d_html(joined, speed=0.6 if slow else 1.0, mirror=mirror, height=440,
                        title=" ".join(s.word for s in segments), segments=segments)
    if hasattr(st, "iframe"):
        st.iframe(page, height=450)
    else:
        import streamlit.components.v1 as components

        components.html(page, height=450)
    note = "The sentence repeats. Drag to turn the person; Close-up zooms onto the face."
    if plan.missing:
        note = ("No sign yet for: " + ", ".join(dict.fromkeys(plan.missing)) + ". " + note)
    st.caption(note)
    return True


def _try_render(
    pipeline: Any, sequence: Any, config: AppConfig, language: str, context: str, text: str
) -> Path | None:
    """Stitch the sequence to one MP4 (cached per text); return path or None."""
    captions = bool(st.session_state.get("captions_on", True))
    out = config.resolve(config.paths.recordings_dir) / f"_preview_{context}_{language}.mp4"
    key = f"{language}|{captions}|{text}"
    if st.session_state.get(f"{context}_render_key") == key and out.is_file():
        return out
    try:
        with st.spinner("Stitching sign clips…"):
            pipeline.render(sequence, out, add_captions=captions)
    except Exception as exc:  # noqa: BLE001 — moviepy/codec/clip issues
        st.info(f"Showing storyboard (video stitching unavailable: {exc}).")
        return None
    st.session_state[f"{context}_render_key"] = key
    return out


def _caption_track(sequence: Any) -> str:
    """One readable caption line; fingerspelled words collapse to ``ALI (A-L-I)``."""
    parts: list[str] = []
    spelled_word: str | None = None
    letters: list[str] = []

    def flush() -> None:
        if spelled_word is not None:
            parts.append(f"{spelled_word} ({'-'.join(letters)})")

    for clip in sequence.clips:
        if clip.is_pause:
            continue
        if clip.is_fallback and " \u00b7 " in clip.caption:
            word, letter = clip.caption.split(" \u00b7 ", 1)
            if word != spelled_word:
                flush()
                spelled_word, letters = word, []
            letters.append(letter)
            continue
        flush()
        spelled_word, letters = None, []
        parts.append(clip.caption)
    flush()
    return " · ".join(parts)


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
