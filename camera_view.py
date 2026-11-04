"""Camera view — the Sign → Speech tab body.

Runs the live recognition loop and renders the Field Guide readout. Because
Streamlit re-runs top-to-bottom rather than hosting a persistent render
loop, the camera is driven in short frame-stepping bursts: each script run
grabs a batch of frames, advances the pipeline, paints the latest annotated
frame, and schedules the next run while "running" is on. Heavy ML imports
(TensorFlow, MediaPipe) are deferred to first Start so the app launches even
before any model is trained.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import streamlit as st

from sign_bridge.config import AppConfig
from sign_bridge.ui.components import readout

# Frames advanced per script run before yielding back to Streamlit.
_FRAMES_PER_BURST = 6


@st.cache_resource(show_spinner="Loading recognizer…")
def _load_pipeline(language: str, _cfg_id: int) -> tuple[Any, str | None]:
    """Load model + build the sign→text pipeline, cached per language.

    Returns ``(pipeline, error)``: on failure ``pipeline`` is None and
    ``error`` carries a user-facing message.
    """
    try:
        from sign_bridge.config import load_config
        from sign_bridge.inference.sign_to_text import SignToTextPipeline
        from sign_bridge.models.evaluator import load_latest_model

        cfg = load_config()
        model, labels, scaler = load_latest_model(language)
        return SignToTextPipeline(model, labels, scaler, cfg), None
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — surface any load error to UI
        return None, f"Could not load the recognizer: {exc}"


@st.cache_resource(show_spinner=False)
def _open_camera(index: int, width: int, height: int, fps: int, mirror: bool) -> Any:
    """Open (and cache) a threaded camera, or return None on failure."""
    try:
        from sign_bridge.capture.webcam import ThreadedCamera

        return ThreadedCamera(index, width, height, fps, mirror).start()
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource(show_spinner=False)
def _open_tracker(complexity: int, det: float, trk: float) -> Any:
    """Open (and cache) a MediaPipe Holistic tracker, or None on failure."""
    try:
        from sign_bridge.capture.holistic import HolisticTracker

        return HolisticTracker(complexity, det, trk)
    except Exception:  # noqa: BLE001
        return None


def _speak(sentence: str) -> None:
    """Synthesize and play (gtts) or speak on host (pyttsx3)."""
    from sign_bridge.tts import TextToSpeech

    engine = st.session_state.get("tts_engine", "gtts")
    tts = TextToSpeech(engine=engine)
    if engine == "gtts":
        audio = tts.to_mp3_bytes(sentence)
        if audio:
            st.audio(audio, format="audio/mp3", autoplay=True)
    else:
        tts.speak_blocking(sentence)


def render(config: AppConfig, context: str = "sign") -> None:
    """Render the Sign → Speech tab.

    Args:
        config: Loaded application config.
        context: Unique prefix for widget keys, so this component can be
            rendered on more than one tab (e.g. Conversation) without
            Streamlit duplicate-element-id collisions.
    """
    language = st.session_state["language"]
    left, right = st.columns([1.35, 1], gap="large")

    # ── controls row ─────────────────────────────────────────────────────
    with right:
        c1, c2, c3 = st.columns(3)
        if c1.button("Start" if not st.session_state["running"] else "Pause",
                     use_container_width=True, key=f"{context}_start"):
            st.session_state["running"] = not st.session_state["running"]
        if c2.button("End sentence", use_container_width=True, key=f"{context}_end"):
            st.session_state["_force_end"] = True
        if c3.button("Speak last", use_container_width=True,
                     key=f"{context}_speak") and st.session_state["transcript"]:
            _speak(st.session_state["transcript"][-1][1])

    pipeline, err = _load_pipeline(language, id(config))
    frame_slot = left.empty()
    cap_slot = left.empty()

    if err:
        frame_slot.info(
            "No trained model for this language yet.\n\n"
            f"{err}\n\nThe interface is fully styled and ready — train a model "
            "to enable live recognition.")
        with right:
            readout.focal_sign(None, False)
            readout.transcript(st.session_state["transcript"], context)
            readout.status_line([("Model — not loaded", False),
                                 (f"Mode — {language.upper()}", True)])
        return

    camera = _open_camera(config.capture.camera_index, config.capture.frame_width,
                          config.capture.frame_height, config.capture.target_fps,
                          config.capture.mirror)
    tracker = _open_tracker(config.landmarks.model_complexity,
                           config.landmarks.min_detection_confidence,
                           config.landmarks.min_tracking_confidence)

    if camera is None or tracker is None:
        frame_slot.warning(
            "Camera or landmark tracker unavailable in this environment. "
            "Run SignBridge on a machine with a webcam to use live mode.")
        with right:
            readout.focal_sign(None, False)
            readout.transcript(st.session_state["transcript"], context)
            readout.status_line([("Camera — unavailable", False),
                                 ("Model — loaded", True)])
        return

    # Handle a queued manual sentence-end.
    if st.session_state.pop("_force_end", False):
        sentence = pipeline.force_end_sentence()
        if sentence:
            _record_sentence(sentence)

    outcome = None
    fps = 0.0
    if st.session_state["running"]:
        from sign_bridge.ui.overlay import draw_landmarks, draw_status_bar

        for _ in range(_FRAMES_PER_BURST):
            try:
                frame = camera.read()
            except Exception:  # noqa: BLE001 — camera dropped mid-loop
                frame_slot.error("Camera disconnected. Press Start to retry.")
                st.session_state["running"] = False
                break
            result = tracker.process(frame)
            outcome = pipeline.process_frame(result)
            if outcome.finalized_sentence:
                _record_sentence(outcome.finalized_sentence)

        fps = camera.fps
        annotated = draw_landmarks(frame, result.raw, st.session_state["high_contrast"])
        annotated = draw_status_bar(annotated, fps,
                                    outcome.latency_ms if outcome else None,
                                    st.session_state["high_contrast"])
        import cv2

        frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
    else:
        frame_slot.markdown(
            '<div style="aspect-ratio:4/3;border:1px solid var(--ink);'
            'display:grid;place-items:center;color:var(--soft);'
            'font-family:Fraunces,serif;font-style:italic">'
            'Camera paused — press Start</div>', unsafe_allow_html=True)

    live = "Recording" if st.session_state["running"] else "Paused"
    cap_slot.markdown(
        f'<div class="fg-figcap"><span>Fig. 1 — Live capture, 21 landmarks tracked'
        f'</span><span class="live">{live} · {fps:4.1f} fps</span></div>',
        unsafe_allow_html=True)

    # ── readout column ───────────────────────────────────────────────────
    with right:
        readout.focal_sign(outcome.current_sign if outcome else None,
                           bool(outcome and outcome.is_locked))
        readout.confidence_gauge(outcome.confidence if outcome else 0.0)
        readout.word_in_progress(outcome.current_word if outcome else "",
                                outcome.pending_sentence if outcome else "")
        st.divider()
        readout.transcript(st.session_state["transcript"], context)
        readout.status_line([
            ("Model — loaded", True),
            (f"Mode — {language.upper()}", True),
            (f"{fps:4.1f} fps", st.session_state["running"]),
        ])

    # Schedule the next burst while running.
    if st.session_state["running"]:
        time.sleep(0.03)
        st.rerun()


def _record_sentence(sentence: str) -> None:
    """Append a finalized sentence to the transcript and optionally speak it."""
    ts = datetime.now().strftime("%H:%M")
    st.session_state["transcript"].append((ts, sentence))
    if st.session_state.get("auto_speak"):
        _speak(sentence)
