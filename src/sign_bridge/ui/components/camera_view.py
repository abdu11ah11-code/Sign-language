"""Camera view — live Sign → Speech recognition.

How the live loop works with Streamlit (v0.2):

1. Every script run, :func:`render` lays out a *view* (empty placeholders
   for the frame, current sign, candidates, word, transcript, status) on
   each tab that shows the camera, and paints its idle state.
2. After **all** tabs are laid out, ``app.py`` calls :func:`run_live_loop`.
   While "running" is on, it loops continuously: grab frame → MediaPipe →
   recognizer → repaint the placeholders in place.
3. Any click anywhere interrupts the loop (Streamlit stops the script and
   reruns it), the click is handled, and the loop resumes.

Compared with the previous burst + ``st.rerun()`` design this removes the
full-page flicker, and the Speech → Sign and Conversation tabs stay usable
while the camera runs (``st.rerun()`` used to stop the script before those
tabs were ever drawn).
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from sign_bridge.config import AppConfig
from sign_bridge.ui.components import readout

#: Test hook: stop the live loop after N frames (0 = run until paused).
_MAX_FRAMES_ENV = "SIGN_BRIDGE_MAX_LOOP_FRAMES"

# The open camera, so Pause can release the device (and the webcam light).
_ACTIVE: dict[str, Any] = {"camera": None}


# ── cached heavy resources ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recognizer…")
def _load_pipeline(language: str, _cfg_id: int) -> tuple[Any, str | None]:
    """Load model + build the sign→text pipeline, cached per language.

    Returns ``(pipeline, error)``: on failure ``pipeline`` is None and
    ``error`` carries a user-facing message.
    """
    try:
        from sign_bridge.config import load_config
        from sign_bridge.inference.sign_to_text import SignToTextPipeline
        from sign_bridge.models.evaluator import load_latest_model, load_model_meta

        cfg = load_config()
        model, labels, scaler = load_latest_model(language)
        meta = load_model_meta(language)
        return SignToTextPipeline(
            model, labels, scaler, cfg,
            feature_mode=str(meta.get("feature_mode", "holistic")),
            window_seconds=meta.get("window_seconds"),
            whole_sign_model=meta.get("clip_mode") == "whole_sign",
            vocabulary=load_my_signs(cfg, language),
        ), None
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — surface any load error to UI
        return None, f"Could not load the recognizer: {exc}"


@st.cache_resource(show_spinner="Opening camera…")
def _open_camera(index: int, width: int, height: int, fps: int, mirror: bool) -> Any:
    """Open (and cache) a threaded camera, or return None on failure."""
    try:
        from sign_bridge.capture.webcam import ThreadedCamera

        return ThreadedCamera(index, width, height, fps, mirror).start()
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource(show_spinner="Starting hand tracker…")
def _open_tracker(complexity: int, det: float, trk: float) -> tuple[Any, str | None]:
    """Open (and cache) a MediaPipe Holistic tracker → ``(tracker, error)``."""
    try:
        from sign_bridge.capture.holistic import HolisticTracker

        return HolisticTracker(complexity, det, trk), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _release_camera() -> None:
    """Release the webcam so its light turns off while paused."""
    cam = _ACTIVE.get("camera")
    if cam is not None:
        try:
            cam.release()
        finally:
            _ACTIVE["camera"] = None
            _open_camera.clear()


# ── "my signs" (vocabulary subset) ───────────────────────────────────────
def _my_signs_path(config: AppConfig, language: str) -> Path:
    return config.resolve(config.paths.data_dir) / f"{language}_my_signs.json"


def load_my_signs(config: AppConfig, language: str) -> list[str]:
    """The saved "my signs" list for a language ([] = use every sign)."""
    try:
        data = json.loads(_my_signs_path(config, language).read_text(encoding="utf-8"))
        return [str(x) for x in data] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_my_signs(config: AppConfig, language: str, signs: list[str]) -> None:
    """Persist the "my signs" list (best effort — the UI works without it)."""
    path = _my_signs_path(config, language)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(signs), indent=2), encoding="utf-8")
    except OSError:
        pass


def _on_my_signs_change(key: str, language: str, cfg_id: int, config: AppConfig) -> None:
    st.session_state[f"{key}_open"] = True        # keep the box open while choosing
    signs = list(st.session_state.get(key) or [])
    save_my_signs(config, language, signs)
    pipeline, _ = _load_pipeline(language, cfg_id)
    if pipeline is not None:
        pipeline.set_vocabulary(signs)


def _my_signs_picker(pipeline: Any, config: AppConfig, language: str, context: str) -> None:
    """Expander to choose the signs the recognizer may answer with."""
    from sign_bridge.language.psl_lexicon import display_name

    labels = list(pipeline.labels)
    chosen = [lbl for lbl in pipeline.vocabulary if lbl in labels] \
        if len(pipeline.vocabulary) < len(labels) else []
    key = f"{context}_my_signs_{language}"
    # The label must not change while signs are added or removed: Streamlit treats an
    # expander with a new label as a new box and draws it closed. So the count goes
    # inside, and once you start choosing the box stays open until you close it.
    with st.expander("My signs", expanded=bool(st.session_state.get(f"{key}_open"))):
        st.caption(f"Recognizing {len(chosen)} of {len(labels)} signs" if chosen
                   else f"Recognizing all {len(labels)} signs")
        st.multiselect(
            "Only recognize these signs", labels, default=chosen, key=key,
            format_func=display_name, placeholder="All signs (leave empty)",
            on_change=_on_my_signs_change, args=(key, language, id(config), config),
            help="Choose the signs you know or are practising. Fewer signs = "
                 "far fewer mix-ups. Leave empty to use every sign the model knows.")
        if language == "asl":
            from sign_bridge.language.asl_lexicon import lookup_url

            show = chosen or labels[:12]
            links = " · ".join(f"[{display_name(lbl)}]({lookup_url(lbl)})" for lbl in show[:40])
            st.caption(("How to sign them: " if chosen else "Look up a sign, e.g.: ") + links)


# ── "how to sign it" animations ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _sign_library(language: str, cfg_id: int) -> Any:
    """The animation library for a language (empty when none is installed)."""
    del cfg_id
    from sign_bridge.config import load_config
    from sign_bridge.ui.animation import DEFAULT_LIBRARY, SignLibrary

    cfg = load_config()
    return SignLibrary.load(cfg.resolve(Path(DEFAULT_LIBRARY.format(language=language))))


def _how_to_sign(slot: Any, config: AppConfig, language: str, context: str,
                 vocabulary: list[str]) -> None:
    """Draw the "How to sign" box: pick a sign and watch it signed, in 3D when possible."""
    from sign_bridge.language.psl_lexicon import display_name
    from sign_bridge.ui.animation import sign_svg
    from sign_bridge.ui.figure3d import sign_3d_html

    library = _sign_library(language, id(config))
    if not len(library):
        return
    choices = [g for g in vocabulary if g in library] or library.glosses()
    with slot.container():
        st.markdown('<div class="fg-label" style="margin-top:1rem">How to sign</div>',
                    unsafe_allow_html=True)
        if library.has_depth:
            top, mid, right, view = st.columns([2, 1, 1, 1])
            solid = view.toggle("3D", value=True, key=f"{context}_howto_3d",
                                help="A solid figure you can turn with the mouse. "
                                     "Off: the flat drawing.")
        else:
            top, mid, right = st.columns([2, 1, 1])
            solid = False
        gloss = top.selectbox("Sign", choices, key=f"{context}_howto_sign",
                              format_func=display_name, label_visibility="collapsed")
        pace = mid.selectbox("Speed", ["Normal", "Slow", "Step"], key=f"{context}_howto_speed",
                             label_visibility="collapsed")
        mirror = right.toggle("Mirror", value=True, key=f"{context}_howto_mirror",
                              help="Flip left/right so you can copy it like a mirror.")
        clip = library.get(str(gloss))
        if clip is None:
            return
        frame = None
        if pace == "Step":
            frame = st.slider("Frame", 1, len(clip.frames), 1,
                              key=f"{context}_howto_frame") - 1
        speed = 0.5 if pace == "Slow" else 1.0
        title = display_name(str(gloss))
        if solid and clip.has_depth:
            # The page is built by SignBridge from its own library file (the sign name
            # is escaped), so it is trusted HTML. st.iframe replaces the deprecated
            # components.v1.html; older Streamlit releases only have the latter.
            page = sign_3d_html(clip, speed=speed, mirror=mirror, frame=frame,
                                height=420, title=title)
            if hasattr(st, "iframe"):
                st.iframe(page, height=430)
            else:
                import streamlit.components.v1 as components

                components.html(page, height=430)
            from sign_bridge.ui.figure3d import find_touch

            touch = find_touch(clip)
            how = ("Red spot: where the " + touch.finger.lower() + " touches. " if touch else "")
            how += ("Drag to turn it — Side shows how far the hands are from the body; "
                    "Close-up zooms onto the head.")
        else:
            svg = sign_svg(clip, speed=speed, mirror=mirror, frame=frame, title=title)
            st.markdown(f'<div class="sa-box">{svg}</div>', unsafe_allow_html=True)
            how = "Green = hands, gold = lips."
        st.caption("Movement from a Deaf signer in the ASL Signs dataset — signs vary "
                   f"between people and regions. {how}")


# ── speech ───────────────────────────────────────────────────────────────
def _speak(sentence: str, slot: Any) -> None:
    """Play gTTS audio in the browser, or speak on the host (pyttsx3)."""
    from sign_bridge.tts import TextToSpeech

    engine = st.session_state.get("tts_engine", "gtts")
    tts = TextToSpeech(engine=engine)
    if engine == "gtts":
        audio = tts.to_mp3_bytes(sentence)
        if audio:
            slot.audio(audio, format="audio/mp3", autoplay=True)
        else:
            slot.caption("Speech unavailable (gTTS needs internet) — try the offline voice.")
    else:
        tts.speak_blocking(sentence)


def _record_sentence(sentence: str, tokens: list[str] | None = None,
                     wait: bool = False) -> str | None:
    """Append a finalized signed sentence to the transcript; start the AI writer.

    Returns the text to speak now, or None when the AI is still writing (the
    sentence is then spoken by :func:`_poll_ai` once it is ready).
    """
    ts = datetime.now().strftime("%H:%M")
    entries = st.session_state["transcript"]
    entries.append((ts, sentence, "signer"))
    words = list(tokens or sentence.split())
    writer = _ai_writer()
    if writer is None or not writer.available:
        from sign_bridge.language.sentence_ai import plain_sentence

        st.session_state["ai_last"] = {
            "signs": words, "sentences": [plain_sentence(words)],
            "error": None if writer is None else "add an API key in the sidebar"}
        return sentence
    if wait:  # button callback: the camera is not running, just wait for it
        result = writer.compose(words)
        _apply_ai(len(entries) - 1, result)
        return str(result.best)
    future = _AI_POOL.submit(writer.compose, words)
    st.session_state["_ai_jobs"].append({"idx": len(entries) - 1, "future": future,
                                         "speak": bool(st.session_state.get("auto_speak"))})
    st.session_state["ai_last"] = {"signs": words, "pending": True}
    return None


# ── AI sentences ─────────────────────────────────────────────────────────
_AI_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sign-ai")


def _ai_writer() -> Any:
    """The session's :class:`SentenceAI`, or None when AI sentences are off."""
    if not st.session_state.get("ai_enabled"):
        return None
    from sign_bridge.config import load_config
    from sign_bridge.language.sentence_ai import SentenceAI, find_api_key

    cfg = load_config().ai
    key = find_api_key(cfg.provider, load_config().resolve(Path(cfg.secrets_file)))
    sig = (cfg.provider, key, cfg.model, st.session_state.get("language"))
    cached = st.session_state.get("_ai_writer")
    if cached is not None and cached[0] == sig:
        return cached[1]
    writer = SentenceAI(provider=cfg.provider, api_key=key, model=cfg.model,
                        max_options=cfg.max_options, timeout=cfg.timeout_seconds,
                        language=str(st.session_state.get("language", "asl")).upper())
    st.session_state["_ai_writer"] = (sig, writer)
    return writer


def _apply_ai(idx: int, result: Any) -> None:
    """Show an AI result and put its sentence into the transcript."""
    entries = st.session_state["transcript"]
    if 0 <= idx < len(entries):
        ts, _, who = entries[idx]
        entries[idx] = (ts, result.best, who)
    st.session_state["ai_last"] = {"signs": list(result.signs),
                                   "sentences": list(result.sentences),
                                   "error": result.error}


def _poll_ai(audio_slot: Any, block: bool = False) -> None:
    """Collect finished AI sentences (speaking them if auto-speak was on)."""
    jobs = st.session_state.get("_ai_jobs") or []
    remaining = []
    for job in jobs:
        fut = job["future"]
        if not (fut.done() or block):
            remaining.append(job)
            continue
        result = fut.result()
        _apply_ai(job["idx"], result)
        if job["speak"] and result.best:
            _speak(result.best, audio_slot)
    st.session_state["_ai_jobs"] = remaining


# ── button callbacks (run before the script, so state is consistent) ─────
def _toggle_running() -> None:
    st.session_state["running"] = not st.session_state["running"]
    if not st.session_state["running"]:
        _release_camera()


def _end_sentence(language: str, cfg_id: int) -> None:
    pipeline, _ = _load_pipeline(language, cfg_id)
    if pipeline is not None:
        sentence = pipeline.force_end_sentence()
        if sentence:
            text = _record_sentence(sentence, pipeline.builder.last_tokens, wait=True)
            if text and st.session_state.get("auto_speak"):
                st.session_state["_pending_speech"] = text


def _speak_last() -> None:
    sentence = readout.last_signed(st.session_state["transcript"])
    if sentence:
        st.session_state["_pending_speech"] = sentence


# ── layout ───────────────────────────────────────────────────────────────
@dataclass
class LiveView:
    """Placeholders for one on-page copy of the live camera readout."""

    pipeline: Any
    frame: Any
    caption: Any
    sign: Any
    gauge: Any
    candidates: Any
    word: Any
    ai: Any
    transcript: Any
    transcript_title: str
    status: Any
    audio: Any
    _last: dict[str, str] = field(default_factory=dict)

    def put(self, name: str, html_text: str) -> None:
        """Repaint a placeholder only when its content changed."""
        if self._last.get(name) != html_text:
            self._last[name] = html_text
            getattr(self, name).markdown(html_text, unsafe_allow_html=True)


def render(
    config: AppConfig,
    context: str = "sign",
    transcript_slot: Any = None,
    transcript_title: str = "Transcript",
) -> LiveView | None:
    """Lay out the live camera view and paint its idle state.

    Args:
        config: Loaded application config.
        context: Unique prefix for widget keys (the view appears on more than
            one tab).
        transcript_slot: Optional placeholder to use for the transcript
            (the Conversation tab passes its shared log); by default the
            transcript sits under the readout.
        transcript_title: Heading for the transcript block.

    Returns:
        A :class:`LiveView` for :func:`run_live_loop`, or None when no model
        is available for the selected language.
    """
    language = st.session_state["language"]
    running = st.session_state["running"]
    left, right = st.columns([1.35, 1], gap="large")

    with right:
        c1, c2, c3 = st.columns(3)
        c1.button("Pause" if running else "Start", width="stretch",
                  key=f"{context}_start", on_click=_toggle_running, type="primary")
        c2.button("Finish", width="stretch", key=f"{context}_end",
                  on_click=_end_sentence, args=(language, id(config)),
                  help="End the current sentence now (instead of waiting for a pause).")
        c3.button("Speak", width="stretch", key=f"{context}_speak",
                  on_click=_speak_last, help="Speak the last signed sentence aloud.")

    pipeline, err = _load_pipeline(language, id(config))
    entries = st.session_state["transcript"]
    if err or pipeline is None:
        left.info(
            "No trained model for this language yet.\n\n"
            f"{err}\n\nThe interface is ready — record signs with `sign-collect` "
            "and train with `sign-train` to enable live recognition.")
        with right:
            readout.focal_sign(None, False)
            readout.transcript(entries, context)
            readout.status_line([("Model — not loaded", False),
                                 (f"Mode — {language.upper()}", True)])
        return None

    with left:
        frame_slot = st.empty()
        caption_slot = st.empty()
        howto_slot = st.empty()
    with right:
        if context == "tab_sign":
            _my_signs_picker(pipeline, config, language, context)
        sign_slot, gauge_slot, cand_slot, word_slot, ai_slot = (st.empty() for _ in range(5))
        st.divider()
        own_transcript = st.empty() if transcript_slot is None else transcript_slot
        status_slot = st.empty()
        audio_slot = st.empty()

    view = LiveView(
        pipeline=pipeline, frame=frame_slot, caption=caption_slot, sign=sign_slot,
        gauge=gauge_slot, candidates=cand_slot, word=word_slot, ai=ai_slot,
        transcript=own_transcript, transcript_title=transcript_title,
        status=status_slot, audio=audio_slot,
    )

    if context == "tab_sign":
        _how_to_sign(howto_slot, config, language, context, pipeline.vocabulary)

    if not running:
        frame_slot.markdown(
            '<div style="aspect-ratio:4/3;border:1px solid var(--ink);'
            'display:grid;place-items:center;color:var(--soft);'
            'font-family:Fraunces,serif;font-style:italic">'
            'Camera paused — press Start</div>', unsafe_allow_html=True)
    _paint_readout(view, None, config, fps=0.0, running=running)
    return view


def _paint_readout(
    view: LiveView, outcome: Any, config: AppConfig, fps: float, running: bool,
    latency_ms: float | None = None,
) -> None:
    """Repaint every text placeholder of a view from the latest outcome."""
    builder = view.pipeline.builder
    segment = view.pipeline.mode == "segment"
    language = st.session_state["language"]
    if segment:
        view.put("sign", readout.segment_sign_html(
            outcome.current_sign if outcome else None,
            bool(outcome and outcome.unsure),
            outcome.phase if outcome else "idle",
            bool(outcome and outcome.hands_visible),
            language,
            outcome.live_sign if outcome else None,
            outcome.live_confidence if outcome else 0.0,
        ))
        unsure = bool(outcome and outcome.unsure)
        view.put("candidates", readout.top_predictions_html(
            outcome.top_predictions if outcome else (), config.inference.segment_confidence,
            title="Did you mean…" if unsure else "Top candidates"))
    else:
        view.put("sign", readout.focal_sign_html(
            outcome.current_sign if outcome else None,
            bool(outcome and outcome.is_locked),
            bool(outcome and outcome.hands_visible),
        ))
        view.put("candidates", readout.top_predictions_html(
            outcome.top_predictions if outcome else (), config.inference.confidence_threshold))
    view.put("gauge", readout.confidence_gauge_html(outcome.confidence if outcome else 0.0))
    view.put("word", readout.word_html(builder.current_word, builder.pending_sentence))
    view.put("ai", readout.ai_bar_html(st.session_state.get("ai_last"),
                                       bool(st.session_state.get("ai_enabled"))))
    view.put("transcript", readout.transcript_html(
        st.session_state["transcript"], view.transcript_title))
    live = "Recording" if running else "Paused"
    view.put("caption",
             '<div class="fg-figcap"><span>Fig. 1 — Live capture, hands + upper body '
             f'tracked</span><span class="live">{live} · {fps:4.1f} fps</span></div>')
    mode = {"hand_shape": "hand shape", "holistic_lips": "hands + body + lips",
            "holistic_lips_clip": "hands + body + lips"}.get(
        view.pipeline.feature_mode, "hands + body")
    count = getattr(view.pipeline, "model_count", 1)
    together = f", {count} models together" if count > 1 else ""
    items = [(f"Model — {len(view.pipeline.labels)} signs, {mode}{together}", True),
             (f"Mode — {language.upper()}", True)]
    n_vocab = len(view.pipeline.vocabulary)
    if n_vocab < len(view.pipeline.labels):
        items.append((f"My signs — {n_vocab}", True))
    if segment:
        signing = bool(outcome and outcome.phase == "signing")
        items.append(("one guess per sign", True))
        items.append((f"motion {outcome.motion if outcome else 0.0:.1f}", signing))
    items.append((f"{fps:4.1f} fps", running))
    if latency_ms is not None:
        items.append((f"inference {latency_ms:.0f} ms", True))
    view.put("status", readout.status_html(items))


# ── the live loop ────────────────────────────────────────────────────────
def run_live_loop(config: AppConfig, views: list[LiveView]) -> None:
    """Drive the camera continuously, repainting every view in place.

    Must be called after all tabs are laid out (see module docstring).
    Returns when paused, when the camera fails, or when Streamlit
    interrupts the script because the user clicked something.
    """
    if not views:
        return
    primary = views[0]

    pending = st.session_state.pop("_pending_speech", None)
    if pending:
        _speak(pending, primary.audio)

    if not st.session_state["running"]:
        if st.session_state.get("_ai_jobs"):   # paused mid-sentence: finish the AI part
            _poll_ai(primary.audio, block=True)
            for v in views:
                _paint_readout(v, None, config, fps=0.0, running=False)
        return

    camera = _open_camera(config.capture.camera_index, config.capture.frame_width,
                          config.capture.frame_height, config.capture.target_fps,
                          config.capture.mirror)
    tracker, tracker_err = _open_tracker(config.landmarks.model_complexity,
                                         config.landmarks.min_detection_confidence,
                                         config.landmarks.min_tracking_confidence)
    if camera is None or tracker is None:
        reason = ("No webcam found (or another app is using it)." if camera is None
                  else f"Hand tracker failed to start: {tracker_err}")
        for v in views:
            v.frame.warning(f"{reason} Live mode needs a working webcam and MediaPipe.")
        st.session_state["running"] = False
        _release_camera()
        return
    _ACTIVE["camera"] = camera

    import cv2

    from sign_bridge.capture.webcam import CameraError
    from sign_bridge.ui.overlay import draw_landmarks, draw_status_bar

    pipeline = primary.pipeline
    max_frames = int(os.environ.get(_MAX_FRAMES_ENV, "0") or 0)
    high_contrast = bool(st.session_state["high_contrast"])
    show_fps = config.ui.show_fps
    fps = 0.0
    last_t = time.perf_counter()
    latency: float | None = None
    n = 0

    while st.session_state.get("running"):
        try:
            frame = camera.read()
        except CameraError as exc:
            for v in views:
                v.frame.error(f"Camera disconnected ({exc}). Press Start to retry.")
            st.session_state["running"] = False
            _release_camera()
            break

        result = tracker.process(frame)
        outcome = pipeline.process_frame(result)
        if outcome.latency_ms is not None:
            latency = outcome.latency_ms

        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 0:  # processing FPS (what the recognizer actually gets)
            fps = 1.0 / dt if fps == 0.0 else 0.9 * fps + 0.1 / dt

        if outcome.finalized_sentence:
            text = _record_sentence(outcome.finalized_sentence, pipeline.builder.last_tokens)
            if text and st.session_state.get("auto_speak"):
                _speak(text, primary.audio)
        if st.session_state.get("_ai_jobs"):
            _poll_ai(primary.audio)

        annotated = draw_landmarks(frame, result.raw, high_contrast)
        if show_fps:
            annotated = draw_status_bar(annotated, fps, latency, high_contrast)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        for v in views:
            v.frame.image(rgb, width="stretch", output_format="JPEG")
            _paint_readout(v, outcome, config, fps, running=True, latency_ms=latency)

        n += 1
        if max_frames and n >= max_frames:
            break
