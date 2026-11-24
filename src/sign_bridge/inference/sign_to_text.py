"""Direction 1 — sign → text pipeline.

Ties the pieces together for live recognition:

    frame ─▶ Holistic ─▶ 162-dim features ─▶ time-stamped motion buffer
          every Nth frame ─▶ last ``window_seconds`` resampled to
          ``window_size`` frames ─▶ scale ─▶ model ─▶ smoother
          stable + confident ─▶ WordBuilder ─▶ word / sentence

Design notes (v0.2):

* **Time-based window.** Training clips are ``collector.clip_seconds`` of
  wall-clock motion resampled to ``window_size`` frames. Live recognition
  uses exactly the same recipe (last ``inference.window_seconds`` of frames,
  resampled), so the model sees motion at the same speed in training and in
  use, whatever frame rate the laptop manages.
* **Dropout tolerance.** MediaPipe routinely loses a hand for a frame or
  two. Gaps shorter than ``inference.hand_grace_seconds`` no longer reset the
  smoother, so a sign can still lock during normal, slightly noisy tracking.
* **Lock in frames, not predictions.** ``inference.lock_frames`` is
  converted to a number of agreeing predictions using the prediction
  cadence, so the configured value means what its name says.

The caller (UI or a headless loop) feeds frames via :meth:`process_frame`
and reads the returned :class:`FrameOutcome` to render the current sign,
confidence, word being built, and any finalized sentence.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from sign_bridge.capture.holistic import HolisticResult
from sign_bridge.config import AppConfig
from sign_bridge.data.preprocess import normalize_frame_mode, normalize_window, resample_clip
from sign_bridge.inference.segmenter import Segment, SegmenterSettings, SignSegmenter
from sign_bridge.inference.smoothing import PredictionSmoother
from sign_bridge.inference.word_builder import WordBuilder
from sign_bridge.language.psl_lexicon import URDU_LETTERS
from sign_bridge.logging_conf import get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)

#: Minimum fraction of ``window_seconds`` the buffer must span before predicting.
_MIN_WINDOW_COVERAGE = 0.75
#: Minimum number of buffered frames before predicting (guards very low FPS).
_MIN_WINDOW_FRAMES = 8
#: Frame rate whole-sign segments are re-timed to before the model sees them
#: (training clips were phone videos at ~30 fps; laptops track at 10–30 fps).
SEGMENT_FPS = 30.0


@dataclass(frozen=True)
class FrameOutcome:
    """Per-frame result for the UI to render.

    Attributes:
        current_sign: Smoothed majority label (may be unstable), or None.
        confidence: Confidence of that label in [0, 1].
        is_locked: True on the frame a sign locks into the word builder.
        locked_sign: The sign locked this frame, if any.
        current_word: Letters accumulated so far.
        pending_sentence: In-progress sentence text.
        finalized_sentence: Set on the frame a sentence completes.
        latency_ms: Inference latency for this frame (None if no predict ran).
        top_predictions: Up to three ``(label, probability)`` pairs from the
            most recent model run, best first.
        hands_visible: Whether a hand was tracked in this frame.
        phase: Segment mode only — ``"signing"`` while a sign is being
            recorded, ``"idle"`` otherwise ("" in window mode).
        unsure: Segment mode only — the last finished sign was not
            recognized confidently (``top_predictions`` holds the guesses).
        motion: Segment mode only — current hand speed (shoulder widths/s).
        live_sign: Segment mode only — while a sign is still being made, the
            model's best guess from the movement so far (not written down;
            the final answer comes when the sign ends).
        live_confidence: Probability of ``live_sign``.
    """

    current_sign: str | None
    confidence: float
    is_locked: bool
    locked_sign: str | None
    current_word: str
    pending_sentence: str
    finalized_sentence: str | None
    latency_ms: float | None
    top_predictions: tuple[tuple[str, float], ...] = ()
    hands_visible: bool = False
    phase: str = ""
    unsure: bool = False
    motion: float = 0.0
    live_sign: str | None = None
    live_confidence: float = 0.0


class SignToTextPipeline:
    """Stateful live sign→text recognizer for one language/model.

    Args:
        model: Trained Keras model.
        labels: Index → gloss mapping that matches ``model``.
        scaler: Fitted StandardScaler used at training time.
        config: Application config (inference + landmark sections).
        top_k: How many ranked predictions to expose for the UI.
        feature_mode: How frames are normalized; must match training
            (saved with the model).
        window_seconds: Length of the live motion window; defaults to
            ``inference.window_seconds``. Models trained on another dataset
            (e.g. ASL Signs on Kaggle) store the value that matches their
            clips.
        mode: ``"window"`` or ``"segment"`` (see ``inference.mode``); None
            uses the config, where ``"auto"`` picks segment mode for
            whole-sign models.
        whole_sign_model: The model was trained on one-sign-per-clip data
            (saved with Kaggle-trained models); makes ``auto`` choose
            segment mode.
        vocabulary: Optional subset of ``labels`` to choose from ("my signs").
    """

    def __init__(
        self, model: Any, labels: list[str], scaler: Any, config: AppConfig, top_k: int = 3,
        feature_mode: str = "holistic", window_seconds: float | None = None,
        mode: str | None = None, whole_sign_model: bool = False,
        vocabulary: Iterable[str] | None = None,
    ) -> None:
        self._model = model
        self._labels = labels
        self._scaler = scaler
        self._cfg = config
        self._top_k = max(1, top_k)
        self._feature_mode = feature_mode
        # The model's own input length wins over the config: a model trained
        # with 64-frame windows must be fed 64 frames.
        in_shape = getattr(model, "input_shape", None)
        model_frames = in_shape[1] if isinstance(in_shape, tuple) and len(in_shape) == 3 else None
        self._window_size = int(model_frames or config.landmarks.window_size)
        self._window_seconds = float(window_seconds or config.inference.window_seconds)
        self._grace = config.inference.hand_grace_seconds
        self._predict_every = config.inference.predict_every_n_frames

        # Time-stamped normalized frames; trimmed to the motion window by time.
        self._frames: deque[tuple[float, NDArray]] = deque(maxlen=max(240, 8 * self._window_size))
        lock_predictions = max(2, math.ceil(config.inference.lock_frames / self._predict_every))
        self._smoother = PredictionSmoother(
            lock_frames=lock_predictions,
            confidence_threshold=config.inference.confidence_threshold,
        )
        # Urdu alphabet signs (e.g. ALIF, BAY) spell Urdu words letter by letter.
        letters = {lbl: URDU_LETTERS[lbl] for lbl in labels if lbl in URDU_LETTERS}
        self._builder = WordBuilder(
            cooldown_frames=config.inference.cooldown_frames, letter_map=letters
        )
        self._frame_idx = 0
        self._last_hands_time: float | None = None
        self._hands_lost_handled = True
        self._sentence_open = False
        self._last_top: tuple[tuple[str, float], ...] = ()

        inf = config.inference
        chosen = mode or inf.mode
        if chosen == "auto":
            whole = whole_sign_model or feature_mode.startswith("holistic_lips")
            chosen = "segment" if whole else "window"
        if chosen not in ("window", "segment"):
            raise ValueError(f"unknown inference mode {chosen!r}")
        self._mode = chosen
        self._segmenter = SignSegmenter(SegmenterSettings(
            start_speed=inf.motion_start_speed, stop_speed=inf.motion_stop_speed,
            hold_seconds=inf.hold_seconds, drop_seconds=min(inf.hand_grace_seconds, 0.3),
            min_seconds=inf.min_sign_seconds, max_seconds=inf.max_sign_seconds,
            merge_gap_seconds=inf.merge_gap_seconds,
        ))
        self._seg_label: str | None = None
        self._seg_conf = 0.0
        self._seg_unsure = False
        # Last accepted sign and when it ended, so one sign reported twice in
        # quick succession is not written down as two words.
        self._last_locked: str | None = None
        self._last_locked_end = -math.inf
        self._vocab_mask: NDArray | None = None
        self.set_vocabulary(vocabulary)
        # Guess while a sign is still moving (shown, never written down).
        self._live: tuple[str, float] | None = None
        self._live_at = -math.inf

    # ── main entry ───────────────────────────────────────────────────────
    def process_frame(self, result: HolisticResult, now: float | None = None) -> FrameOutcome:
        """Process one tracked frame and advance all state.

        Args:
            result: Output of :meth:`HolisticTracker.process`.
            now: Timestamp in seconds (``time.monotonic()`` when omitted;
                injectable for tests and offline replays).

        Returns:
            A :class:`FrameOutcome` describing what to display now.
        """
        now = time.monotonic() if now is None else now
        self._frame_idx += 1
        self._builder.tick()
        if self._mode == "segment":
            return self._process_segment(result, now)

        # Hands just came back after a real absence: start a fresh window so
        # the first predictions are not made on mostly-empty frames.
        if result.hands_visible and self._hands_lost_handled:
            self._frames.clear()

        # Every frame joins the motion window — including brief dropouts where
        # a hand is missing (zero-filled), exactly as in the training clips.
        self._frames.append((now, normalize_frame_mode(result.features, self._feature_mode)))
        self._trim(now)

        if not result.hands_visible:
            return self._handle_no_hands(now)

        self._last_hands_time = now
        self._hands_lost_handled = False

        run = self._window_ready(now) and self._frame_idx % self._predict_every == 0
        if not run:
            return self._outcome(hands=True)

        label, confidence, latency_ms = self._predict(now)
        smoothed = self._smoother.update(label, confidence)

        locked_sign: str | None = None
        if smoothed.is_stable and smoothed.label is not None and self._builder.commit_sign(
            smoothed.label
        ):
            locked_sign = smoothed.label
            self._sentence_open = True
            self._smoother.reset()
            log.info("sign_locked", sign=locked_sign, confidence=round(confidence, 3))

        return FrameOutcome(
            current_sign=smoothed.label, confidence=smoothed.confidence,
            is_locked=locked_sign is not None, locked_sign=locked_sign,
            current_word=self._builder.current_word,
            pending_sentence=self._builder.pending_sentence,
            finalized_sentence=None, latency_ms=latency_ms,
            top_predictions=self._last_top, hands_visible=True,
        )

    # ── segment mode ─────────────────────────────────────────────────────
    def _process_segment(self, result: HolisticResult, now: float) -> FrameOutcome:
        """One guess per finished sign (see :mod:`sign_bridge.inference.segmenter`)."""
        seg = self._segmenter.update(now, result.features, result.hands_visible,
                                     (now, result.features))
        if result.hands_visible:
            self._last_hands_time = now

        locked: str | None = None
        latency_ms: float | None = None
        if seg is None:
            latency_ms = self._live_guess(now)
        else:
            self._live = None
            label, conf, margin, latency_ms = self._classify_segment(seg)
            inf = self._cfg.inference
            accepted = conf >= inf.segment_confidence and margin >= inf.segment_margin
            # The same sign again after only a short pause is almost always one
            # sign that got cut in two, not the signer repeating themselves.
            echo = (accepted and label == self._last_locked
                    and seg.start - self._last_locked_end < inf.repeat_window_seconds)
            self._seg_label = label if accepted else None
            self._seg_conf = conf
            self._seg_unsure = not accepted
            if echo:
                log.info("sign_repeat_ignored", sign=label,
                         gap=round(seg.start - self._last_locked_end, 3))
                self._last_locked_end = seg.end
            elif accepted and self._builder.commit_sign(label, allow_repeat=True):
                locked = label
                self._sentence_open = True
                self._last_locked, self._last_locked_end = label, seg.end
                log.info("sign_locked", sign=label, confidence=round(conf, 3), mode="segment")
            elif not accepted:
                log.info("sign_unsure", best=label, confidence=round(conf, 3))

        finalized: str | None = None
        absent = math.inf if self._last_hands_time is None else now - self._last_hands_time
        if (not result.hands_visible and self._sentence_open
                and absent >= self._cfg.inference.sentence_pause_seconds):
            finalized = self._builder.end_sentence()
            self._sentence_open = False
            self._seg_label, self._seg_unsure, self._last_top = None, False, ()
            if finalized:
                log.info("sentence_finalized", text=finalized)

        phase = self._segmenter.phase
        if phase != "signing":
            self._live = None
        return FrameOutcome(
            current_sign=self._seg_label, confidence=self._seg_conf,
            is_locked=locked is not None, locked_sign=locked,
            current_word=self._builder.current_word,
            pending_sentence=self._builder.pending_sentence,
            finalized_sentence=finalized, latency_ms=latency_ms,
            top_predictions=self._last_top, hands_visible=result.hands_visible,
            phase=phase, unsure=self._seg_unsure, motion=self._segmenter.speed,
            live_sign=self._live[0] if self._live else None,
            live_confidence=self._live[1] if self._live else 0.0,
        )

    #: How often (seconds) to guess while a sign is moving, and how much of it
    #: must have been seen first. Each guess is one model run (~30 ms on CPU).
    LIVE_EVERY = 0.4
    LIVE_MIN_SECONDS = 0.4

    def _live_guess(self, now: float) -> float | None:
        """Guess the sign from the movement so far; returns the latency if it ran."""
        cur = self._segmenter.current
        if cur is None or cur.duration < self.LIVE_MIN_SECONDS:
            return None
        if not math.isfinite(self.LIVE_EVERY) or now - self._live_at < self.LIVE_EVERY:
            return None                     # (an infinite interval turns live guesses off)
        self._live_at = now
        label, conf, _margin, latency_ms = self._classify_segment(cur)
        self._live = (label, conf)
        return latency_ms

    def segment_input(self, seg: Segment) -> NDArray:
        """Model input ``(window_size, F)`` for a finished segment (before scaling).

        The segment is first re-timed to :data:`SEGMENT_FPS` (so a sign looks
        equally fast whatever the laptop's tracking rate), then given to the
        same ``resample_clip`` → normalization recipe as the training clips.
        """
        times = np.array([t for t, _ in seg.frames], dtype=np.float64)
        raw = np.stack([f for _, f in seg.frames]).astype(np.float32)
        n = max(3, int(round((times[-1] - times[0]) * SEGMENT_FPS)) + 1)
        grid = np.linspace(times[0], times[-1], n)
        idx = np.clip(np.searchsorted(times, grid, side="left"), 0, len(times) - 1)
        prev = np.clip(idx - 1, 0, len(times) - 1)
        nearest = np.where(np.abs(times[prev] - grid) < np.abs(times[idx] - grid), prev, idx)
        clip = resample_clip(raw[nearest], self._window_size)
        return normalize_window(clip, self._feature_mode)

    def _classify_segment(self, seg: Segment) -> tuple[str, float, float, float]:
        """``(label, probability, margin over runner-up, latency ms)``."""
        window = self.segment_input(seg)[None, ...]
        flat = self._scaler.transform(window.reshape(-1, window.shape[-1]))
        window = flat.reshape(window.shape).astype(np.float32)
        t0 = time.perf_counter()
        probs = self._restrict(np.asarray(self._model.predict_on_batch(window))[0])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        order = np.argsort(probs)[::-1][: max(2, self._top_k)]
        self._last_top = tuple((self._labels[int(i)], float(probs[int(i)]))
                               for i in order[: self._top_k])
        best, second = float(probs[order[0]]), float(probs[order[1]]) if len(order) > 1 else 0.0
        return self._labels[int(order[0])], best, best - second, latency_ms

    # ── vocabulary ("my signs") ──────────────────────────────────────────
    def set_vocabulary(self, vocabulary: Iterable[str] | None) -> None:
        """Only choose among these labels (None or empty = the whole model)."""
        wanted = {v.strip().upper() for v in (vocabulary or []) if v.strip()}
        if not wanted:
            self._vocab_mask = None
            return
        mask = np.array([lbl.upper() in wanted for lbl in self._labels])
        self._vocab_mask = mask if mask.any() else None

    @property
    def vocabulary(self) -> list[str]:
        """Labels currently allowed (all labels when no subset is set)."""
        if self._vocab_mask is None:
            return list(self._labels)
        return [lbl for lbl, keep in zip(self._labels, self._vocab_mask, strict=True) if keep]

    def _restrict(self, probs: NDArray) -> NDArray:
        """Renormalize probabilities over the allowed vocabulary."""
        if self._vocab_mask is None:
            return probs
        kept = np.where(self._vocab_mask, probs, 0.0)
        total = float(kept.sum())
        return kept / total if total > 0 else kept

    @property
    def mode(self) -> str:
        """``"window"`` or ``"segment"``."""
        return self._mode

    # ── helpers ──────────────────────────────────────────────────────────
    def _handle_no_hands(self, now: float) -> FrameOutcome:
        absent = math.inf if self._last_hands_time is None else now - self._last_hands_time

        # A short dropout is tracking noise: keep the smoother and the word.
        if absent < self._grace:
            return self._outcome(hands=False)

        if not self._hands_lost_handled:
            self._smoother.reset()
            self._builder.hands_lost()
            self._last_top = ()
            self._hands_lost_handled = True

        finalized = None
        if self._sentence_open and absent >= self._cfg.inference.sentence_pause_seconds:
            finalized = self._builder.end_sentence()
            self._sentence_open = False
            if finalized:
                log.info("sentence_finalized", text=finalized)
        return FrameOutcome(
            current_sign=None, confidence=0.0, is_locked=False, locked_sign=None,
            current_word=self._builder.current_word,
            pending_sentence=self._builder.pending_sentence,
            finalized_sentence=finalized, latency_ms=None,
        )

    def _outcome(self, hands: bool) -> FrameOutcome:
        """Outcome for a frame where no new prediction ran."""
        label, conf = self._smoother.latest
        return FrameOutcome(
            current_sign=label, confidence=conf, is_locked=False, locked_sign=None,
            current_word=self._builder.current_word,
            pending_sentence=self._builder.pending_sentence,
            finalized_sentence=None, latency_ms=None,
            top_predictions=self._last_top, hands_visible=hands,
        )

    def _trim(self, now: float) -> None:
        """Drop frames older than the motion window."""
        horizon = now - self._window_seconds
        while self._frames and self._frames[0][0] < horizon:
            self._frames.popleft()

    def _window_ready(self, now: float) -> bool:
        if len(self._frames) < _MIN_WINDOW_FRAMES:
            return False
        span = now - self._frames[0][0]
        return span >= _MIN_WINDOW_COVERAGE * self._window_seconds

    def window(self) -> NDArray:
        """The current model input ``(window_size, F)`` (before scaling)."""
        clip = np.stack([f for _, f in self._frames])
        return resample_clip(clip, self._window_size)

    def _predict(self, now: float) -> tuple[str, float, float]:
        window = self.window()[None, ...]  # (1, T, F)
        flat = self._scaler.transform(window.reshape(-1, window.shape[-1]))
        window = flat.reshape(window.shape).astype(np.float32)
        t0 = time.perf_counter()
        # predict_on_batch skips Keras' per-call dataset setup: ~3 ms instead
        # of ~60 ms for model.predict on a single window.
        probs = self._restrict(np.asarray(self._model.predict_on_batch(window))[0])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        order = np.argsort(probs)[::-1][: self._top_k]
        self._last_top = tuple((self._labels[int(i)], float(probs[int(i)])) for i in order)
        idx = int(order[0])
        return self._labels[idx], float(probs[idx]), latency_ms

    # ── session control ──────────────────────────────────────────────────
    def force_end_sentence(self) -> str | None:
        """Manually finalize the current sentence (e.g. a UI button)."""
        self._sentence_open = False
        return self._builder.end_sentence()

    @property
    def builder(self) -> WordBuilder:
        """Access the underlying word builder (transcript, sentences)."""
        return self._builder

    @property
    def feature_mode(self) -> str:
        """Feature mode this recognizer normalizes frames with."""
        return self._feature_mode

    @property
    def labels(self) -> list[str]:
        """Vocabulary of the loaded model, index order."""
        return list(self._labels)

    @property
    def model_count(self) -> int:
        """How many trained models answer together (1 unless an ensemble is loaded)."""
        return len(self._model) if hasattr(self._model, "__len__") else 1

    def reset(self) -> None:
        """Reset all state for a fresh session."""
        self._frames.clear()
        self._smoother.reset()
        self._builder.reset()
        self._frame_idx = 0
        self._last_hands_time = None
        self._hands_lost_handled = True
        self._sentence_open = False
        self._last_top = ()
        self._segmenter.reset()
        self._seg_label, self._seg_conf, self._seg_unsure = None, 0.0, False
        self._last_locked, self._last_locked_end = None, -math.inf
        self._live, self._live_at = None, -math.inf
