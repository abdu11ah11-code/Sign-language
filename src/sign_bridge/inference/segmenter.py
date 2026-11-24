"""Motion-based sign segmentation for live recognition.

Isolated-sign models (such as those trained on Google's ASL Signs data) are
trained on clips that each hold exactly **one whole sign**. Feeding them a
sliding window of the live stream forces a guess every moment — including
while the hands rest or travel between signs — which shows up as a stream
of random words.

:class:`SignSegmenter` instead watches how fast the hands move and cuts the
stream into sign-sized pieces:

    idle ──(hands appear / start moving)──▶ signing
    signing ──(hands still for ``hold_seconds``, gone for ``drop_seconds``,
               or ``max_seconds`` reached)──▶ one finished segment ──▶ idle

The caller then runs the model **once per finished segment**, on the whole
segment resampled to the model's window — the same recipe the training clips
were made with.

Speed is measured in *shoulder widths per second* on the raw MediaPipe
landmarks, so it does not depend on how far the signer sits from the camera.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from sign_bridge.capture.holistic import FEATURE_DIM
from sign_bridge.typing import NDArray

#: Positions of the left/right shoulder inside the 12-point pose block.
_L_SHOULDER, _R_SHOULDER = 4, 5
#: Fallback shoulder width (normalized image units) when the pose is missing.
_DEFAULT_SHOULDER_WIDTH = 0.25
#: Speed reported for a hand that has just appeared (counts as movement).
_APPEAR_SPEED = 10.0


@dataclass(frozen=True)
class SegmenterSettings:
    """Thresholds for :class:`SignSegmenter` (see ``inference`` in config.yaml).

    Attributes:
        start_speed: Hand speed (shoulder widths / s) that starts a sign.
        stop_speed: Below this the hands count as still.
        hold_seconds: Still this long while signing → the sign is over.
        drop_seconds: Hands gone this long while signing → the sign is over.
        min_seconds: Shorter segments are discarded as twitches.
        max_seconds: A segment is closed at this length regardless.
        pre_roll_seconds: Frames kept from just before the start was detected.
        merge_gap_seconds: A new burst of movement starting within this long
            of the previous one is treated as the same sign continuing, not a
            new one. Signers often pause briefly in the middle of a sign, and
            without this the sign is reported twice. 0 turns merging off; the
            cost is that a finished sign is only reported after this delay.
        smoothing: EMA factor for the speed signal (1 = no smoothing).
    """

    start_speed: float = 0.6
    stop_speed: float = 0.25
    hold_seconds: float = 0.35
    drop_seconds: float = 0.25
    min_seconds: float = 0.3
    max_seconds: float = 3.0
    pre_roll_seconds: float = 0.15
    merge_gap_seconds: float = 0.3
    smoothing: float = 0.5


@dataclass(frozen=True)
class Segment:
    """One finished sign: the payloads recorded for it, oldest first."""

    frames: list[Any]
    start: float
    end: float

    @property
    def duration(self) -> float:
        """Length of the segment in seconds."""
        return self.end - self.start


def _hands_xy(raw: NDArray) -> tuple[NDArray | None, NDArray | None]:
    """Left/right hand points ``(21, 2)``, or None for an undetected hand."""
    left = raw[0:63].reshape(21, 3)[:, :2]
    right = raw[63:126].reshape(21, 3)[:, :2]
    return (left if np.any(left) else None), (right if np.any(right) else None)


def shoulder_width(raw: NDArray) -> float:
    """Shoulder width in normalized image units (fallback when not tracked)."""
    pose = raw[126:FEATURE_DIM].reshape(12, 3)[:, :2]
    ls, rs = pose[_L_SHOULDER], pose[_R_SHOULDER]
    if not (np.any(ls) and np.any(rs)):
        return _DEFAULT_SHOULDER_WIDTH
    width = float(np.linalg.norm(ls - rs))
    return width if width > 1e-3 else _DEFAULT_SHOULDER_WIDTH


def hand_speed(prev: NDArray | None, cur: NDArray, dt: float) -> float:
    """Fastest hand speed between two raw frames, in shoulder widths / s.

    A hand that was missing in ``prev`` and is visible now counts as fast
    movement (raising the hands into view starts a sign).
    """
    if dt <= 0:
        return 0.0
    cur_hands = _hands_xy(cur)
    if prev is None:
        return _APPEAR_SPEED if any(h is not None for h in cur_hands) else 0.0
    prev_hands = _hands_xy(prev)
    scale = shoulder_width(cur)
    best = 0.0
    for p, c in zip(prev_hands, cur_hands, strict=True):
        if c is None:
            continue
        if p is None:
            best = max(best, _APPEAR_SPEED)
            continue
        disp = float(np.mean(np.linalg.norm(c - p, axis=1)))
        best = max(best, disp / scale / dt)
    return best


class SignSegmenter:
    """Cuts a live landmark stream into one segment per sign.

    Feed every frame to :meth:`update`; it returns a :class:`Segment` on the
    frame a sign finishes, otherwise None. :attr:`phase` says whether a sign
    is currently being watched.
    """

    def __init__(self, settings: SegmenterSettings | None = None) -> None:
        self.s = settings or SegmenterSettings()
        self._pre: deque[tuple[float, Any]] = deque()
        self._frames: list[tuple[float, Any]] = []
        self._prev_raw: NDArray | None = None
        self._prev_t: float | None = None
        self._speed = 0.0
        self._signing = False
        self._last_active = 0.0
        self._last_hands = 0.0
        # A finished sign waits here for ``merge_gap_seconds`` in case the
        # signer was only pausing and the same sign continues.
        self._held: list[tuple[float, Any]] = []
        self._held_at = 0.0

    # ── public state ─────────────────────────────────────────────────────
    @property
    def phase(self) -> str:
        """``"signing"`` while a sign is being recorded, else ``"idle"``."""
        return "signing" if (self._signing or self._held) else "idle"

    @property
    def current(self) -> Segment | None:
        """The sign being watched so far (None when idle), for a guess while signing."""
        frames = self._held + self._frames if self._signing else []
        if len(frames) < 3:
            return None
        return Segment(frames=[p for _, p in frames], start=frames[0][0], end=frames[-1][0])

    @property
    def speed(self) -> float:
        """Smoothed hand speed (shoulder widths / s) of the latest frame."""
        return self._speed

    def reset(self) -> None:
        """Forget everything (new session)."""
        self.__init__(self.s)  # type: ignore[misc]

    # ── main entry ───────────────────────────────────────────────────────
    def update(self, t: float, raw: NDArray, hands_visible: bool, payload: Any,
               speed: float | None = None) -> Segment | None:
        """Advance by one frame.

        Args:
            t: Frame time in seconds.
            raw: Raw landmark vector (the tracker's ``features``).
            hands_visible: Whether a hand was detected this frame.
            payload: What to store for this frame (e.g. the normalized
                feature vector the model needs).
            speed: This frame's raw hand speed, when it has already been
                measured (see :func:`hand_speed`). Offline tools that try many
                settings on the same recording compute it once and pass it in;
                live callers leave it None and it is measured here. It does not
                depend on the settings, so the result is identical either way.

        Returns:
            The finished :class:`Segment` on the frame a sign ends, else None.
        """
        s = self.s
        dt = 0.0 if self._prev_t is None else t - self._prev_t
        if hands_visible:
            if speed is not None:
                raw_speed = speed
            elif dt > 0:
                raw_speed = hand_speed(self._prev_raw if self._prev_t is not None else None,
                                       raw, dt)
            else:
                raw_speed = 0.0
            self._speed = s.smoothing * raw_speed + (1 - s.smoothing) * self._speed
            self._last_hands = t
        else:
            self._speed = 0.0
        self._prev_raw = raw if hands_visible else None
        self._prev_t = t

        if not self._signing:
            self._pre.append((t, payload))
            while self._pre and self._pre[0][0] < t - s.pre_roll_seconds:
                self._pre.popleft()
            if self._held and t - self._held_at > s.merge_gap_seconds:
                # the pause was too long: that sign really is finished. A new one
                # can start on the next frame.
                return self._release()
            if hands_visible and self._speed >= s.start_speed:
                self._signing = True
                if self._held:
                    # the signer only paused — carry on with the same sign
                    held_until = self._held[-1][0]
                    self._frames = self._held + [f for f in self._pre if f[0] > held_until]
                    self._held = []
                else:
                    self._frames = list(self._pre)
                self._pre.clear()
                self._last_active = t
            return None

        self._frames.append((t, payload))
        if hands_visible and self._speed >= s.stop_speed:
            self._last_active = t

        start = self._frames[0][0]
        still_too_long = hands_visible and t - self._last_active >= s.hold_seconds
        gone_too_long = (not hands_visible) and t - self._last_hands >= s.drop_seconds
        too_long = t - start >= s.max_seconds
        if not (still_too_long or gone_too_long or too_long):
            return None

        frames = self._frames
        if gone_too_long:  # drop the hand-less tail
            frames = [f for f in frames if f[0] <= self._last_hands]
        self._signing = False
        self._frames = []
        self._pre.clear()
        if not frames:
            return None
        self._held, self._held_at = frames, t
        # A sign that ended because the hands left, or because it hit the length
        # limit, is finished — there is nothing to merge it with.
        if gone_too_long or too_long or s.merge_gap_seconds <= 0:
            return self._release()
        return None

    def _release(self) -> Segment | None:
        """Emit the sign waiting in ``_held``, if it is long enough to be real."""
        frames, self._held = self._held, []
        if not frames:
            return None
        seg = Segment(frames=[p for _, p in frames], start=frames[0][0], end=frames[-1][0])
        if seg.duration < self.s.min_seconds or len(seg.frames) < 3:
            return None
        return seg
