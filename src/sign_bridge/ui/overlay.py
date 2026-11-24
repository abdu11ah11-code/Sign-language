"""Landmark overlay rendering — Field Guide palette on the live frame.

Hands and upper-body pose are drawn as a fine ink-green skeleton with
paper-white joints, matching design direction #4 ("The Field Guide"):
connections in deep green #1E5A45, points in paper #F7F5F0 with ink
outlines. High-contrast mode thickens strokes and switches to pure
black/yellow for low-vision users.
"""

from __future__ import annotations

from typing import Any

import cv2

from sign_bridge.typing import NDArray

# BGR equivalents of the Field Guide palette.
_GREEN = (69, 90, 30)      # #1E5A45
_PAPER = (240, 245, 247)   # #F7F5F0
_INK = (17, 22, 24)        # #181611
_HC_LINE = (0, 230, 255)   # high-contrast yellow
_HC_DOT = (0, 0, 0)


def _mp_modules() -> tuple[Any, Any]:
    """Return ``(drawing_utils, holistic)`` from MediaPipe, imported lazily.

    Deferring the import keeps this module (and the collector that uses it)
    importable in environments where MediaPipe is absent — the call only
    happens when something actually draws an overlay.
    """
    import mediapipe as mp

    return mp.solutions.drawing_utils, mp.solutions.holistic


def draw_landmarks(
    frame_bgr: NDArray,
    results: Any,
    high_contrast: bool = False,
) -> NDArray:
    """Draw hand + upper-pose skeletons onto a copy of the frame.

    Args:
        frame_bgr: ``(H, W, 3)`` uint8 BGR frame.
        results: Raw MediaPipe Holistic results (``HolisticResult.raw``).
        high_contrast: Thicker, black-on-yellow rendering.

    Returns:
        A new frame with the overlay drawn (input is not mutated).
    """
    out = frame_bgr.copy()
    if results is None:
        return out

    line = _HC_LINE if high_contrast else _GREEN
    dot = _HC_DOT if high_contrast else _PAPER
    thick = 3 if high_contrast else 2
    radius = 4 if high_contrast else 3

    mp_draw, mp_holistic = _mp_modules()
    line_spec = mp_draw.DrawingSpec(color=line, thickness=thick)
    dot_spec = mp_draw.DrawingSpec(color=dot, thickness=1, circle_radius=radius)

    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand is not None:
            mp_draw.draw_landmarks(
                out, hand, mp_holistic.HAND_CONNECTIONS, dot_spec, line_spec
            )

    if results.pose_landmarks is not None:
        mp_draw.draw_landmarks(
            out,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=dot, thickness=1, circle_radius=2),
            mp_draw.DrawingSpec(color=line, thickness=1),
        )
    return out


def draw_status_bar(
    frame_bgr: NDArray,
    fps: float,
    latency_ms: float | None = None,
    high_contrast: bool = False,
) -> NDArray:
    """Stamp a small FPS / latency readout in the frame's top-right corner."""
    out = frame_bgr.copy()
    text = f"{fps:4.1f} fps" + (f"  {latency_ms:.0f} ms" if latency_ms is not None else "")
    color = _HC_DOT if high_contrast else _INK
    bg = _HC_LINE if high_contrast else _PAPER
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = out.shape[1] - tw - 18
    cv2.rectangle(out, (x - 8, 8), (out.shape[1] - 8, 18 + th + 8), bg, -1)
    cv2.putText(out, text, (x, 18 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
