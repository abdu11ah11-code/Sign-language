"""MediaPipe Holistic wrapper producing fixed-size landmark feature vectors.

Raw frame layout (282 floats), kept stable across the codebase:

    [  0:63 ]  left hand   — 21 landmarks × (x, y, z)
    [ 63:126]  right hand  — 21 landmarks × (x, y, z)
    [126:162]  upper pose  — 12 landmarks × (x, y, z), see POSE_LANDMARK_INDICES
    [162:282]  lips        — 40 face-mesh landmarks × (x, y, z), see LIP_LANDMARK_INDICES

The first 162 values are the original layout (``FEATURE_DIM``); the
``holistic`` and ``hand_shape`` feature modes read only those, so data and
models made before lips were added keep working. Clips recorded before the
lips block existed have 162 columns and are padded with zeros on load.

Missing detections (a hand out of frame) are zero-filled, never dropped,
so the temporal window keeps a constant shape for the model.

Coordinates are MediaPipe-normalized image coordinates (x, y in [0, 1],
z roughly in [-1, 1] relative to the hips/wrist). Geometric normalization
(centering on the chest, scaling by shoulder width) happens later in
:mod:`sign_bridge.data.preprocess` so raw captures stay lossless on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from sign_bridge.logging_conf import get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)

#: Pose landmarks kept (MediaPipe Pose indices): nose, eyes, mouth-left,
#: shoulders, elbows, wrists, hips. Enough to encode where the hands are
#: relative to the torso/face — the signal that distinguishes e.g.
#: THANK-YOU (chin) from PLEASE (chest) — without dragging in legs.
POSE_LANDMARK_INDICES: Final[tuple[int, ...]] = (0, 2, 5, 9, 11, 12, 13, 14, 15, 16, 23, 24)

#: Lip landmarks (MediaPipe Face Mesh indices): outer and inner lip contours,
#: the 40-point set used by the top solutions of Google's Isolated Sign
#: Language Recognition competition. Mouth shapes carry meaning in ASL.
LIP_LANDMARK_INDICES: Final[tuple[int, ...]] = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
)

#: Left/right mirror counterparts among the lip landmarks (face-mesh indices);
#: points on the midline (0, 17, 13, 14) map to themselves.
LIP_MIRROR_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (61, 291), (185, 409), (40, 270), (39, 269), (37, 267),
    (146, 375), (91, 321), (181, 405), (84, 314),
    (78, 308), (191, 415), (80, 310), (81, 311), (82, 312),
    (95, 324), (88, 318), (178, 402), (87, 317),
)

HAND_POINTS: Final[int] = 21
FEATURE_DIM: Final[int] = (2 * HAND_POINTS + len(POSE_LANDMARK_INDICES)) * 3  # = 162
LIP_POINTS: Final[int] = len(LIP_LANDMARK_INDICES)  # = 40
RAW_DIM: Final[int] = FEATURE_DIM + LIP_POINTS * 3  # = 282, full raw frame

# Offsets into the 162-dim vector (start, stop), used by preprocess/augment.
LEFT_HAND_SLICE: Final[slice] = slice(0, 63)
RIGHT_HAND_SLICE: Final[slice] = slice(63, 126)
POSE_SLICE: Final[slice] = slice(126, 162)
LIPS_SLICE: Final[slice] = slice(162, RAW_DIM)

# Where the shoulders sit inside the pose block (indices 11 and 12 are at
# positions 4 and 5 of POSE_LANDMARK_INDICES; each point is 3 floats).
LEFT_SHOULDER_OFFSET: Final[int] = 126 + 4 * 3
RIGHT_SHOULDER_OFFSET: Final[int] = 126 + 5 * 3


@dataclass(frozen=True)
class HolisticResult:
    """One processed frame.

    Attributes:
        features: ``(282,)`` float32 raw vector in the layout documented above.
        hands_visible: True when at least one hand was detected.
        left_hand_visible: True when the left hand was detected.
        right_hand_visible: True when the right hand was detected.
        raw: The raw MediaPipe results object (for overlay drawing).
    """

    features: NDArray
    hands_visible: bool
    left_hand_visible: bool
    right_hand_visible: bool
    raw: Any


def _landmarks_to_array(landmark_list: Any, n_points: int) -> NDArray:
    """Convert a MediaPipe landmark list to ``(n_points*3,)`` or zeros."""
    if landmark_list is None:
        return np.zeros(n_points * 3, dtype=np.float32)
    pts = landmark_list.landmark
    out = np.empty(n_points * 3, dtype=np.float32)
    for i in range(n_points):
        p = pts[i]
        out[i * 3 : i * 3 + 3] = (p.x, p.y, p.z)
    return out


def extract_features(results: Any) -> NDArray:
    """Build the 282-dim raw feature vector from Holistic results.

    Args:
        results: Object returned by ``mp.solutions.holistic.Holistic.process``.

    Returns:
        ``(282,)`` float32 array; missing parts are zero-filled.
    """
    left = _landmarks_to_array(getattr(results, "left_hand_landmarks", None), HAND_POINTS)
    right = _landmarks_to_array(getattr(results, "right_hand_landmarks", None), HAND_POINTS)

    pose_lms = getattr(results, "pose_landmarks", None)
    pose = np.zeros(len(POSE_LANDMARK_INDICES) * 3, dtype=np.float32)
    if pose_lms is not None:
        for j, idx in enumerate(POSE_LANDMARK_INDICES):
            p = pose_lms.landmark[idx]
            pose[j * 3 : j * 3 + 3] = (p.x, p.y, p.z)

    face_lms = getattr(results, "face_landmarks", None)
    lips = np.zeros(LIP_POINTS * 3, dtype=np.float32)
    if face_lms is not None:
        for j, idx in enumerate(LIP_LANDMARK_INDICES):
            p = face_lms.landmark[idx]
            lips[j * 3 : j * 3 + 3] = (p.x, p.y, p.z)

    return np.concatenate([left, right, pose, lips])


class HolisticTracker:
    """Thin lifecycle wrapper around MediaPipe Holistic.

    Use as a context manager so native resources are always released::

        with HolisticTracker() as tracker:
            result = tracker.process(frame_bgr)

    Args:
        model_complexity: MediaPipe complexity 0/1/2 (speed vs accuracy).
        min_detection_confidence: Detector confidence floor.
        min_tracking_confidence: Tracker confidence floor.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp  # lazy: feature constants import without MediaPipe

        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "holistic"):
            raise RuntimeError(
                f"mediapipe {getattr(mp, '__version__', '?')} has no 'solutions.holistic' API. "
                "SignBridge needs the Holistic solution: pip install mediapipe==0.10.14"
            )
        self._holistic = solutions.holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_face_landmarks=False,  # lips come from the standard 468-point mesh
        )
        log.info("holistic_initialized", model_complexity=model_complexity)

    def process(self, frame_bgr: NDArray) -> HolisticResult:
        """Run Holistic on one BGR frame (OpenCV convention).

        Args:
            frame_bgr: ``(H, W, 3)`` uint8 BGR image.

        Returns:
            :class:`HolisticResult` with features and visibility flags.
        """
        import cv2  # local import keeps module importable without OpenCV (tests)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # MediaPipe perf hint
        results = self._holistic.process(rgb)

        left_ok = results.left_hand_landmarks is not None
        right_ok = results.right_hand_landmarks is not None
        return HolisticResult(
            features=extract_features(results),
            hands_visible=left_ok or right_ok,
            left_hand_visible=left_ok,
            right_hand_visible=right_ok,
            raw=results,
        )

    def close(self) -> None:
        """Release MediaPipe native resources."""
        self._holistic.close()

    def __enter__(self) -> HolisticTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
