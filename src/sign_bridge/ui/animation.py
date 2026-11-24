"""Looping figure animations that show how a sign is made.

A small library file (built by ``notebooks/make_sign_animations.ipynb``) holds
one representative recording per sign, scaled into a 0–100 drawing box (x right,
y down, shoulders 26 units apart). Two layouts exist:

* **version 1** — 94 points, ``x, y`` only: both hands, 12 upper-body points
  and 40 lip points. The original stick figure.
* **version 2** — the same hands and body plus a **face**: its outline, eyes,
  eyebrows, nose and lips (:data:`FACE_INDICES`, taken from MediaPipe's
  468-point face mesh), every point with ``x, y, z``. The face matters in ASL:
  raised eyebrows mark a yes/no question, a frown a *wh-* question. The depth
  lets :mod:`sign_bridge.ui.figure3d` draw a solid figure that can be turned.

:func:`sign_svg` draws either layout as an animated SVG (no depth needed);
version 2 also feeds the 3D view.

Library format (JSON, optionally gzipped)::

    {"version": 2, "language": "asl", "fps": 12, "face_indices": [...],
     "signs": {"MOM": {"frames": [[x0, y0, z0, x1, y1, z1, ...], ...]}}}

Points per frame, in order: left hand (21), right hand (21), pose (12), then
lips (40, version 1) or the face points (version 2). Untracked points are 0.
"""

from __future__ import annotations

import gzip
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Version 1: points per frame and the blocks they fall into.
POINTS = 94
LEFT_HAND = range(0, 21)
RIGHT_HAND = range(21, 42)
POSE = range(42, 54)
LIPS = range(54, 94)

#: Pose points, by position inside the 12-point block.
_NOSE, _EYE_L, _EYE_R = 0, 1, 2
_SH_L, _SH_R, _EL_L, _EL_R, _WR_L, _WR_R, _HIP_L, _HIP_R = 4, 5, 6, 7, 8, 9, 10, 11

#: Chains of hand points that are drawn as one line (MediaPipe hand topology).
_HAND_CHAINS = ([0, 5, 9, 13, 17, 0], [0, 1, 2, 3, 4], [5, 6, 7, 8],
                [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20])
#: Body chains, by position inside the pose block.
_BODY_CHAINS = ([_SH_L, _EL_L, _WR_L], [_SH_R, _EL_R, _WR_R], [_SH_L, _SH_R],
                [_SH_L, _HIP_L, _HIP_R, _SH_R])
#: Version 1 lip outlines: outer loop, then inner loop (capture.holistic lip order).
_LIP_CHAINS = ([*range(0, 11), *range(19, 10, -1), 0], [*range(20, 30), *range(39, 29, -1), 20])

#: Face features as chains of MediaPipe face-mesh indices (from MediaPipe's own
#: ``face_mesh_connections``; the person's left eye is on the image's right).
FACE_MESH_CHAINS: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = (
    ("oval", ((10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176,
               148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284,
               332, 297, 338, 10),)),
    ("left_eye", ((249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
                   263, 249),)),
    ("right_eye", ((7, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144,
                    163, 7),)),
    ("left_brow", ((276, 283, 282, 295, 285), (336, 296, 334, 293, 300))),
    ("right_brow", ((46, 53, 52, 65, 55), (107, 66, 105, 63, 70))),
    ("nose", ((168, 6, 197, 195, 5, 4, 45, 220, 115, 48, 64, 98, 97, 2, 94, 19, 1, 4, 275,
               440, 344, 278, 294, 327, 326, 2),)),
    ("lips", ((0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
               270, 269, 267, 0),
              (13, 82, 81, 80, 191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
               310, 311, 312, 13))),
)
#: The face points a version-2 library stores, in this order.
FACE_INDICES: tuple[int, ...] = tuple(sorted(
    {i for _, chains in FACE_MESH_CHAINS for chain in chains for i in chain}))
#: Points per frame in a version-2 library.
POINTS_V2 = 54 + len(FACE_INDICES)

#: Default library location, relative to the project root.
DEFAULT_LIBRARY = "assets/sign_animations/{language}.json.gz"


@dataclass(frozen=True)
class Layout:
    """Where everything sits inside one frame of a library.

    Attributes:
        version: Library format version.
        points: Points per frame.
        dims: Numbers per point (2 = ``x, y``; 3 = ``x, y, z``).
        face: ``(feature, chains)`` pairs; chains hold absolute point indices.
    """

    version: int
    points: int
    dims: int
    face: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]

    @property
    def width(self) -> int:
        """Numbers per frame."""
        return self.points * self.dims

    def face_points(self) -> range:
        """The block holding lips (version 1) or the whole face (version 2)."""
        return range(54, self.points)


def _v2_face() -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
    pos = {mesh: 54 + i for i, mesh in enumerate(FACE_INDICES)}
    return tuple((name, tuple(tuple(pos[i] for i in chain) for chain in chains))
                 for name, chains in FACE_MESH_CHAINS)


LAYOUT_V1 = Layout(1, POINTS, 2, (("lips", tuple(tuple(54 + i for i in c) for c in _LIP_CHAINS)),))
LAYOUT_V2 = Layout(2, POINTS_V2, 3, _v2_face())


@dataclass(frozen=True)
class SignClip:
    """One sign's animation.

    Attributes:
        gloss: Sign label, upper case.
        frames: Per frame, ``layout.points`` points of ``layout.dims`` numbers.
        fps: Frames per second the clip was sampled at.
        layout: Which library layout the frames follow.
    """

    gloss: str
    frames: list[list[float]]
    fps: float = 12.0
    layout: Layout = LAYOUT_V1

    @property
    def seconds(self) -> float:
        """Natural length of the clip."""
        return len(self.frames) / self.fps if self.fps else 0.0

    @property
    def has_depth(self) -> bool:
        """Whether the clip can be drawn in 3D (version 2)."""
        return self.layout.dims == 3


class SignLibrary:
    """The animations available for one language.

    Args:
        clips: Mapping of upper-case gloss → :class:`SignClip`.
        source: Where the library was loaded from (shown in the UI).
    """

    def __init__(self, clips: dict[str, SignClip], source: str = "") -> None:
        self._clips = clips
        self.source = source

    def __len__(self) -> int:
        return len(self._clips)

    def __contains__(self, gloss: object) -> bool:
        return isinstance(gloss, str) and gloss.strip().upper() in self._clips

    def get(self, gloss: str) -> SignClip | None:
        """The clip for ``gloss``, or None when the library has no example."""
        return self._clips.get(gloss.strip().upper())

    def glosses(self) -> list[str]:
        """Every sign with an animation, alphabetically."""
        return sorted(self._clips)

    @property
    def has_depth(self) -> bool:
        """Whether this library's clips can be drawn in 3D."""
        return any(c.has_depth for c in self._clips.values())

    @classmethod
    def load(cls, path: Path) -> SignLibrary:
        """Read a library file (``.json`` or ``.json.gz``); empty if unreadable.

        A version-2 file must list the same ``face_indices`` this code draws
        with, so a file from a different SignBridge release is never drawn
        with the wrong points.
        """
        try:
            opener: Any = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            version = int(data.get("version", 1))
            if version >= 2:
                if tuple(data.get("face_indices") or ()) != FACE_INDICES:
                    return cls({}, source="")
                layout = LAYOUT_V2
            else:
                layout = LAYOUT_V1
            fps = float(data.get("fps", 12.0))
            clips = {}
            for gloss, entry in (data.get("signs") or {}).items():
                frames = entry.get("frames") if isinstance(entry, dict) else entry
                if (isinstance(frames, list) and len(frames) >= 2
                        and all(len(f) == layout.width for f in frames)):
                    key = str(gloss).strip().upper()
                    clips[key] = SignClip(key, [[float(v) for v in f] for f in frames], fps,
                                          layout)
            return cls(clips, source=str(path))
        except (OSError, ValueError, TypeError, AttributeError):
            return cls({}, source="")


# ── SVG drawing ──────────────────────────────────────────────────────────
def _xy(frame: list[float], idx: int, dims: int = 2) -> tuple[float, float]:
    return frame[dims * idx], frame[dims * idx + 1]


def _visible(frames: list[list[float]], block: range | tuple[int, ...], dims: int = 2) -> bool:
    """Whether a block of points was tracked at all (not left at 0)."""
    return any(any(frame[dims * i] or frame[dims * i + 1] for i in block) for frame in frames)


def _chain_values(frames: list[list[float]], offset: int, chain: list[int] | tuple[int, ...],
                  dims: int = 2) -> str:
    """SMIL ``values`` for one polyline: one frame's points per list item."""
    out = []
    for frame in frames:
        pts = []
        for i in chain:
            x, y = _xy(frame, offset + i, dims)
            pts.append(f"{x:.1f},{y:.1f}")
        out.append(" ".join(pts))
    return ";".join(out)


def _animated_polyline(values: str, dur: str, width: float, cls: str) -> str:
    first = values.split(";", 1)[0]
    return (f'<polyline class="{cls}" points="{first}" fill="none" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round">'
            f'<animate attributeName="points" values="{values}" dur="{dur}" '
            f'repeatCount="indefinite"/></polyline>')


def _head(frames: list[list[float]], dur: str, dims: int = 2) -> str:
    """Head circle, sized from the distance between the eyes."""
    cx, cy, r = [], [], []
    for frame in frames:
        (lx, ly) = _xy(frame, POSE.start + _EYE_L, dims)
        (rx, ry) = _xy(frame, POSE.start + _EYE_R, dims)
        nx, ny = _xy(frame, POSE.start + _NOSE, dims)
        if not (lx or ly or rx or ry):
            lx, ly, rx, ry = nx - 3, ny, nx + 3, ny
        eye = max(((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5, 1.0)
        cx.append(f"{(lx + rx) / 2:.1f}")
        cy.append(f"{(ly + ry) / 2 + eye * 0.25:.1f}")
        r.append(f"{eye * 1.9:.1f}")
    anim = "".join(
        f'<animate attributeName="{name}" values="{";".join(vals)}" '
        f'dur="{dur}" repeatCount="indefinite"/>'
        for name, vals in (("cx", cx), ("cy", cy), ("r", r)))
    return (f'<circle class="sa-head" cx="{cx[0]}" cy="{cy[0]}" r="{r[0]}" fill="none" '
            f'stroke-width="1.1">{anim}</circle>')


#: Stroke width and CSS class per face feature.
_FACE_STYLE = {"oval": (1.1, "sa-head"), "left_eye": (0.6, "sa-head"),
               "right_eye": (0.6, "sa-head"), "left_brow": (0.9, "sa-head"),
               "right_brow": (0.9, "sa-head"), "nose": (0.5, "sa-head"),
               "lips": (0.8, "sa-lips")}


def sign_svg(
    clip: SignClip, speed: float = 1.0, mirror: bool = True, frame: int | None = None,
    height: int = 260, title: str = "",
) -> str:
    """An animated SVG of one sign.

    Args:
        clip: The sign to draw.
        speed: Playback speed (0.5 = half speed, easier to copy).
        mirror: Flip left↔right, so the learner can copy it like a mirror.
        frame: Show this single frame instead of animating (for stepping).
        height: Drawing height in pixels (the box is square).
        title: Accessible name for the drawing.

    Returns:
        SVG markup, ready for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    layout, dims = clip.layout, clip.layout.dims
    frames = clip.frames if frame is None else [clip.frames[frame % len(clip.frames)]] * 2
    frames = [*frames, frames[0]]            # repeat the first frame for a smooth loop
    dur = f"{max(clip.seconds / max(speed, 0.1), 0.3):.2f}s"
    parts = []
    face_outline = dict(layout.face).get("oval")
    if not (face_outline and _visible(frames, face_outline[0], dims)):
        parts.append(_head(frames, dur, dims))
    for chain in _BODY_CHAINS:
        parts.append(_animated_polyline(
            _chain_values(frames, POSE.start, chain, dims), dur, 1.6, "sa-body"))
    for block in (LEFT_HAND, RIGHT_HAND):
        if not _visible(frames, block, dims):
            continue
        for chain in _HAND_CHAINS:
            parts.append(_animated_polyline(
                _chain_values(frames, block.start, chain, dims), dur, 1.3, "sa-hand"))
    for name, chains in layout.face:
        width, cls = _FACE_STYLE.get(name, (0.6, "sa-head"))
        for points in chains:
            if _visible(frames, points, dims):
                parts.append(_animated_polyline(
                    _chain_values(frames, 0, points, dims), dur, width, cls))
    body = "".join(parts)
    if mirror:
        body = f'<g transform="translate(100,0) scale(-1,1)">{body}</g>'
    label = html.escape(title or clip.gloss, quote=True)
    return (f'<svg class="sa-svg" viewBox="0 0 100 100" width="100%" height="{height}" '
            f'role="img" aria-label="Animated figure showing how to sign {label}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')
