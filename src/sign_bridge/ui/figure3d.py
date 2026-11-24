"""A solid 3D figure for the "How to sign" box, drawn in the browser.

Version-2 sign libraries (see :mod:`sign_bridge.ui.animation`) hold every point
with depth, so a sign can be shown as a figure the learner **turns with the
mouse** — the clearest way to see whether a hand is in front of the face or
beside it, which decides signs such as DAD and DONKEY.

The figure is a **3D person** (``static/person.glb``, made in Blender from
MakeHuman CC0 data, see ``avatar/README.txt``) whose skeleton is posed from the
recording every frame: arms by two-bone IK, hands and each finger joint by the
recorded directions, the head and jaw from the face. Near the face, hand
positions are measured against the face, so a thumb on the forehead lands on
the person's forehead. Where a finger touches the face (:func:`find_touch`), a
red spot marks the place and the fingertip is brought onto it.

:func:`sign_3d_html` returns one HTML page with three.js (MIT licence,
``vendor/``) inside; the person's model is the one file it fetches, from the
app's own server (Streamlit static serving). Without the model the older
mannequin is drawn; without WebGL, the flat SVG drawing.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from sign_bridge.typing import NDArray
from sign_bridge.ui.animation import FACE_INDICES, SignClip, sign_svg

_VENDOR = Path(__file__).with_name("vendor") / "three.min.js"
_GLTF_LOADER = Path(__file__).with_name("vendor") / "GLTFLoader.js"
_AVATAR = Path(__file__).with_name("avatar")
_PAGE = Path(__file__).with_name("figure3d_page.html")
_PERSON_PAGE = Path(__file__).with_name("figure3d_person.html")
#: The 3D person, served by Streamlit's static file serving (``server.enableStaticServing``).
PERSON_MODEL = Path(__file__).with_name("static") / "person.glb"
PERSON_URL = "/app/static/person.glb"
_TOKEN = re.compile(r"__(HEIGHT|LABEL|FLAT|DATA|THREE|GLTF)__")

#: Face-mesh points that orient the head: forehead top, chin, the face's two sides.
_TOP, _CHIN, _SIDE_A, _SIDE_B = 10, 152, 234, 454
#: The middle of the upper and lower inner lip: how far the mouth is open.
_LIP_UP, _LIP_DOWN = 13, 14
#: Face-mesh points for the face's expression (MediaPipe numbering), person's left then right:
#: middle of each eyebrow over the top of the upper lid; the upper and lower lid with the eye's
#: two corners (how open the eye is); the two inner brow ends (a frown pulls them together).
_BROW = (334, 386, 105, 159)
_LIDS = (386, 374, 263, 362, 159, 145, 33, 133)
_KNIT = (285, 55)
#: How far a sign's steady head tilt may go (radians, about 10 degrees). The library's
#: recordings lean the head back or forward by up to 40 degrees (median -3), mostly from the
#: camera's height and the tracker's depth guess; past this a teacher looks odd, so the
#: steady part is eased in while nods within the sign are kept.
_TILT_LIMIT = 0.175


@lru_cache(maxsize=1)
def _three_js() -> str:
    return _VENDOR.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def person_model() -> dict[str, object] | None:
    """The 3D person (``static/person.glb``) and where its face points are.

    ``face`` lists, for each of :data:`FACE_INDICES`, the point on the person's
    face at rest (x, y, z in metres, glTF axes) and the bone it moves with
    (``head`` or ``jaw``); ``orient`` gives the positions in that list of the
    forehead top, chin and the face's two sides. None when the model is not
    installed. See ``ui/avatar/README.txt`` for how the person was made.
    """
    points = _AVATAR / "person_face.json"
    if not (PERSON_MODEL.is_file() and points.is_file()):
        return None
    info = json.loads(points.read_text(encoding="utf-8"))
    if len(info.get("face", [])) != len(FACE_INDICES):
        return None                       # made for a different set of face points
    return {"url": PERSON_URL, "face": info["face"], "orient": info["orient"]}


@lru_cache(maxsize=1)
def _gltf_loader() -> str:
    return _GLTF_LOADER.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _person_page() -> str:
    return _PERSON_PAGE.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _page() -> str:
    return _PAGE.read_text(encoding="utf-8")


def _inside(poly: NDArray, pts: NDArray) -> NDArray:
    """Even–odd point-in-polygon test, vectorised over ``pts``."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < at)
    return inside


def face_triangles(clip: SignClip, open_eyes: bool = True) -> list[int]:
    """Triangles spanning the face points, as indices into the face block.

    Triangulated once, on the clip's average face, so the surface keeps the
    same triangles while the face moves. The inside of the mouth is left open
    and no triangle reaches outside the face outline. The eyes are left open
    too, for drawn eyes; ``open_eyes=False`` covers them, for a face photo.
    """
    from scipy.spatial import Delaunay  # type: ignore[import-untyped]

    if not clip.has_depth:
        return []
    face = clip.layout.face_points()
    arr = np.asarray(clip.frames, dtype=np.float64).reshape(len(clip.frames), -1, 3)
    xy = arr[:, face.start:face.stop, :2]
    tracked = np.any(xy != 0.0, axis=-1)
    if tracked.mean() < 0.5:
        return []
    mean = np.where(tracked[..., None], xy, np.nan)
    with np.errstate(all="ignore"):
        avg: NDArray = np.asarray(np.nanmean(mean, axis=0))
    if np.isnan(avg).any():
        return []
    tri = Delaunay(avg).simplices
    centres = avg[tri].mean(axis=1)
    chains = dict(clip.layout.face)

    def outline(name: str, which: int = 0) -> NDArray:
        return np.asarray(avg[[p - face.start for p in chains[name][which]]])

    keep = _inside(outline("oval"), centres)
    holes = [outline("lips", 1)]
    if open_eyes:
        holes += [outline("left_eye"), outline("right_eye")]
    for hole in holes:
        keep &= ~_inside(hole, centres)
    return [int(v) for v in tri[keep].ravel()]


#: Fingertips (inside a hand's 21 points) and what to call them.
FINGERTIPS = {4: "Thumb", 8: "Index finger", 12: "Middle finger", 16: "Ring finger",
              20: "Little finger"}


@dataclass(frozen=True)
class Touch:
    """Where a sign touches the face, found from the recording itself.

    Attributes:
        point: Index of the touching fingertip in the frame (absolute).
        finger: "Thumb", "Index finger", ...
        frames: Frames in which it touches.
    """

    point: int
    finger: str
    frames: tuple[int, ...]


def _face_coords(frame: NDArray, pos: dict[int, int], p: NDArray) -> tuple[float, float] | None:
    """A point in the face's own coordinates: ``u`` across (±1 at the face's sides),
    ``v`` up (0 at the chin, 1 at the top of the forehead). Turning the head does
    not change them, which flat screen distances would."""
    chin, top = frame[pos[_CHIN], :2], frame[pos[_TOP], :2]
    side_a, side_b = frame[pos[_SIDE_A], :2], frame[pos[_SIDE_B], :2]
    basis = np.column_stack([(side_b - side_a) / 2, top - chin])
    if abs(np.linalg.det(basis)) < 1e-6:
        return None
    w = np.linalg.solve(basis, p - chin)
    w0 = np.linalg.solve(basis, (side_a + side_b) / 2 - chin)
    return float(w[0] - w0[0]), float(w[1])


def _at_face(u: float, v: float) -> bool:
    """Whether face coordinates fall on the face (or just at its edge)."""
    return -0.15 <= v <= 1.08 and abs(u) <= (1.0 if v > 0.85 else 1.2)


def find_touch(clip: SignClip) -> Touch | None:
    """The fingertip that touches the face in this sign, if one does.

    From the front, every fingertip of a hand held before the face *looks*
    inside it, so position alone cannot tell which one touches. Two things do:

    * **depth within the hand** — MediaPipe's depth of fingers relative to the
      wrist is reliable (unlike its arm depth), and the touching fingertip is
      the one pushed furthest towards the face;
    * **sitting on a face feature** — a fingertip right on the nose tip, the
      lips or the chin is touching it.

    Checked on the real library against signs with a known answer: MOM and
    GRANDMA (thumb, chin), DAD, GRANDPA and MAN (thumb, forehead), DONKEY
    (thumb, temple), NOSE, LIPS, TOOTH and THINK (index finger), and no touch
    for YES or AIRPLANE. Naming the *region* was not reliable — heads in the
    recordings are often turned — so the figure shows the spot, not a word.
    """
    if not clip.has_depth:
        return None
    face = clip.layout.face_points()
    pos = {mesh: face.start + i for i, mesh in enumerate(FACE_INDICES)}
    arr = np.asarray(clip.frames, dtype=np.float64).reshape(len(clip.frames), -1, 3)
    found: dict[int, dict[str, list[int]]] = {}
    for f, frame in enumerate(arr):
        face_xy = frame[face.start:face.stop, :2]
        if not face_xy.any():
            continue
        width = float(np.ptp(face_xy[:, 0])) or 1.0
        near: dict[int, float] = {}
        for hand in (0, 21):
            for tip in FINGERTIPS:
                p = frame[hand + tip, :2]
                if not p.any():
                    continue
                uv = _face_coords(frame, pos, p)
                if uv and _at_face(*uv):
                    near[hand + tip] = float(np.min(np.linalg.norm(face_xy - p, axis=1))) / width
        if not near:
            continue
        deepest = max(frame[i, 2] for i in near)       # larger z = further from the camera
        for i, gap in near.items():
            c = found.setdefault(i, {"at": [], "deep": [], "on": []})
            c["at"].append(f)
            if frame[i, 2] >= deepest - 0.5:
                c["deep"].append(f)
            if gap < 0.06:
                c["on"].append(f)
    best: tuple[int, int] | None = None
    for i, c in found.items():
        if len(c["at"]) < 4 or len(c["deep"]) + len(c["on"]) < 3:
            continue
        score = len(c["at"]) + 2 * len(c["deep"]) + 3 * len(c["on"])
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return None
    c = found[best[1]]
    frames = tuple(sorted(set(c["deep"]) | set(c["on"])))
    return Touch(best[1], FINGERTIPS[best[1] % 21], frames)   # 21 points per hand


@dataclass(frozen=True)
class Segment:
    """One sign inside a joined sentence clip: its name and frames (inclusive)."""

    word: str
    start: int
    end: int


def join_clips(clips: list[SignClip], words: list[str] | None = None,
               blend: int = 4, rest: int = 6) -> tuple[SignClip, list[Segment]]:
    """Join several signs into one clip the 3D person can perform in a row.

    Between two signs the hands move smoothly from where one sign ends to
    where the next starts (``blend`` frames); after the last sign the
    figure holds still for ``rest`` frames before the sentence repeats.

    Args:
        clips: Version-2 clips, all with the same layout.
        words: What to call each sign in the caption (default: the glosses).
        blend: Frames of movement from one sign into the next.
        rest: Frames held after the last sign.

    Returns:
        The joined clip and where each sign sits in it.
    """
    if not clips:
        raise ValueError("no signs to join")
    names = words or [c.gloss for c in clips]
    frames: list[list[float]] = []
    segments: list[Segment] = []
    for k, clip in enumerate(clips):
        if frames and blend > 0:
            a = np.asarray(frames[-1], dtype=np.float64).reshape(-1, 3)
            b = np.asarray(clip.frames[0], dtype=np.float64).reshape(-1, 3)
            both = np.any(a[:, :2] != 0, axis=1) & np.any(b[:, :2] != 0, axis=1)
            for i in range(1, blend + 1):
                w = i / (blend + 1)
                mid = np.where(both[:, None], a + (b - a) * w, b if w >= 0.5 else a)
                frames.append([float(v) for v in mid.ravel()])
        start = len(frames)
        frames.extend(list(f) for f in clip.frames)
        segments.append(Segment(names[k], start, len(frames) - 1))
    frames.extend(list(frames[-1]) for _ in range(rest))
    joined = SignClip(" ".join(names), frames, clips[0].fps, clips[0].layout)
    return joined, segments


def sign_3d_html(
    clip: SignClip, speed: float = 1.0, mirror: bool = True, frame: int | None = None,
    height: int = 420, title: str = "", person: bool = True, close_up: bool = False,
    segments: list[Segment] | None = None,
) -> str:
    """A self-contained HTML page showing the sign as a turnable 3D figure.

    Args:
        clip: A version-2 clip (``clip.has_depth``); anything else gets the
            flat drawing only.
        speed: Playback speed (0.5 = half speed).
        mirror: Flip left↔right, so the learner can copy it like a mirror.
        frame: Hold this single frame instead of animating (for stepping).
        height: Height of the view in pixels.
        title: Accessible name.
        person: Show the 3D person (``static/person.glb``) when it is
            installed; otherwise, or when False, the mannequin.
        close_up: Start with the camera on the head.
        segments: For a sentence made by :func:`join_clips`: where each sign
            is, so every sign gets its own touch spot and its name is shown
            while it is signed.

    Returns:
        HTML for ``st.iframe``.
    """
    label = title or clip.gloss
    flat = sign_svg(clip, speed=speed, mirror=mirror, frame=frame, height=height - 20,
                    title=label)                       # sign_svg escapes the title itself
    if not clip.has_depth:
        return f'<div style="height:{height}px">{flat}</div>'
    face = clip.layout.face_points()
    pos = {mesh: face.start + i for i, mesh in enumerate(FACE_INDICES)}
    model = person_model() if person else None
    data = {
        "frames": [[round(v, 1) for v in f] for f in clip.frames],
        "fps": clip.fps, "speed": max(speed, 0.1), "mirror": mirror,
        "frame": None if frame is None else int(frame) % len(clip.frames),
        "faceStart": face.start, "facePoints": len(face),
        "tri": face_triangles(clip),
        "features": {name: [list(c) for c in chains] for name, chains in clip.layout.face},
        "orient": [pos[_TOP], pos[_CHIN], pos[_SIDE_A], pos[_SIDE_B]],
        "lipsIn": [pos[_LIP_UP], pos[_LIP_DOWN]],
        "tiltFix": _tilt_fix(clip, pos[_TOP], pos[_CHIN], segments),
        "expr": {"brow": [pos[m] for m in _BROW], "lids": [pos[m] for m in _LIDS],
                 "knit": [pos[m] for m in _KNIT]},
        "touch": None if segments else _touch_data(find_touch(clip), len(clip.frames)),
        "touches": _sentence_touches(clip, segments) if segments else None,
        "words": [{"word": g.word, "start": g.start, "end": g.end}
                  for g in segments] if segments else None,
        "person": None,
        "avatar": model,
        "closeUp": close_up,
    }
    fill = {"__HEIGHT__": str(height), "__LABEL__": _escape(label), "__FLAT__": flat,
            "__DATA__": json.dumps(data, separators=(",", ":")), "__THREE__": _three_js(),
            "__GLTF__": _gltf_loader() if model else ""}
    # one pass, so a value can never be mistaken for a later placeholder
    return _TOKEN.sub(lambda m: fill[m.group(0)], _person_page() if model else _page())


def head_tilt(frame: list[float], top: int, chin: int) -> float | None:
    """How far the recorded head leans (radians; + = top of the head toward the camera)."""
    t, c = frame[3 * top: 3 * top + 3], frame[3 * chin: 3 * chin + 3]
    if not (t[0] or t[1]) or not (c[0] or c[1]):
        return None
    up, toward = -(t[1] - c[1]), -(t[2] - c[2])        # drawing y and z point down and away
    if abs(up) + abs(toward) < 1e-6:
        return None
    return float(np.arctan2(toward, up))


def _tilt_fix(clip: SignClip, top: int, chin: int,
              segments: list[Segment] | None) -> list[float]:
    """Per frame, the turn (radians) that brings each sign's steady head tilt within limits."""
    frames = clip.frames
    spans = [(g.start, g.end) for g in segments] if segments else [(0, len(frames) - 1)]
    known: dict[int, float] = {}
    for start, end in spans:
        found = (head_tilt(frames[i], top, chin) for i in range(start, end + 1))
        tilts = [x for x in found if x is not None]
        if not tilts:
            continue
        steady = float(np.median(tilts))
        turn = _TILT_LIMIT * float(np.tanh(steady / _TILT_LIMIT)) - steady
        known.update({i: turn for i in range(start, end + 1)})
    if not known:
        return [0.0] * len(frames)
    # the frames between signs of a sentence ease from one sign's turn to the next one's
    at = sorted(known)
    every = np.interp(np.arange(len(frames)), at, [known[i] for i in at])
    return [round(float(x), 4) for x in every]


def _sentence_touches(clip: SignClip, segments: list[Segment]) -> list[dict[str, object]]:
    """A touch for each sign of a sentence that has one, in whole-clip frames."""
    out: list[dict[str, object]] = []
    total = len(clip.frames)
    for seg in segments:
        part = SignClip(seg.word, clip.frames[seg.start:seg.end + 1], clip.fps, clip.layout)
        touch = find_touch(part)
        data = _touch_data(touch, len(part.frames))
        if touch is None or data is None:
            continue
        weight = [0.0] * total
        part_weight: list[float] = data["weight"]  # type: ignore[assignment]
        for i, w in enumerate(part_weight):
            weight[seg.start + i] = w
        out.append({**data, "weight": weight, "start": seg.start, "end": seg.end})
    return out


def _touch_data(touch: Touch | None, frames: int) -> dict[str, object] | None:
    """What the page needs to show a touch: the fingertip, its hand, and how
    strongly it touches in each frame (1 while touching, easing in and out
    over a frame on either side so the hand does not jump)."""
    if touch is None:
        return None
    weight = [0.0] * frames
    for f in touch.frames:
        weight[f] = 1.0
    eased = list(weight)
    for f in range(frames):
        if weight[f] == 0.0 and (weight[f - 1] or weight[(f + 1) % frames]):
            eased[f] = 0.5
    return {"point": touch.point, "hand": 0 if touch.point < 21 else 21,
            "finger": touch.finger, "weight": eased}


def _escape(text: str) -> str:
    return html.escape(text, quote=True)
