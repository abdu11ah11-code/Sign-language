"""Tests for the version-2 figure library and the 3D figure page."""

from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

from sign_bridge.ui.animation import (
    FACE_INDICES,
    LAYOUT_V2,
    POINTS_V2,
    SignClip,
    SignLibrary,
    sign_svg,
)
from sign_bridge.ui.figure3d import (
    _TILT_LIMIT,
    _VENDOR,
    Segment,
    face_triangles,
    head_tilt,
    join_clips,
    sign_3d_html,
)


def _v2_frame(t: float = 0.0, mouth_open: float = 1.0) -> list[float]:
    """A plausible version-2 frame: body, a rising right hand, and a drawn face."""
    pts = np.zeros((POINTS_V2, 3))
    body = {0: (50, 22, -10), 1: (48, 20, -9), 2: (52, 20, -9), 3: (50, 26, -9),
            4: (37, 46, 0), 5: (63, 46, 0), 6: (33, 60, -5), 7: (67, 60, -5),
            8: (40, 62, -15), 9: (60, 50 - 15 * t, -18), 10: (41, 76, 2), 11: (59, 76, 2)}
    for k, xyz in body.items():
        pts[42 + k] = xyz
    for k in range(21):                                     # right hand, fingers up
        pts[21 + k] = (60 + (k % 5) * 0.8, 48 - 15 * t - (k // 5) * 1.5, -18 - 0.2 * k)
    face = dict(LAYOUT_V2.face)
    centre = np.array([50.0, 20.0])

    def ring(chain: tuple[int, ...], cx: float, cy: float, rx: float, ry: float, z: float) -> None:
        loop = chain[:-1] if chain[0] == chain[-1] else chain
        for n, p in enumerate(loop):
            a = 2 * math.pi * n / len(loop)
            pts[p] = (cx + rx * math.cos(a), cy + ry * math.sin(a), z)

    ring(face["oval"][0], *centre, 8, 10, -6)
    ring(face["left_eye"][0], 53, 17, 1.6, 0.6, -10)
    ring(face["right_eye"][0], 47, 17, 1.6, 0.6, -10)
    for chain, (cx, cy) in ((face["left_brow"][0], (53, 14.5)), (face["left_brow"][1], (53, 14)),
                            (face["right_brow"][0], (47, 14.5)), (face["right_brow"][1], (47, 14))):
        for n, p in enumerate(chain):
            pts[p] = (cx - 2 + n, cy, -10)
    for n, p in enumerate(face["nose"][0]):
        pts[p] = (50 + 0.8 * math.sin(n), 18 + n * 0.2, -12)
    ring(face["lips"][0], 50, 25, 3, 1.2 + mouth_open * 0.4, -10)
    ring(face["lips"][1], 50, 25, 2, 0.3 + mouth_open * 0.5, -10)
    return [float(v) for v in pts.ravel()]


def _v2_clip(n: int = 6) -> SignClip:
    return SignClip("MOM", [_v2_frame(i / (n - 1), mouth_open=i / (n - 1)) for i in range(n)],
                    12.0, LAYOUT_V2)


# ── the version-2 library ────────────────────────────────────────────────
def test_face_points_are_a_fixed_known_set() -> None:
    assert len(FACE_INDICES) == 152 and POINTS_V2 == 206
    assert tuple(sorted(set(FACE_INDICES))) == FACE_INDICES
    assert all(0 <= i < 468 for i in FACE_INDICES)
    assert 1 in FACE_INDICES                     # the nose tip the notebook anchors on


def test_v2_library_roundtrip(tmp_path: Path) -> None:
    clip = _v2_clip()
    path = tmp_path / "asl.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"version": 2, "fps": 12, "face_indices": list(FACE_INDICES),
                   "signs": {"MOM": {"frames": clip.frames}}}, fh)
    lib = SignLibrary.load(path)
    got = lib.get("mom")
    assert got is not None and got.has_depth and lib.has_depth
    assert got.layout is LAYOUT_V2 and len(got.frames[0]) == 618


def test_v2_library_from_another_release_is_refused(tmp_path: Path) -> None:
    """Different face points would be drawn in the wrong places — refuse instead."""
    path = tmp_path / "asl.json"
    path.write_text(json.dumps({"version": 2, "face_indices": [1, 2, 3],
                                "signs": {"MOM": {"frames": _v2_clip().frames}}}))
    assert len(SignLibrary.load(path)) == 0


def test_flat_drawing_shows_the_face_for_v2() -> None:
    svg = sign_svg(_v2_clip(), mirror=False)
    assert svg.count("sa-lips") == 2                 # outer and inner lip
    assert "<circle" not in svg                      # the face outline replaces the circle
    assert svg.count("sa-head") >= 7                 # outline, eyes, brows, nose


# ── face surface ─────────────────────────────────────────────────────────
def test_face_surface_leaves_eyes_and_mouth_open() -> None:
    clip = _v2_clip()
    tri = face_triangles(clip)
    assert tri and len(tri) % 3 == 0
    assert max(tri) < len(FACE_INDICES)
    arr = np.asarray(clip.frames).reshape(len(clip.frames), -1, 3)[:, 54:, :2].mean(0)
    centres = arr[np.asarray(tri).reshape(-1, 3)].mean(1)
    # nothing covers the mouth opening or the middle of an eye
    for cx, cy, r in ((50, 25, 0.3), (53, 17, 0.4), (47, 17, 0.4)):
        assert not np.any(np.hypot(centres[:, 0] - cx, centres[:, 1] - cy) < r)


def test_face_surface_needs_a_tracked_face() -> None:
    frames = [[0.0] * 618 for _ in range(4)]
    assert face_triangles(SignClip("X", frames, 12.0, LAYOUT_V2)) == []


# ── the 3D page ──────────────────────────────────────────────────────────
def _data(page: str) -> dict:
    match = re.search(r"const D = (\{.*?\});\n", page)
    assert match, "page carries its data"
    return json.loads(match.group(1))


def test_page_is_self_contained() -> None:
    for person in (True, False):
        page = sign_3d_html(_v2_clip(), title="Mom", person=person)
        assert page.startswith("<!doctype html>")
        assert "THREE.WebGLRenderer" in page and "Three.js Authors" in page   # three.js inlined
        assert not re.search(r'(src|href)\s*=\s*["\']?https?:', page)          # nothing from web
        assert "showing how to sign Mom" in page
        assert "<svg" in page                                                 # flat fallback inside


def test_page_data_matches_the_clip() -> None:
    clip = _v2_clip()
    data = _data(sign_3d_html(clip, speed=0.5, mirror=False))
    assert len(data["frames"]) == len(clip.frames) and len(data["frames"][0]) == 618
    assert data["speed"] == 0.5 and data["mirror"] is False and data["frame"] is None
    assert data["faceStart"] == 54 and data["facePoints"] == 152
    assert data["tri"] and set(data["features"]) >= {"oval", "lips", "left_eye", "right_brow"}
    assert _data(sign_3d_html(clip, frame=8))["frame"] == 8 % len(clip.frames)


def test_old_library_gets_the_flat_drawing_only() -> None:
    v1 = SignClip("OLD", [[50.0] * 188, [51.0] * 188])
    page = sign_3d_html(v1)
    assert "<script>" not in page and "<svg" in page


def test_title_is_escaped() -> None:
    page = sign_3d_html(_v2_clip(), title='<b>"x"</b>')
    assert "<b>" not in page.split("<script>", 1)[0]
    assert "&lt;b&gt;&quot;x&quot;" in page
    assert "&lt;b&gt;" in sign_svg(_v2_clip(), title="<b>")      # the flat drawing too


def test_three_js_is_bundled_with_its_licence() -> None:
    assert _VENDOR.is_file() and _VENDOR.stat().st_size > 500_000
    assert (_VENDOR.parent / "THREE_LICENSE.txt").read_text(encoding="utf-8").startswith(
        "The MIT License")


def test_every_placeholder_is_filled_once() -> None:
    from sign_bridge.ui.figure3d import _three_js

    for person in (True, False):
        page = sign_3d_html(_v2_clip(), title="Mom", person=person)
        # three.js itself contains "__THREE__" (window.__THREE__); only those may remain
        left = re.findall(r"__(?:HEIGHT|LABEL|FLAT|DATA|THREE|GLTF)__", page)
        assert left == re.findall(r"__(?:HEIGHT|LABEL|FLAT|DATA|THREE|GLTF)__", _three_js())
        assert page.count("Three.js Authors") == 1 and page.count("const D = ") == 1


def test_visibility_is_always_a_real_boolean() -> None:
    """three.js hides an object only when ``visible === false``: a 0 still draws.

    A 0/1 flag once drew an untracked hand as a blob at the corner of the box.
    """
    page = Path(__file__).resolve().parents[1] / "src/sign_bridge/ui/figure3d_page.html"
    js = page.read_text(encoding="utf-8")
    assert "Uint8Array(N)" not in js
    for line in js.splitlines():
        if re.search(r"\.visible = .*seen\[", line):
            assert "!!" in line, line


# ── where a sign touches the face ────────────────────────────────────────
def _with_anatomy(frame: list[float]) -> np.ndarray:
    """The synthetic face, with its top, chin and sides where a real face has them."""
    pts = np.asarray(frame).reshape(-1, 3).copy()
    at = {mesh: 54 + i for i, mesh in enumerate(FACE_INDICES)}
    pts[at[10]], pts[at[152]] = (50, 10, -6), (50, 30, -6)       # top of forehead, chin
    pts[at[234]], pts[at[454]] = (42, 20, -6), (58, 20, -6)      # the face's two sides
    return pts


def _touching_clip(touch: bool) -> SignClip:
    frames = []
    for i in range(8):
        pts = _with_anatomy(_v2_frame(0.0))
        if touch:                       # right thumb tip on the mouth, pushed towards the face
            pts[21 + 4] = (50, 25, -6)
            for tip in (8, 12, 16, 20):
                pts[21 + tip] = (51 + tip * 0.2, 22 + i * 0.1, -14)
        frames.append([float(v) for v in pts.ravel()])
    return SignClip("TEST", frames, 12.0, LAYOUT_V2)


def test_finds_the_touching_fingertip() -> None:
    from sign_bridge.ui.figure3d import find_touch

    touch = find_touch(_touching_clip(True))
    assert touch is not None
    assert touch.finger == "Thumb" and touch.point == 21 + 4
    assert len(touch.frames) == 8


def test_no_touch_when_the_hand_stays_away() -> None:
    from sign_bridge.ui.figure3d import find_touch

    assert find_touch(_touching_clip(False)) is None
    assert find_touch(SignClip("OLD", [[50.0] * 188, [51.0] * 188])) is None   # no depth


def test_face_photo_covers_the_eyes() -> None:
    """A photo has its own eyes, so the surface must span them; drawn eyes stay open."""
    clip = _v2_clip()
    assert len(face_triangles(clip, open_eyes=False)) > len(face_triangles(clip))


# ── the 3D person ───────────────────────────────────────────────────────
def _glb_json() -> dict:
    import struct

    from sign_bridge.ui.figure3d import PERSON_MODEL

    raw = PERSON_MODEL.read_bytes()
    assert raw[:4] == b"glTF" and struct.unpack("<I", raw[4:8])[0] == 2
    length, kind = struct.unpack("<II", raw[12:20])
    assert kind == 0x4E4F534A                                # the JSON chunk comes first
    return json.loads(raw[20:20 + length])


def test_person_model_is_installed_and_fits_the_face_points() -> None:
    from sign_bridge.ui.figure3d import PERSON_URL, person_model

    model = person_model()
    assert model is not None and model["url"] == PERSON_URL == "/app/static/person.glb"
    face, orient = model["face"], model["orient"]
    assert isinstance(face, list) and len(face) == len(FACE_INDICES)
    placed = [f for f in face if f]
    assert len(placed) >= 140 and {f[3] for f in placed} == {"head", "jaw"}
    assert all(face[k] for k in orient)                       # the four orientation points
    top, chin = face[orient[0]], face[orient[1]]
    assert 0.15 < top[1] - chin[1] < 0.25                     # a face is about 20 cm tall


def test_person_has_every_bone_the_page_moves() -> None:
    nodes = {n.get("name") for n in _glb_json()["nodes"]}
    needed = {"spine01", "spine02", "spine03", "spine04", "neck01", "neck02", "neck03",
              "head", "jaw"}
    for side in "LR":
        needed |= {f"{b}_{side}" for b in ("upperarm01", "lowerarm01", "lowerarm02", "wrist")}
        needed |= {f"finger{f}-{k}_{side}" for f in range(1, 6) for k in (1, 2, 3)}
        # the face: eyes look at the viewer, lids blink, brows rise and knit
        needed |= {f"{b}_{side}" for b in ("eye", "orbicularis03", "orbicularis04", "oculi01")}
    assert needed <= nodes, needed - nodes
    page = (Path(__file__).resolve().parents[1]
            / "src/sign_bridge/ui/figure3d_person.html").read_text(encoding="utf-8")
    for name in ("upperarm01_", "lowerarm02_", "wrist_", "finger5-3", "neck03", "spine04",
                 "orbicularis03_", "oculi01_", "eye_L"):
        assert name in page


def test_person_model_carries_its_provenance() -> None:
    readme = (Path(__file__).resolve().parents[1] / "src/sign_bridge/ui/avatar/README.txt"
              ).read_text(encoding="utf-8")
    assert "CC0" in readme and "Gemini" in readme and "not a photograph" in readme
    assert (Path(__file__).resolve().parents[1] / "tools/avatar/build_all.sh").is_file()


def test_page_shows_the_person_when_installed() -> None:
    page = sign_3d_html(_v2_clip(), title="Mom")
    data = _data(page)
    assert data["avatar"]["url"] == "/app/static/person.glb"
    assert "THREE.GLTFLoader = GLTFLoader" in page and 'aria-label="3D person showing' in page
    assert data["lipsIn"] and len(data["lipsIn"]) == 2
    old = sign_3d_html(_v2_clip(), title="Mom", person=False)
    assert _data(old)["avatar"] is None and "GLTFLoader" not in old
    assert 'aria-label="3D figure showing how to sign Mom"' in old


def test_page_falls_back_to_the_mannequin_without_the_model(monkeypatch: object,
                                                            tmp_path: Path) -> None:
    import sign_bridge.ui.figure3d as f3

    monkeypatch.setattr(f3, "PERSON_MODEL", tmp_path / "missing.glb")  # type: ignore[attr-defined]
    f3.person_model.cache_clear()
    try:
        page = sign_3d_html(_v2_clip())
        assert _data(page)["avatar"] is None and "GLTFLoader" not in page
    finally:
        f3.person_model.cache_clear()


def test_app_serves_the_model() -> None:
    import inspect

    from sign_bridge import main

    assert '"--server.enableStaticServing", "true"' in inspect.getsource(main.run_app)
    root = Path(__file__).resolve().parents[1]
    cfg = (root / ".streamlit/config.toml").read_text(encoding="utf-8")
    assert "enableStaticServing = true" in cfg


def test_page_has_a_close_up_button() -> None:
    for person in (True, False):
        page = sign_3d_html(_v2_clip(), person=person)
        assert 'id="zoom"' in page and "Close-up" in page


def test_page_carries_the_touch() -> None:
    data = _data(sign_3d_html(_touching_clip(True)))
    assert data["touch"]["finger"] == "Thumb" and data["touch"]["hand"] == 21
    assert len(data["touch"]["weight"]) == 8 and max(data["touch"]["weight"]) == 1.0
    assert _data(sign_3d_html(_touching_clip(False)))["touch"] is None


# ── the person's face and head ───────────────────────────────────────────
def _upright() -> list[float]:
    """A frame whose forehead top sits straight above its chin (the drawn face's ring does not)."""
    pts = np.asarray(_v2_frame(), float).reshape(-1, 3)
    face = LAYOUT_V2.face_points()
    pos = {m: face.start + i for i, m in enumerate(FACE_INDICES)}
    pts[pos[10]], pts[pos[152]] = (50, 10, -8), (50, 30, -8)
    return [float(v) for v in pts.ravel()]


def _leaning(frame: list[float], degrees: float) -> list[float]:
    """The frame with the whole face turned about the x axis (+ = top toward the camera)."""
    pts = np.asarray(frame, float).reshape(-1, 3).copy()
    face = LAYOUT_V2.face_points()
    a = math.radians(degrees)
    # drawing y and z point down and away: turn in (up, toward) around the face's middle
    up, toward = 20.0 - pts[face.start:, 1], -8.0 - pts[face.start:, 2]
    up2, toward2 = up * math.cos(a) - toward * math.sin(a), up * math.sin(a) + toward * math.cos(a)
    pts[face.start:, 1], pts[face.start:, 2] = 20.0 - up2, -8.0 - toward2
    return [float(v) for v in pts.ravel()]


def test_head_tilt_reads_the_lean() -> None:
    face = LAYOUT_V2.face_points()
    pos = {m: face.start + i for i, m in enumerate(FACE_INDICES)}
    upright = _upright()
    assert abs(head_tilt(upright, pos[10], pos[152]) or 0.0) < 1e-6
    back = head_tilt(_leaning(upright, -30), pos[10], pos[152])
    front = head_tilt(_leaning(upright, 20), pos[10], pos[152])
    assert back is not None and back < -0.3
    assert front is not None and front > 0.2
    assert head_tilt([0.0] * len(upright), pos[10], pos[152]) is None


def test_steady_lean_is_eased_but_nods_are_kept() -> None:
    back = SignClip("SAD", [_leaning(_upright(), -40) for _ in range(6)], 12.0, LAYOUT_V2)
    fix = _data(sign_3d_html(back))["tiltFix"]
    assert len(fix) == 6 and len(set(fix)) == 1           # one turn for the whole sign
    face = LAYOUT_V2.face_points()
    pos = {m: face.start + i for i, m in enumerate(FACE_INDICES)}
    lean = head_tilt(back.frames[0], pos[10], pos[152]) or 0.0
    assert -_TILT_LIMIT <= lean + fix[0] < 0               # still leaning back, but less
    still = SignClip("MOM", [_upright() for _ in range(6)], 12.0, LAYOUT_V2)
    upright = _data(sign_3d_html(still))["tiltFix"]
    assert max(abs(x) for x in upright) < 0.01             # an upright sign is left alone


def test_sentence_eases_the_tilt_between_signs() -> None:
    back = SignClip("SAD", [_leaning(_upright(), -40) for _ in range(6)], 12.0, LAYOUT_V2)
    still = SignClip("MOM", [_upright() for _ in range(6)], 12.0, LAYOUT_V2)
    clip, segments = join_clips([back, still], ["SAD", "MOM"])
    fix = _data(sign_3d_html(clip, segments=segments))["tiltFix"]
    first, second = segments
    assert all(isinstance(g, Segment) for g in segments)
    assert fix[first.start] > 0.2 and abs(fix[second.end]) < 0.01
    between = fix[first.end:second.start + 1]
    assert between == sorted(between, reverse=True)        # eases down, no jump


def test_page_carries_the_expression_points() -> None:
    data = _data(sign_3d_html(_v2_clip()))
    face = LAYOUT_V2.face_points()
    names = {face.start + i: m for i, m in enumerate(FACE_INDICES)}
    expr = data["expr"]
    assert [names[i] for i in expr["brow"]] == [334, 386, 105, 159]
    assert [names[i] for i in expr["lids"]] == [386, 374, 263, 362, 159, 145, 33, 133]
    assert [names[i] for i in expr["knit"]] == [285, 55]
    page = sign_3d_html(_v2_clip())
    for part in ("function poseFace", "function poseEyes", "function relaxHand",
                 "function wristSeen", "blinkAmount"):
        assert part in page
