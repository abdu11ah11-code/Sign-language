"""Tests for the animated sign helper (library loading + SVG drawing)."""

from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

from sign_bridge.ui.animation import (
    LEFT_HAND,
    LIPS,
    POINTS,
    RIGHT_HAND,
    SignClip,
    SignLibrary,
    sign_svg,
)

FPS = 12.0


def _frame(t: float, left_hand: bool = True, lips: bool = True) -> list[float]:
    """One frame: body still, right hand moving up, optional left hand / lips."""
    pts = [0.0] * (POINTS * 2)

    def put(i: int, x: float, y: float) -> None:
        pts[2 * i], pts[2 * i + 1] = x, y

    for k in range(21):                                    # right hand, rising
        put(RIGHT_HAND.start + k, 58 + k * 0.4, 60 - 20 * t + k * 0.3)
    if left_hand:
        for k in range(21):
            put(LEFT_HAND.start + k, 36 + k * 0.4, 62 + k * 0.3)
    body = {0: (50, 26), 1: (46, 24), 2: (54, 24), 3: (50, 30), 4: (38, 42), 5: (62, 42),
            6: (34, 56), 7: (66, 56), 8: (36, 68), 9: (64, 68), 10: (42, 76), 11: (58, 76)}
    for k, (x, y) in body.items():
        put(42 + k, x, y)
    if lips:
        for k in range(40):
            put(LIPS.start + k, 47 + (k % 10) * 0.6, 31 + (k // 10) * 0.5)
    return pts


def _clip(n: int = 8, **kw: bool) -> SignClip:
    return SignClip("MOM", [_frame(i / (n - 1), **kw) for i in range(n)], FPS)


def test_clip_length() -> None:
    clip = _clip(12)
    assert math.isclose(clip.seconds, 1.0)


def test_library_roundtrip(tmp_path: Path) -> None:
    clip = _clip()
    path = tmp_path / "asl.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"version": 1, "language": "asl", "fps": FPS,
                   "signs": {"MOM": {"frames": clip.frames},
                             "dad": {"frames": _clip().frames}}}, fh)
    lib = SignLibrary.load(path)
    assert len(lib) == 2
    assert lib.glosses() == ["DAD", "MOM"]
    assert "mom" in lib and "nothing" not in lib           # lookup ignores case
    got = lib.get("MoM")
    assert got is not None and len(got.frames) == len(clip.frames)
    assert lib.get("NOPE") is None


def test_library_missing_or_broken_file_is_empty(tmp_path: Path) -> None:
    assert len(SignLibrary.load(tmp_path / "nope.json.gz")) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert len(SignLibrary.load(bad)) == 0
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"signs": {"X": {"frames": [[1.0, 2.0]]}}}), encoding="utf-8")
    assert len(SignLibrary.load(short)) == 0               # wrong width → skipped


def test_svg_structure_and_timing() -> None:
    svg = sign_svg(_clip(), speed=1.0, mirror=False)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert 'viewBox="0 0 100 100"' in svg
    assert "aria-label=" in svg and "sign MOM" in svg
    assert svg.count("<animate") >= 3                      # head + body + hands
    durs = set(re.findall(r'dur="([\d.]+)s"', svg))
    assert durs == {f"{8 / FPS:.2f}"}                      # every part shares one duration
    # every animated polyline lists all frames (+1 repeated for a smooth loop)
    for values in re.findall(r'values="([^"]*,[^"]*)"', svg):
        assert values.count(";") == 8


def test_speed_and_mirror() -> None:
    assert f'dur="{16 / FPS:.2f}s"' in sign_svg(_clip(), speed=0.5, mirror=False)
    assert "scale(-1,1)" in sign_svg(_clip(), mirror=True)
    assert "scale(-1,1)" not in sign_svg(_clip(), mirror=False)


def test_single_frame_is_static() -> None:
    svg = sign_svg(_clip(), frame=3)
    for values in re.findall(r'values="([^"]*,[^"]*)"', svg):
        points = values.split(";")
        assert len(set(points)) == 1                       # the pose never changes


def test_untracked_parts_are_not_drawn() -> None:
    both = sign_svg(_clip(), mirror=False)
    one = sign_svg(_clip(left_hand=False), mirror=False)
    no_lips = sign_svg(_clip(lips=False), mirror=False)
    assert one.count("sa-hand") < both.count("sa-hand")    # absent hand omitted
    assert "sa-lips" in both and "sa-lips" not in no_lips
    assert "sa-body" in one                                # the body is always drawn


def test_no_zero_coordinates_leak_into_drawn_hand() -> None:
    """An untracked point must never pull a drawn line to the corner."""
    svg = sign_svg(_clip(left_hand=False), mirror=False)
    hand_lines = [m for m in re.findall(r'<polyline class="sa-hand"[^>]*points="([^"]*)"', svg)]
    assert hand_lines
    for points in hand_lines:
        assert "0.0,0.0" not in points


def test_animation_notebook_builds(tmp_path: Path) -> None:
    """The Kaggle notebook generator produces valid, compilable cells."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    import build_animation_notebook as builder

    path = builder.build(tmp_path / "nb.ipynb")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4 and len(data["cells"]) >= 6
    for i, src in enumerate(builder.code_cells()):
        compile(src, f"<cell {i}>", "exec")
    joined = "\n".join(builder.code_cells())
    assert "asl.json.gz" in joined and "animation_preview.png" in joined
    assert "face_indices" in joined and "bone_depth" in joined      # version-2 library
