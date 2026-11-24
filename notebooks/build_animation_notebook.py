"""Generate ``make_sign_animations.ipynb`` — the sign-animation builder.

Run ``python notebooks/build_animation_notebook.py`` to (re)write the notebook.
Keeping the cells here as plain strings means they stay lintable, diffable and
testable; :func:`code_cells` is used by the tests.

The notebook itself runs on Kaggle (CPU only, a few minutes) against the Google
ASL Signs data and writes ``asl.json.gz`` — one representative recording per
sign for SignBridge's "How to sign" animation.
"""

from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
PY = "code"

CELLS: list[tuple[str, str]] = [
    (MD, r"""# SignBridge — build the "How to sign" animations

Picks **one clear, typical recording of every sign** from Google's ASL Signs data and
saves it as a small animation library the SignBridge app draws as a figure — in 3D, which
you can turn to see a hand from the side, or flat.

* **No GPU needed** — a few minutes on CPU.
* **No video, no images** — only landmark positions: both hands, 12 upper-body points and
  the face's outline, eyes, eyebrows, nose and lips (152 of MediaPipe's 468 face points).
  The face is included because it is part of ASL grammar: raised eyebrows mark a yes/no
  question, a frown a *wh-* question.
* Every point keeps its **depth (z)**. MediaPipe measures each hand's depth from its own
  wrist and the face's from its own centre, so both are re-anchored on the body's wrists
  and nose, putting everything in one shared 3D space.
* For each sign, the clip **closest to the average** of that sign is chosen, so the
  example is typical rather than unusual, and clips where a hand was lost are skipped.

### Setup (same inputs as the training notebook)
1. **+ Add Input** → the competition **Google - Isolated Sign Language Recognition**.
2. **Upload** → your SignBridge project zip (any recent version).
3. **Run All**, then download `asl.json.gz` from the Output panel into the project's
   `assets/sign_animations/` folder.

> **Licence.** These animations are derived from the competition data. Keep the file
> local (it is git-ignored) — do not publish or redistribute it."""),
    (PY, r"""# ── settings ─────────────────────────────────────────────────────────────
import os
from pathlib import Path

FRAMES = int(os.environ.get("SB_AFRAMES", 24))     # frames kept per sign
FPS = float(os.environ.get("SB_AFPS", 12))         # playback speed of the animation
CANDIDATES = int(os.environ.get("SB_ACAND", 25))   # clips examined per sign
MIN_HAND_COVERAGE = 0.6                            # a hand must be tracked in this share of frames
BOX, SHOULDER_UNITS, CHEST_Y = 100.0, 26.0, 46.0   # drawing box: size, shoulder width, chest height
SUBSET_SIGNS = int(os.environ["SB_SUBSET"]) if os.environ.get("SB_SUBSET") else None
SEED = 42

DATA_DIR = Path(os.environ.get("SB_DATA_DIR", "/kaggle/input/asl-signs"))
SRC_SEARCH = Path(os.environ.get("SB_SRC", "/kaggle/input"))
OUT = Path(os.environ.get("SB_OUT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)

# Kaggle has used more than one folder layout for attached data; find train.csv wherever it is.
if not (DATA_DIR / "train.csv").is_file():
    hits = [c.parent for pat in ("*/train.csv", "*/*/train.csv", "*/*/*/train.csv")
            for c in sorted(SRC_SEARCH.glob(pat)) if (c.parent / "train_landmark_files").is_dir()]
    if not hits:
        raise FileNotFoundError(
            "ASL Signs data not found. In the right panel: Add Input → Competitions → "
            "'Google - Isolated Sign Language Recognition' (accept the competition rules first).")
    DATA_DIR = hits[0]
print("data:", DATA_DIR, "| output:", OUT, "| frames per sign:", FRAMES)"""),
    (PY, r"""# ── make the SignBridge code importable (same landmark layout as the live app) ──
import json, re, shutil, sys, time

def find_signbridge_src(root: Path) -> Path:
    # several project versions may be attached at once — take the newest
    def version(init: Path) -> tuple:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', init.read_text(encoding="utf-8"))
        return tuple(int(p) for p in m.group(1).split(".")) if m else (0,)

    if (root / "sign_bridge" / "__init__.py").is_file() and root.name == "src":
        return root
    found = [i for i in sorted(root.rglob("sign_bridge/__init__.py"))
             if i.parent.parent.name == "src" and (i.parent / "data" / "preprocess.py").is_file()]
    if not found:
        raise FileNotFoundError(
            f"SignBridge source not found under {root}. Add your uploaded SignBridge "
            "dataset as an input (it must contain src/sign_bridge/).")
    best = max(found, key=version)
    print("using SignBridge", ".".join(str(v) for v in version(best)), "from", best.parent.parent)
    return best.parent.parent

src_dir = find_signbridge_src(SRC_SEARCH)
project_dir = src_dir.parent
local_copy = OUT / "signbridge_src"
SKIP_TOP = {"data", "models", "logs", "recordings", ".venv", "assets"}

def _ignore(directory, names):
    top = Path(directory).resolve() == project_dir.resolve()
    return [n for n in names
            if (top and n in SKIP_TOP) or n == "__pycache__" or n.endswith(".egg-info")]

if local_copy.resolve() != project_dir.resolve():
    shutil.rmtree(local_copy, ignore_errors=True)
    shutil.copytree(project_dir, local_copy, ignore=_ignore)
sys.path.insert(0, str(local_copy / "src"))
os.environ["SIGN_BRIDGE_CONFIG"] = str(local_copy / "config.yaml")
# Drop any sign_bridge modules a previous run left in memory, so re-running this cell
# after attaching a newer project version really does pick up the newer code.
for _name in [m for m in sys.modules if m == "sign_bridge" or m.startswith("sign_bridge.")]:
    del sys.modules[_name]

import numpy as np, pandas as pd
import sign_bridge
from sign_bridge.capture.holistic import (
    LEFT_SHOULDER_OFFSET, LIP_LANDMARK_INDICES, POSE_LANDMARK_INDICES, RAW_DIM,
    RIGHT_SHOULDER_OFFSET)
from sign_bridge.data.preprocess import resample_clip
from sign_bridge.ui.animation import FACE_INDICES, POINTS_V2

print("SignBridge", sign_bridge.__version__, "| points per frame:", POINTS_V2, "(x, y, z)")"""),
    (MD, r"""## Landmarks → drawing coordinates
Every clip is placed in the same space: the chest sits at a fixed spot and the shoulders
are always 26 units apart, so all signs are drawn at one scale, whatever the camera
distance. Depth uses the same units, with the chest at z = 0 and negative z towards the
camera (MediaPipe's convention). MediaPipe's own arm depth is unreliable, so the arms'
depth is worked out from **bone lengths**: an upper arm that looks short from the front
must be pointing at the camera, by exactly the amount Pythagoras gives. Points the tracker never found stay at zero and are
not drawn; a hand that disappears for a moment keeps its last position."""),
    (PY, r"""from multiprocessing import Pool

TYPE_OFFSET = {"face": 0, "left_hand": 468, "pose": 489, "right_hand": 522}
SEL = np.concatenate([
    468 + np.arange(21), 522 + np.arange(21),
    489 + np.array(POSE_LANDMARK_INDICES), np.array(FACE_INDICES)])
assert SEL.size == POINTS_V2
BLOCKS = {"left": slice(0, 21), "right": slice(21, 42), "pose": slice(42, 54),
          "face": slice(54, POINTS_V2)}
P_NOSE, P_SH_L, P_SH_R, P_EL_L, P_EL_R, P_WR_L, P_WR_R = 42, 46, 47, 48, 49, 50, 51
P_HIP_L, P_HIP_R = 52, 53
F_NOSE_TIP = 54 + FACE_INDICES.index(1)                         # face-mesh point 1
UPPER_ARM, FOREARM = 21.0, 18.0  # typical bone lengths, in units of a 26-unit shoulder width
Z_SMOOTH = 5                     # depth smoothing window (frames)

def load_raw(rel_path: str) -> np.ndarray:
    # (frames, POINTS_V2, 3) raw landmarks; NaN where the tracker found nothing
    df = pd.read_parquet(DATA_DIR / rel_path)
    idx_col = "landmark_index" if "landmark_index" in df.columns else "landmark_idx"
    g = df["type"].map(TYPE_OFFSET).to_numpy() + df[idx_col].to_numpy()
    frames, f_idx = np.unique(df["frame"].to_numpy(), return_inverse=True)
    arr = np.full((len(frames), 543, 3), np.nan, dtype=np.float32)
    arr[f_idx, g] = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return arr[:, SEL, :]

def hold_last(pts: np.ndarray, block: slice, present: np.ndarray) -> np.ndarray:
    # carry a block's last tracked position through frames where it was lost
    if not present.any():
        return pts
    idx = np.where(present, np.arange(len(pts)), 0)
    idx = np.maximum.accumulate(idx)
    first = int(np.argmax(present))
    idx[:first] = first                       # before the first sighting, show the first pose
    pts[:, block] = pts[idx][:, block]
    return pts

def smooth(z: np.ndarray, window: int) -> np.ndarray:
    # moving average over time, per column
    if window <= 1 or len(z) < 3:
        return z
    pad = window // 2
    padded = np.pad(z, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(z.shape[1])], 1)

def bone_depth(a: np.ndarray, b: np.ndarray, typical: float) -> np.ndarray:
    # How far b sits in front of a (negative = towards the camera), from the bone's length.
    # MediaPipe's arm depth is unreliable, but a bone has a fixed length: the shorter it
    # looks from the front, the more it points at the camera. Only the direction comes
    # from MediaPipe, and only when it is clear; signing happens in front of the body.
    flat = np.linalg.norm(b[:, :2] - a[:, :2], axis=1)
    length = float(np.clip(np.nanpercentile(flat, 90), 0.85 * typical, 1.25 * typical))
    reach = np.sqrt(np.clip(length ** 2 - flat ** 2, 0.0, None))
    mp = smooth((b[:, 2] - a[:, 2])[:, None], Z_SMOOTH)[:, 0]
    behind = mp > 0.3 * length
    return np.where(behind, reach, -reach)

def to_drawing(raw: np.ndarray) -> np.ndarray | None:
    # (frames, POINTS_V2, 3) raw -> (FRAMES, POINTS_V2 * 3) in drawing units
    ls, rs = raw[:, P_SH_L, :2], raw[:, P_SH_R, :2]
    ok = ~np.isnan(ls).any(1) & ~np.isnan(rs).any(1)
    width = np.linalg.norm(ls - rs, axis=1)
    ok &= np.nan_to_num(width) > 1e-3
    if ok.sum() < 3:
        return None
    centre, scale = np.median(((ls + rs) / 2)[ok], axis=0), float(np.median(width[ok]))
    k = SHOULDER_UNITS / scale

    present = {name: ~np.isnan(raw[:, b, 0]).all(axis=1) for name, b in BLOCKS.items()}
    detected = ~np.isnan(raw[..., :2]).any(-1)
    pts = np.empty_like(raw)
    pts[..., :2] = (raw[..., :2] - centre) * k + np.array([BOX / 2, CHEST_Y])
    pts[..., 2] = raw[..., 2] * k
    pts = np.where(detected[..., None], np.nan_to_num(pts), 0.0)
    for name in ("left", "right", "pose", "face"):
        pts = hold_last(pts, BLOCKS[name], present[name])
    seen = np.any(pts[..., :2] != 0.0, axis=-1)

    # body depth: shoulders and hips near the chest plane (their MediaPipe depth is noisy
    # and twists the torso), arms from their bone lengths
    body = pts[:, BLOCKS["pose"]].copy()
    chest = np.nanmedian((body[:, 4, 2] + body[:, 5, 2]) / 2)
    for j in (4, 5, 10, 11):                                   # shoulders, hips
        body[:, j, 2] = np.clip(smooth((body[:, j, 2] - chest)[:, None], Z_SMOOTH)[:, 0], -3, 3)
    for sh, el, wr in ((4, 6, 8), (5, 7, 9)):
        body[:, el, 2] = body[:, sh, 2] + bone_depth(body[:, sh], body[:, el], UPPER_ARM)
        body[:, wr, 2] = body[:, el, 2] + bone_depth(body[:, el], body[:, wr], FOREARM)
    # the face sits in front of the shoulder line: a nose is ~10 units ahead of it
    for j in (0, 1, 2, 3):
        body[:, j, 2] = (body[:, 4, 2] + body[:, 5, 2]) / 2 - 10.0
    pts[:, BLOCKS["pose"]] = body

    # hands from their wrist, face from its nose tip, onto the body's points
    for block, wrist in ((BLOCKS["left"], P_WR_L), (BLOCKS["right"], P_WR_R)):
        rel = pts[:, block, 2] - pts[:, block.start:block.start + 1, 2]
        pts[:, block, 2] = smooth(rel, Z_SMOOTH) + pts[:, wrist:wrist + 1, 2]
    rel = pts[:, BLOCKS["face"], 2] - pts[:, F_NOSE_TIP:F_NOSE_TIP + 1, 2]
    pts[:, BLOCKS["face"], 2] = smooth(rel, Z_SMOOTH) + pts[:, P_NOSE:P_NOSE + 1, 2]
    pts[..., 2] = np.where(seen, pts[..., 2], 0.0)

    out = resample_clip(pts.reshape(len(pts), -1), FRAMES).reshape(FRAMES, POINTS_V2, 3)
    out[..., :2] = np.clip(out[..., :2], 0.0, BOX)
    return out.reshape(FRAMES, -1)

def hand_coverage(raw: np.ndarray) -> float:
    seen = (~np.isnan(raw[:, BLOCKS["left"], 0]).all(1)) | (~np.isnan(raw[:, BLOCKS["right"], 0]).all(1))
    return float(seen.mean())

def prepare(rel_path: str):
    try:
        raw = load_raw(rel_path)
    except Exception:
        return None
    if len(raw) < 4 or hand_coverage(raw) < MIN_HAND_COVERAGE:
        return None
    return to_drawing(raw)"""),
    (PY, r"""# ── choose the most typical recording of each sign ───────────────────────
meta = pd.read_csv(DATA_DIR / "train.csv")
signs = sorted(meta["sign"].unique())
if SUBSET_SIGNS:
    signs = signs[:SUBSET_SIGNS]
rng = np.random.default_rng(SEED)

def pick_for_sign(sign: str):
    rows = meta[meta["sign"] == sign]
    if len(rows) > CANDIDATES:
        rows = rows.iloc[rng.choice(len(rows), CANDIDATES, replace=False)]
    clips = [c for c in (prepare(p) for p in rows["path"]) if c is not None]
    if not clips:
        return None
    stack = np.stack(clips)
    # the medoid: the clip closest to the average of all of them
    distance = np.linalg.norm(stack.reshape(len(stack), -1) - stack.mean(0).ravel(), axis=1)
    return stack[int(np.argmin(distance))], len(clips)

t0 = time.time()
library, skipped = {}, []
with Pool(max(1, os.cpu_count() or 1)) as pool:
    for i, (sign, result) in enumerate(zip(signs, pool.imap(pick_for_sign, signs)), 1):
        if result is None:
            skipped.append(sign)
            continue
        clip, n = result
        library[sign.upper()] = {"frames": np.round(clip, 1).tolist(), "clips_examined": n}
        if i % 50 == 0 or i == len(signs):
            print(f"  {i}/{len(signs)} signs  {time.time() - t0:5.0f}s", flush=True)
print(f"\n{len(library)} signs ready in {time.time() - t0:.0f}s")
if skipped:
    print("no usable recording for:", ", ".join(skipped))"""),
    (PY, r"""# ── save the library ─────────────────────────────────────────────────────
import gzip
from datetime import datetime, timezone

path = OUT / "asl.json.gz"
with gzip.open(path, "wt", encoding="utf-8") as fh:
    json.dump({"version": 2, "language": "asl", "fps": FPS, "frames_per_sign": FRAMES,
               "face_indices": list(FACE_INDICES),
               "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
               "source": "Google ASL Signs (Kaggle asl-signs) — most typical training clip per sign",
               "signbridge": sign_bridge.__version__,
               "signs": library}, fh)
print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(library)} signs)")

# reload it exactly as the app does, as a check
from sign_bridge.ui.animation import SignLibrary, sign_svg
from sign_bridge.ui.figure3d import sign_3d_html
check = SignLibrary.load(path)
print("loads in the app:", len(check), "signs | 3D:", check.has_depth)
example = check.get(sorted(library)[0])
print("example:", example.gloss, "|", len(example.frames), "frames |",
      f"{example.seconds:.1f}s |", len(sign_svg(example)), "bytes of SVG |",
      len(sign_3d_html(example)) // 1000, "kB of 3D page")"""),
    (MD, r"""## Preview
A few signs drawn the way the app will draw them — check that they look like a person
signing before you download the file."""),
    (PY, r"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sign_bridge.ui.animation import LAYOUT_V2

HAND = ([0, 5, 9, 13, 17, 0], [0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
        [13, 14, 15, 16], [17, 18, 19, 20])
BODY = ([4, 6, 8], [5, 7, 9], [4, 5], [4, 10, 11, 5])
FACE_COLOUR = {"lips": "#A8731E"}

def draw(ax, frame, axes=(0, 1)):
    # axes=(0, 1): front view (x, y); axes=(2, 1): side view (z, y)
    p = np.asarray(frame).reshape(-1, 3)
    def line(idx, colour, lw):
        q = p[list(idx)]
        if np.all(q[:, :2] != 0):
            ax.plot(q[:, axes[0]], -q[:, axes[1]], color=colour, lw=lw, solid_capstyle="round")
    for chain in BODY:
        line([42 + i for i in chain], "#181611", 1.6)
    for start in (0, 21):
        for chain in HAND:
            line([start + i for i in chain], "#1E5A45", 1.2)
    for name, chains in LAYOUT_V2.face:
        for chain in chains:
            line(chain, FACE_COLOUR.get(name, "#3A352C"), 0.9 if name != "oval" else 1.1)
    lo = -50 if axes[0] == 2 else 0
    ax.set_xlim(lo, lo + BOX); ax.set_ylim(-BOX, 0); ax.set_aspect("equal"); ax.axis("off")

show = [s for s in ("MOM", "DAD", "YES", "NO", "DRINK", "AIRPLANE") if s in library][:4]
show = show or sorted(library)[:4]
steps = [0, FRAMES // 3, 2 * FRAMES // 3, FRAMES - 1]
fig, axes = plt.subplots(len(show), len(steps) + 1, figsize=(2.2 * (len(steps) + 1), 2.4 * len(show)))
axes = np.atleast_2d(axes)
for r, sign in enumerate(show):
    for c, step in enumerate(steps):
        draw(axes[r, c], library[sign]["frames"][step])
        if c == 0:
            axes[r, c].set_title(sign, loc="left", fontsize=11)
    draw(axes[r, -1], library[sign]["frames"][FRAMES // 2], axes=(2, 1))
    axes[r, -1].set_title("side view", fontsize=8)
fig.suptitle("How each sign starts, moves and ends (frames over time →), plus a side view", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "animation_preview.png", dpi=130)
print("preview written to animation_preview.png")"""),
    (MD, r"""## Done
1. Download **`asl.json.gz`** from the **Output** panel.
2. Put it in the project at **`assets/sign_animations/asl.json.gz`**.
3. Restart `sign-bridge` — the **How to sign** box shows the 3D figure.

Check `animation_preview.png` first. In the **side view** (last column) the hands should
be in front of the body (to the left, towards the camera) and the face upright. Arm depth
comes from bone lengths (`UPPER_ARM`, `FOREARM` in the conversion cell), not from
MediaPipe's arm depth, which is unreliable."""),
]


def code_cells() -> list[str]:
    """Just the code cells, in order (used by the tests)."""
    return [src for kind, src in CELLS if kind == PY]


def build(path: Path) -> Path:
    """Write the notebook to ``path``."""
    cells = [{"cell_type": kind, "metadata": {},
              **({"source": src.splitlines(keepends=True)} if kind == MD else
                 {"source": src.splitlines(keepends=True), "outputs": [], "execution_count": None})}
             for kind, src in CELLS]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    print("wrote", build(Path(__file__).with_name("make_sign_animations.ipynb")))
