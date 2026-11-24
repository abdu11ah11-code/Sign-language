"""Generate ``tune_segmenter.ipynb`` — measure and tune live sign segmentation.

Run ``python notebooks/build_tuning_notebook.py`` to (re)write the notebook.

The v0.4 live-pipeline check showed the deployed app scoring far below the
model's offline accuracy, and that most of the loss happens **before** the
model: the segmenter decides where a sign starts and ends, and when it cuts in
the wrong place the model never gets a fair chance.

This notebook finds better settings from data instead of guesswork:

1. stream real test clips past :class:`SignSegmenter` alone (no model — fast)
   and score every candidate setting on how well it finds exactly one sign and
   how closely that sign matches the true clip;
2. run the best few settings through the **whole** app pipeline, model
   included, and report correct / wrong / not sure / missed;
3. print the ``config.yaml`` block for the winner.
"""

from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
PY = "code"

CELLS: list[tuple[str, str]] = [
    (MD, r"""# SignBridge — tune the live sign segmenter

The app does not get tidy one-sign clips: it watches a stream and has to decide **when a
sign starts and ends**. The v0.4 measurement showed that this step, not the model, is
where most live accuracy is lost — the model scored **69.1%** on clips it was handed
directly, but only **31.0%** when the app had to find the signs itself.

This notebook measures the segmenter on real clips and picks better settings.

| | |
|---|---|
| **Stage 1** | Segmenter only, no model — thousands of settings scored in seconds on *did it find exactly one sign, and did that sign cover the real one?* |
| **Stage 2** | The best settings run through the full app pipeline (model included) for correct / wrong / not sure / missed |
| **Output** | `segmenter_tuning.csv` and a ready-to-paste `config.yaml` block |

### Setup
1. **+ Add Input** → the competition **Google - Isolated Sign Language Recognition**.
2. **Upload** → your SignBridge project zip **and** `signbridge_asl_models.zip`
   (upload the models zip as its own dataset so the trained model is available here).
3. **Settings → Accelerator: GPU** (stage 2 is much faster with it).
4. **Run All** — a few minutes."""),
    (PY, r"""# ── settings ─────────────────────────────────────────────────────────────
import os
from pathlib import Path

N_CLIPS = int(os.environ.get("SB_TCLIPS", 800))   # test clips streamed past the segmenter
LIVE_FPS = float(os.environ.get("SB_TFPS", 30))    # frame rate the stream is replayed at
LEAD_BLANK, TAIL_BLANK = 15, 45                    # empty frames before/after, as if hands rested
TEST_PARTICIPANTS, VAL_PARTICIPANTS = 4, 2         # same split as training
STAGE2_CONFIGS = int(os.environ.get("SB_TOP", 4))  # settings taken through the full pipeline
SEED = 42

DATA_DIR = Path(os.environ.get("SB_DATA_DIR", "/kaggle/input/asl-signs"))
SRC_SEARCH = Path(os.environ.get("SB_SRC", "/kaggle/input"))
OUT = Path(os.environ.get("SB_OUT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)

if not (DATA_DIR / "train.csv").is_file():
    hits = [c.parent for pat in ("*/train.csv", "*/*/train.csv", "*/*/*/train.csv")
            for c in sorted(SRC_SEARCH.glob(pat)) if (c.parent / "train_landmark_files").is_dir()]
    if not hits:
        raise FileNotFoundError(
            "ASL Signs data not found. Add Input → Competitions → "
            "'Google - Isolated Sign Language Recognition'.")
    DATA_DIR = hits[0]
print("data:", DATA_DIR, "| clips:", N_CLIPS, "| replay fps:", LIVE_FPS)"""),
    (PY, r"""# ── make the SignBridge code importable ──────────────────────────────────
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
    LIP_LANDMARK_INDICES, POSE_LANDMARK_INDICES, RAW_DIM, HolisticResult)
from sign_bridge.config import load_config
from sign_bridge.inference.segmenter import SegmenterSettings, SignSegmenter, hand_speed

import logging

from sign_bridge.logging_conf import configure_logging

configure_logging(logging.ERROR)   # the pipeline logs every locked sign — far too noisy here

CURRENT = load_config().inference
print("SignBridge", sign_bridge.__version__)
print("current settings:", {k: getattr(CURRENT, k) for k in
      ("motion_start_speed", "motion_stop_speed", "hold_seconds",
       "min_sign_seconds", "max_sign_seconds", "segment_confidence", "segment_margin")})"""),
    (PY, r"""# ── load raw test clips (unseen signers only) ────────────────────────────
from multiprocessing import Pool

TYPE_OFFSET = {"face": 0, "left_hand": 468, "pose": 489, "right_hand": 522}
SEL = np.concatenate([468 + np.arange(21), 522 + np.arange(21),
                      489 + np.array(POSE_LANDMARK_INDICES), np.array(LIP_LANDMARK_INDICES)])

def load_raw(rel_path: str) -> np.ndarray:
    df = pd.read_parquet(DATA_DIR / rel_path)
    idx_col = "landmark_index" if "landmark_index" in df.columns else "landmark_idx"
    g = df["type"].map(TYPE_OFFSET).to_numpy() + df[idx_col].to_numpy()
    frames, f_idx = np.unique(df["frame"].to_numpy(), return_inverse=True)
    arr = np.full((len(frames), 543, 3), np.nan, dtype=np.float32)
    arr[f_idx, g] = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return np.nan_to_num(arr[:, SEL, :], nan=0.0).reshape(len(frames), RAW_DIM).astype(np.float32)

meta = pd.read_csv(DATA_DIR / "train.csv")
sign_map_path = DATA_DIR / "sign_to_prediction_index_map.json"
signs_by_index = (sorted(json.loads(sign_map_path.read_text()), key=json.loads(sign_map_path.read_text()).get)
                  if sign_map_path.is_file() else sorted(meta["sign"].unique()))
LABELS = [s.upper() for s in signs_by_index]
class_of = {s: i for i, s in enumerate(signs_by_index)}

participants = np.array(sorted(meta["participant_id"].unique()))
rng = np.random.default_rng(SEED)
participants = participants[rng.permutation(len(participants))]
test_p = participants[:TEST_PARTICIPANTS]
test_rows = meta[meta["participant_id"].isin(test_p)].reset_index(drop=True)
take = rng.choice(len(test_rows), size=min(N_CLIPS, len(test_rows)), replace=False)
test_rows = test_rows.iloc[take].reset_index(drop=True)

t0 = time.time()
with Pool(max(1, os.cpu_count() or 1)) as pool:
    CLIPS = list(pool.imap(load_raw, test_rows["path"], chunksize=16))
TRUTH = np.array([class_of[s] for s in test_rows["sign"]])
lengths = np.array([len(c) for c in CLIPS])
print(f"{len(CLIPS)} clips from signers {list(test_p)} in {time.time() - t0:.0f}s")
print(f"clip length: median {np.median(lengths):.0f} frames "
      f"({np.median(lengths) / LIVE_FPS:.1f}s), 5-95%: {np.percentile(lengths, 5):.0f}"
      f"-{np.percentile(lengths, 95):.0f} frames "
      f"({np.percentile(lengths, 5) / LIVE_FPS:.1f}-{np.percentile(lengths, 95) / LIVE_FPS:.1f}s)")
print(f"longest clip: {lengths.max()} frames ({lengths.max() / LIVE_FPS:.1f}s)")"""),
    (MD, r"""## Stage 1 — the segmenter alone
Each clip is replayed as a stream: empty frames, the sign, empty frames again. For every
candidate setting we record how many signs the segmenter found and how well the one it
found lines up with the real sign (overlap = intersection over union, 1.0 = perfect).

Both length limits matter. `min_sign_seconds` throws away anything shorter as a twitch, and
`max_sign_seconds` cuts anything longer in half — compare the clip lengths printed above with
the values swept below."""),
    (PY, r"""# ── stage 1: sweep settings, no model needed ─────────────────────────────
import itertools

STREAMS = []
for clip in CLIPS:
    n = len(clip)
    blank = np.zeros((1, RAW_DIM), np.float32)
    stream = np.concatenate([np.repeat(blank, LEAD_BLANK, 0), clip,
                             np.repeat(blank, TAIL_BLANK, 0)])
    visible = np.array([bool(np.any(f[:126])) for f in stream])
    STREAMS.append((stream, visible, (LEAD_BLANK, LEAD_BLANK + n)))

# Hand speed depends only on the recording, never on the settings, so measure it once per
# clip and hand it to the segmenter (`speed=`) instead of re-measuring it for all 324
# settings. The segments come out identical — there is a test for that — but the sweep
# runs in minutes instead of an hour.
t0 = time.time()
SPEEDS = []
for stream, visible, _ in STREAMS:
    speeds, prev_raw, prev_t = [], None, None
    for k, frame in enumerate(stream):
        t = k / LIVE_FPS
        dt = 0.0 if prev_t is None else t - prev_t
        speeds.append(hand_speed(prev_raw, frame, dt) if (visible[k] and dt > 0) else 0.0)
        prev_raw, prev_t = (frame if visible[k] else None), t
    SPEEDS.append(speeds)
print(f"hand speed measured for {len(STREAMS)} clips in {time.time() - t0:.0f}s")

def run_segmenter(settings) -> dict:
    # how many signs found per clip, and how well the first one covers the real sign
    counts, ious = [], []
    for (stream, visible, (lo, hi)), speeds in zip(STREAMS, SPEEDS):
        seg = SignSegmenter(settings)
        found = []
        for k, frame in enumerate(stream):
            out = seg.update(k / LIVE_FPS, frame, bool(visible[k]), k, speed=speeds[k])
            if out is not None:
                found.append((out.frames[0], out.frames[-1]))
        counts.append(len(found))
        if found:
            a, b = found[0]
            inter = max(0, min(b, hi) - max(a, lo))
            union = max(b, hi) - min(a, lo)
            ious.append(inter / union if union else 0.0)
        else:
            ious.append(0.0)
    counts = np.array(counts)
    return {"one_segment": float((counts == 1).mean()), "missed": float((counts == 0).mean()),
            "split": float((counts > 1).mean()), "iou": float(np.mean(ious)),
            "iou_good": float(np.mean(np.array(ious) > 0.7))}

# min_seconds matters as much as max_seconds: the clip lengths printed above show many
# real signs last well under half a second, and anything shorter than min_seconds is thrown
# away as a twitch. Both ends of the length range are swept.
GRID = {
    "start_speed": [0.4, 0.6, 0.9],
    "stop_speed": [0.15, 0.3],
    "hold_seconds": [0.4, 0.6],
    "min_seconds": [0.05, 0.15],
    "max_seconds": [6.0, 12.0],
    "pre_roll_seconds": [0.15],
    "merge_gap_seconds": [0.0, 0.3, 0.6],
}
keys = list(GRID)
rows = []
t0 = time.time()
for i, values in enumerate(itertools.product(*GRID.values()), 1):
    cfg = dict(zip(keys, values))
    settings = SegmenterSettings(drop_seconds=min(CURRENT.hand_grace_seconds, 0.3), **cfg)
    rows.append({**cfg, **run_segmenter(settings)})
    if i % 27 == 0:
        print(f"  {i}/{len(list(itertools.product(*GRID.values())))} settings tried"
              f"  {time.time() - t0:5.0f}s", flush=True)

tuning = pd.DataFrame(rows)
# a good setting finds exactly one sign and covers it well
tuning["score"] = tuning["one_segment"] * tuning["iou"]
tuning = tuning.sort_values("score", ascending=False).reset_index(drop=True)
tuning.to_csv(OUT / "segmenter_tuning.csv", index=False)

base = SegmenterSettings(
    start_speed=CURRENT.motion_start_speed, stop_speed=CURRENT.motion_stop_speed,
    hold_seconds=CURRENT.hold_seconds, drop_seconds=min(CURRENT.hand_grace_seconds, 0.3),
    min_seconds=CURRENT.min_sign_seconds, max_seconds=CURRENT.max_sign_seconds,
    merge_gap_seconds=CURRENT.merge_gap_seconds)
current_row = run_segmenter(base)
print(f"\n{len(tuning)} settings tried in {time.time() - t0:.0f}s")
print("\ncurrent settings:", {k: round(v, 3) for k, v in current_row.items()})
print("\nbest 10:")
print(tuning.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))"""),
    (MD, r"""## Stage 2 — the best settings through the whole app
Stage 1 only asks whether the segmenter cut in the right place. Here the model is loaded
and the full `SignToTextPipeline` runs, so we see what the user would actually get."""),
    (PY, r"""# ── stage 2: full pipeline with the trained model ────────────────────────
import keras
from sign_bridge.inference.sign_to_text import SignToTextPipeline
from sign_bridge.models.registry import register_custom_objects

register_custom_objects()
import joblib

def find_model(root: Path):
    # the newest exported model + its labels and scaler, from the uploaded models zip
    pointers = sorted(root.rglob("asl_latest.json"))
    if not pointers:
        raise FileNotFoundError(
            "No asl_latest.json found. Upload signbridge_asl_models.zip as a dataset "
            "and add it as an input.")
    info = json.loads(pointers[-1].read_text())
    folder = pointers[-1].parent
    stem = info["stem"]
    model = keras.models.load_model(folder / f"{stem}.keras", compile=False)
    labels = json.loads((folder / f"{stem}.labels.json").read_text())
    scaler = joblib.load(folder / f"{stem}.scaler.joblib")
    return model, labels, scaler, info

model, labels, scaler, info = find_model(SRC_SEARCH)
print("model:", info["architecture"], "| features:", info["feature_mode"],
      "| classes:", len(labels))
cfg = load_config()

def run_full(settings, confidence: float, margin: float) -> dict:
    pipe = SignToTextPipeline(model, labels, scaler, cfg,
                              feature_mode=info["feature_mode"], mode="segment")
    pipe._segmenter = SignSegmenter(settings)
    pipe.LIVE_EVERY = float("inf")   # count only answers given when a sign ends
    object.__setattr__(pipe._cfg.inference, "segment_confidence", confidence)
    object.__setattr__(pipe._cfg.inference, "segment_margin", margin)
    status = []
    for (stream, visible, _), truth in zip(STREAMS, TRUTH):
        pipe.reset()
        pipe._segmenter = SignSegmenter(settings)
        answers, segments = [], 0
        for k, frame in enumerate(stream):
            left, right = bool(np.any(frame[:63])), bool(np.any(frame[63:126]))
            out = pipe.process_frame(
                HolisticResult(features=frame, hands_visible=left or right,
                               left_hand_visible=left, right_hand_visible=right, raw=None),
                now=k / LIVE_FPS)
            if out.latency_ms is not None:
                segments += 1
            if out.is_locked:
                answers.append(out.locked_sign)
        if segments == 0:
            status.append("missed")
        elif not answers:
            status.append("not sure")
        elif answers == [labels[truth]]:
            status.append("correct")
        else:
            status.append("wrong")
    s = pd.Series(status).value_counts(normalize=True)
    return {k: float(s.get(k, 0.0)) for k in ("correct", "wrong", "not sure", "missed")}

candidates = [("current", base, CURRENT.segment_confidence, CURRENT.segment_margin)]
for i in range(min(STAGE2_CONFIGS, len(tuning))):
    row = tuning.iloc[i]
    candidates.append((
        f"#{i + 1}",
        SegmenterSettings(start_speed=row.start_speed, stop_speed=row.stop_speed,
                          hold_seconds=row.hold_seconds, min_seconds=row.min_seconds,
                          max_seconds=row.max_seconds, pre_roll_seconds=row.pre_roll_seconds,
                          merge_gap_seconds=row.merge_gap_seconds,
                          drop_seconds=min(CURRENT.hand_grace_seconds, 0.3)),
        CURRENT.segment_confidence, CURRENT.segment_margin))

results = []
t0 = time.time()
for name, settings, conf, margin in candidates:
    out = run_full(settings, conf, margin)
    results.append({"settings": name, **out,
                    "start": settings.start_speed, "stop": settings.stop_speed,
                    "hold": settings.hold_seconds, "min": settings.min_seconds,
                    "max": settings.max_seconds, "merge": settings.merge_gap_seconds})
    print(f"  {name:>8}: correct {out['correct']:.1%}  wrong {out['wrong']:.1%}  "
          f"not sure {out['not sure']:.1%}  missed {out['missed']:.1%}"
          f"   ({time.time() - t0:.0f}s)", flush=True)

full = pd.DataFrame(results).sort_values("correct", ascending=False)
full.to_csv(OUT / "segmenter_full_pipeline.csv", index=False)
print("\n" + full.to_string(index=False, float_format=lambda v: f"{v:.3f}"))"""),
    (PY, r"""# ── the winner, ready to paste into config.yaml ──────────────────────────
best = full.iloc[0]
current_correct = float(full[full["settings"] == "current"]["correct"].iloc[0])
gain = best["correct"] - current_correct
print(f"best: {best['settings']} — correct {best['correct']:.1%} "
      f"(current {current_correct:.1%}, {gain:+.1%})\n")
if gain <= 0.005:
    print("The current settings are already as good as anything tried — keep them.")
else:
    print("Paste this into config.yaml under `inference:`\n")
    print(f"  motion_start_speed: {best['start']}")
    print(f"  motion_stop_speed: {best['stop']}")
    print(f"  hold_seconds: {best['hold']}")
    print(f"  min_sign_seconds: {best['min']}")
    print(f"  max_sign_seconds: {best['max']}")
    print(f"  merge_gap_seconds: {best['merge']}")
"""),
    (MD, r"""## What to do with this
* Send Claude `segmenter_tuning.csv`, `segmenter_full_pipeline.csv` and the printed block.
* Paste the `config.yaml` lines into the project and restart the app.
* These numbers are measured on clips from **signers the model never saw**, so they are
  honest and quotable — but they are still pre-recorded phone clips, not your webcam."""),
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
    print("wrote", build(Path(__file__).with_name("tune_segmenter.ipynb")))
