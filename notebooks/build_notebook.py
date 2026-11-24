"""Generate ``train_asl_signs_kaggle.ipynb`` from the cell list below.

Run ``python notebooks/build_notebook.py`` after editing a cell. Keeping the
notebook as code makes it reviewable and lets the test-suite execute the
exact same cells on a small synthetic dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
PY = "code"

CELLS: list[tuple[str, str]] = [
    (MD, r"""# SignBridge — ASL sign recognition: LSTM vs Transformer vs Conv1D-Transformer

Trains three architectures on **Google's ASL Signs dataset** (Kaggle competition
*Google – Isolated Sign Language Recognition*: 250 signs, ~94k clips, MediaPipe Holistic
landmarks) under **one identical, signer-independent protocol**, and exports every model in
SignBridge's format so the live app can use it directly.

| | |
|---|---|
| **Features** | `holistic_lips_clip` — both hands, 12 upper-body points and 40 lip points, 2D → 188 values per frame. Normalized per clip: centred on the chest, scaled by shoulder width, and corrected for the camera's shape (phone video vs laptop webcam). Identical code to the live app. |
| **Window** | every clip resampled to `WINDOW` frames |
| **Split** | by **participant**: test and validation signers never appear in training |
| **Training** | identical for all three models: AdamW + cosine decay over the full `EPOCHS`, label smoothing, augmentation; the best validation epoch is kept |
| **Reported** | top-1, top-5, macro-F1 on unseen signers, parameter count, CPU latency, and a **live-pipeline check**: real test clips streamed frame by frame through the app's own recognizer |

### Setup on Kaggle (once)
1. Open the competition page **Google - Isolated Sign Language Recognition** and click **Join / I understand and accept** (needed to attach its data).
2. Upload the SignBridge project zip as a **Kaggle Dataset** (Datasets → New Dataset → upload `sign_bridge_*.zip`), e.g. named `signbridge-src`.
3. Create a new Notebook → **File → Import Notebook** → this `.ipynb`.
4. **Add Input**: the competition `asl-signs` and your `signbridge-src` dataset.
5. **Settings → Accelerator: GPU** (T4 ×2 or P100). Internet is not required.
6. **Run All.** The first run converts the data (~15–30 min, cached afterwards); training takes roughly 1–3 h for all three models at the default settings. Lower `EPOCHS` or set `SUBSET_SIGNS = 50` for a quick trial.
7. Download `signbridge_asl_models.zip` from the **Output** panel and unzip it into SignBridge's `models/` folder.

### Optional: more signers from ASL Citizen ("run B")
If an **ASL Citizen import** is attached as an input too (the `data/asl_citizen` folder made by
`sign-import-aslcitizen` and checked by `sign-check-aslcitizen`, uploaded as a **private** dataset —
its licence allows non-commercial research only and no redistribution), its *train* and *val*
signers are added to training, only for the glosses the check kept. Validation and the main test
stay the same ASL Signs signers, so the scores compare directly with a run without it; ASL
Citizen's own *test* signers are reported as a second, independent test.
"""),
    (PY, r"""# ── settings ─────────────────────────────────────────────────────────────
import os
from pathlib import Path

WINDOW = int(os.environ.get("SB_WINDOW", 64))          # frames per clip given to the model
WINDOW_SECONDS = 1.5      # live window the app uses with these models (ASL Signs clips hold one sign)
ARCHS = os.environ.get("SB_ARCHS", "lstm,transformer,conv_transformer").split(",")
EPOCHS = int(os.environ.get("SB_EPOCHS", 100))
# Stronger augmentation: also hide a hand or the lips now and then, and blank a few
# frames — what a webcam does when tracking drops out. 0 = as in runs A and B.
STRONG_AUG = os.environ.get("SB_STRONG_AUG", "0") == "1"
# Motion features of the conv-transformer: frame gaps (1 = frame to frame; 1,2 adds 2-frame change)
MOTION_LAGS = [int(v) for v in os.environ.get("SB_MOTION_LAGS", "1").split(",")]
BATCH = int(os.environ.get("SB_BATCH", 128))
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.1
SUBSET_SIGNS = int(os.environ["SB_SUBSET"]) if os.environ.get("SB_SUBSET") else None  # e.g. 50
TEST_PARTICIPANTS = int(os.environ.get("SB_TEST_P", 4))  # signers held out for the final test
VAL_PARTICIPANTS = int(os.environ.get("SB_VAL_P", 2))    # signers held out for early stopping
SEED = 42          # the signer split and the live-check clips: never change, so runs compare
# Seed of the model's start weights, shuffling and augmentation. A second model trained the
# same way with another seed makes different mistakes, so the two average into a better one.
TRAIN_SEED = int(os.environ.get("SB_TRAIN_SEED", SEED))

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
print("data:", DATA_DIR, "| output:", OUT, "| window:", WINDOW, "| epochs:", EPOCHS,
      "| models:", ARCHS, "| strong aug:", STRONG_AUG, "| motion lags:", MOTION_LAGS,
      "| training seed:", TRAIN_SEED)

# Optional ASL Citizen import (a folder holding manifest.csv + check.json), see above.
ASLC_DIR = Path(os.environ["SB_ASLC"]) if os.environ.get("SB_ASLC") else next(
    (m.parent for pat in ("*/manifest.csv", "*/*/manifest.csv", "*/*/*/manifest.csv", "*/*/*/*/manifest.csv")
     for m in sorted(SRC_SEARCH.glob(pat)) if (m.parent / "check.json").is_file()), None)
print("ASL Citizen:", ASLC_DIR or "not attached (training on ASL Signs only)")
"""),
    (PY, r"""# ── make the SignBridge code importable (same preprocessing + models as the live app) ──
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
SKIP_TOP = {"data", "models", "logs", "recordings", ".venv", "assets"}  # large / not needed

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

import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
import sign_bridge
from sign_bridge.capture.holistic import LIP_LANDMARK_INDICES, POSE_LANDMARK_INDICES, RAW_DIM
from sign_bridge.config import load_config
from sign_bridge.data.augment import LAYOUTS, mirror_permutation
from sign_bridge.data.preprocess import fit_scaler, mode_dim, normalize_window, resample_clip
from sign_bridge.models.registry import build_from_spec, default_params, make_spec
from sign_bridge.models.conv_transformer import SparseSmoothedCrossentropy

tf.keras.utils.set_random_seed(TRAIN_SEED)
FEATURE_MODE = "holistic_lips_clip"
FEAT = mode_dim(FEATURE_MODE)
print("SignBridge", sign_bridge.__version__, "| TF", tf.__version__, "| Keras", keras.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU") or "none (training will be slow)")
"""),
    (PY, r"""# ── dataset index ──────────────────────────────────────────────────────────
meta = pd.read_csv(DATA_DIR / "train.csv")
sign_map = json.loads((DATA_DIR / "sign_to_prediction_index_map.json").read_text()) \
    if (DATA_DIR / "sign_to_prediction_index_map.json").is_file() else \
    {s: i for i, s in enumerate(sorted(meta["sign"].unique()))}
signs_by_index = sorted(sign_map, key=sign_map.get)
if SUBSET_SIGNS:
    keep = set(signs_by_index[:SUBSET_SIGNS])
    meta = meta[meta["sign"].isin(keep)].reset_index(drop=True)
    signs_by_index = [s for s in signs_by_index if s in keep]
LABELS = [s.upper() for s in signs_by_index]          # SignBridge glosses are upper case
class_of = {s: i for i, s in enumerate(signs_by_index)}
meta["y"] = meta["sign"].map(class_of)
print(f"{len(meta):,} clips | {len(LABELS)} signs | {meta['participant_id'].nunique()} participants")
"""),
    (MD, r"""## Landmarks → SignBridge features
Each parquet file holds 543 landmarks per frame (face 468, left hand 21, pose 33, right hand 21;
missing = NaN). We keep the 94 points SignBridge uses live, in its raw layout
(left hand, right hand, 12 pose points, 40 lip points), set missing points to 0 exactly as the
live tracker does, resample to `WINDOW` frames and apply SignBridge's own `holistic_lips_clip`
normalization. Converted arrays are cached, so re-runs skip this step."""),
    (PY, r"""from multiprocessing import Pool

TYPE_OFFSET = {"face": 0, "left_hand": 468, "pose": 489, "right_hand": 522}
SEL = np.concatenate([
    468 + np.arange(21),                         # left hand
    522 + np.arange(21),                         # right hand
    489 + np.array(POSE_LANDMARK_INDICES),       # 12 upper-body points
    np.array(LIP_LANDMARK_INDICES),              # 40 lip points (face mesh)
])
assert SEL.size * 3 == RAW_DIM

def load_raw(rel_path: str) -> np.ndarray:
    # (frames, 282) raw landmarks in SignBridge's live layout; missing points = 0
    df = pd.read_parquet(DATA_DIR / rel_path)
    idx_col = "landmark_index" if "landmark_index" in df.columns else "landmark_idx"
    g = df["type"].map(TYPE_OFFSET).to_numpy() + df[idx_col].to_numpy()
    frames, f_idx = np.unique(df["frame"].to_numpy(), return_inverse=True)
    arr = np.full((len(frames), 543, 3), np.nan, dtype=np.float32)
    arr[f_idx, g] = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return np.nan_to_num(arr[:, SEL, :], nan=0.0).reshape(len(frames), RAW_DIM)

def load_clip(rel_path: str):
    raw = load_raw(rel_path)
    feats = normalize_window(resample_clip(raw, WINDOW), FEATURE_MODE)
    return feats.astype(np.float16), len(raw)

cache = OUT / f"asl_signs_{FEATURE_MODE}_T{WINDOW}_{len(LABELS)}signs.npz"
if cache.is_file():
    z = np.load(cache)
    X, LENGTHS = z["X"], z["lengths"]
    print("loaded cache", cache.name, X.shape)
else:
    t0 = time.time()
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        out = []
        for i, item in enumerate(pool.imap(load_clip, meta["path"], chunksize=64), 1):
            out.append(item)
            if i % 5000 == 0 or i == len(meta):
                print(f"  {i:,}/{len(meta):,} clips  {time.time() - t0:5.0f}s", flush=True)
    X = np.stack([o[0] for o in out])
    LENGTHS = np.array([o[1] for o in out])
    np.savez(cache, X=X, lengths=LENGTHS)
    print("converted", X.shape, f"in {time.time() - t0:.0f}s")
Y = meta["y"].to_numpy()
print(f"frames per clip: median {np.median(LENGTHS):.0f}, 5-95%: {np.percentile(LENGTHS, 5):.0f}-{np.percentile(LENGTHS, 95):.0f}")
"""),
    (MD, r"""## Signer-independent split
Participants are shuffled once (fixed seed); `TEST_PARTICIPANTS` signers form the test set,
`VAL_PARTICIPANTS` the validation set, the rest train. No signer appears in two sets, so
scores measure generalization to **new people** — the situation the live app is in."""),
    (PY, r"""participants = np.array(sorted(meta["participant_id"].unique()))
rng = np.random.default_rng(SEED)
participants = participants[rng.permutation(len(participants))]
test_p = participants[:TEST_PARTICIPANTS]
val_p = participants[TEST_PARTICIPANTS:TEST_PARTICIPANTS + VAL_PARTICIPANTS]
train_p = participants[TEST_PARTICIPANTS + VAL_PARTICIPANTS:]
pid = meta["participant_id"].to_numpy()
tr, va, te = (np.isin(pid, p) for p in (train_p, val_p, test_p))
print(f"train {tr.sum():,} clips / {len(train_p)} signers | val {va.sum():,} / {len(val_p)} | "
      f"test {te.sum():,} / {len(test_p)}  (test signers: {list(test_p)})")
"""),
    (MD, r"""## ASL Citizen (only if attached)
ASL Citizen clips were tracked with the same MediaPipe settings and stored in the same raw layout,
so they get exactly the same normalization. Its train + val signers join **training**; its test
signers become a second test set. Glosses that `sign-check-aslcitizen` dropped (a different sign
filed under the same English word, e.g. a regional variant) are left out."""),
    (PY, r"""from sign_bridge.data.asl_citizen import training_clips

def to_features(raw_clips):
    return np.stack([normalize_window(resample_clip(c.astype(np.float32), WINDOW), FEATURE_MODE)
                     for c in raw_clips]).astype(np.float16) if len(raw_clips) else np.zeros((0, WINDOW, FEAT), np.float16)

if ASLC_DIR:
    aslc_train = training_clips(ASLC_DIR, LABELS, ("train", "val"))
    aslc_test = training_clips(ASLC_DIR, LABELS, ("test",))
    XC, YC = to_features(aslc_train.clips), aslc_train.labels
    XCT, YCT = to_features(aslc_test.clips), aslc_test.labels
    print(f"ASL Citizen: +{len(YC):,} training clips from {len(set(aslc_train.participants))} signers, "
          f"{len(set(YC.tolist()))} signs | test {len(YCT):,} clips from {len(set(aslc_test.participants))} signers")
else:
    XC, YC = np.zeros((0, WINDOW, FEAT), np.float16), np.zeros(0, np.int64)
    XCT, YCT = XC, YC
X_TRAIN = np.concatenate([X[tr], XC]); Y_TRAIN = np.concatenate([Y[tr], YC])
SCALER_SAMPLE = rng.choice(len(Y_TRAIN), size=min(20000, len(Y_TRAIN)), replace=False)
print(f"training clips: {len(Y_TRAIN):,} ({int(tr.sum()):,} ASL Signs + {len(YC):,} ASL Citizen)")
"""),
    (MD, r"""## Augmentation (GPU, on the fly)
Applied in normalized landmark space, before standardization, to training clips only:
horizontal flip (hands, symmetric pose and lip points swapped — the same permutation as the app),
rotation ±15°, scale ±15 %, aspect stretch ±20 % (phone video vs. 4:3 webcam), small shift,
and temporal crop/speed change. Undetected points stay exactly zero."""),
    (PY, r"""PERM = tf.constant(mirror_permutation(LAYOUTS[FEATURE_MODE]), tf.int32)
NPTS = FEAT // 2

def augment(x):
    pts = tf.reshape(tf.cast(x, tf.float32), (WINDOW, NPTS, 2))
    flip = tf.random.uniform(()) < 0.5
    pts = tf.cond(flip, lambda: tf.gather(pts, PERM, axis=1) * tf.constant([-1.0, 1.0]), lambda: pts)
    mask = tf.reduce_any(tf.not_equal(pts, 0.0), axis=-1, keepdims=True)
    theta = tf.random.uniform((), -15.0, 15.0) * np.pi / 180.0
    c, s = tf.cos(theta), tf.sin(theta)
    rot = tf.stack([tf.stack([c, -s]), tf.stack([s, c])])
    pts = tf.einsum("tpc,dc->tpd", pts, rot)
    sx = tf.random.uniform((), 0.85, 1.15)
    aspect = tf.random.uniform((), 0.8, 1.2)
    pts = pts * tf.stack([sx * aspect, sx]) + tf.random.uniform((1, 1, 2), -0.1, 0.1)
    pts = tf.where(mask, pts, 0.0)
    # temporal crop + speed change: resample a random sub-span back to WINDOW frames
    start = tf.random.uniform((), 0.0, 0.15) * (WINDOW - 1)
    end = tf.random.uniform((), 0.85, 1.0) * (WINDOW - 1)
    idx = tf.cast(tf.round(tf.linspace(start, end, WINDOW)), tf.int32)
    pts = tf.gather(pts, idx)
    if STRONG_AUG:
        # hide a whole part now and then (points 0-20 one hand, 21-41 the other, 54-93 lips);
        # hidden points are exactly zero, as the live tracker reports them
        keep = tf.concat([tf.fill([21], tf.random.uniform(()) > 0.12),
                          tf.fill([21], tf.random.uniform(()) > 0.12),
                          tf.fill([12], True),
                          tf.fill([NPTS - 54], tf.random.uniform(()) > 0.10)], 0)
        pts = pts * tf.cast(keep, tf.float32)[None, :, None]
        # blank a short run of frames (dropped tracking)
        span = tf.cast(tf.random.uniform((), 0, 0.12) * WINDOW, tf.int32)
        t0 = tf.random.uniform((), 0, WINDOW - span + 1, tf.int32)
        frames = tf.range(WINDOW)
        blank = (frames >= t0) & (frames < t0 + span) & (tf.random.uniform(()) < 0.5)
        pts = tf.where(blank[:, None, None], 0.0, pts)
    return tf.reshape(pts, (WINDOW, FEAT))

# Second-stage standardization (the same scaler artifact the app loads), fitted on
# *augmented* training clips so its statistics cover what the model sees; per-feature
# std is floored (shoulders are constant after normalization — dividing by their ~0
# spread would blow small changes up into huge values).
aug_sample = np.stack([augment(tf.constant(x)).numpy() for x in X_TRAIN[SCALER_SAMPLE[:5000]]])
scaler = fit_scaler(np.concatenate([X_TRAIN[SCALER_SAMPLE].astype(np.float32), aug_sample]))
MEAN = tf.constant(scaler.mean_, tf.float32)
SCALE = tf.constant(scaler.scale_, tf.float32)

def standardize(x):
    return (tf.cast(x, tf.float32) - MEAN) / SCALE

def make_ds(Xs, Ys, train: bool):
    ds = tf.data.Dataset.from_tensor_slices((Xs, Ys))
    if train:
        ds = ds.shuffle(min(len(Ys), 20000), seed=TRAIN_SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda x, y: (standardize(augment(x)), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (standardize(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_TRAIN, Y_TRAIN, train=True)
val_ds = make_ds(X[va], Y[va], train=False)
"""),
    (MD, r"""## Models
The three architectures come from SignBridge's model registry (the same code the app loads).
Capacity is set so each model is reasonably sized for 250 classes; all use identical training settings."""),
    (PY, r"""cfg = load_config()
PARAMS = {
    "lstm": {**default_params("lstm", cfg), "lstm_units": [256, 128], "dense_units": 256, "dropout": 0.4},
    "transformer": {**default_params("transformer", cfg), "d_model": 192, "num_heads": 4,
                    "ff_dim": 384, "num_blocks": 3, "dropout": 0.3},
    "conv_transformer": {**default_params("conv_transformer", cfg), "motion_lags": MOTION_LAGS},
}

def compile_same(model, steps_per_epoch: int):
    schedule = keras.optimizers.schedules.CosineDecay(
        0.0, decay_steps=max(1, EPOCHS * steps_per_epoch),
        warmup_target=LEARNING_RATE, warmup_steps=max(1, 2 * steps_per_epoch))
    model.compile(optimizer=keras.optimizers.AdamW(schedule, weight_decay=0.05),
                  loss=SparseSmoothedCrossentropy(LABEL_SMOOTHING),
                  metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5")])
    return model

steps = int(np.ceil(len(Y_TRAIN) / BATCH))
for arch in ARCHS:
    spec = make_spec(arch, len(LABELS), WINDOW, FEAT, PARAMS[arch])
    print(f"{arch:>17}: {build_from_spec(spec).count_params():,} parameters")
"""),
    (PY, r"""import gc

class KeepBest(keras.callbacks.Callback):
    # remember the weights of the epoch with the best validation accuracy
    def __init__(self):
        super().__init__()
        self.best, self.best_epoch, self.weights = -1.0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        acc = (logs or {}).get("val_accuracy", -1.0)
        if acc > self.best:
            self.best, self.best_epoch = acc, epoch + 1
            self.weights = self.model.get_weights()

    def restore(self, model):
        if self.weights is not None:
            model.set_weights(self.weights)

RESULTS, HISTORIES, MODELS, SPECS = {}, {}, {}, {}
for arch in ARCHS:
    print(f"\n━━ training {arch} ━━")
    keras.backend.clear_session(); gc.collect()
    tf.keras.utils.set_random_seed(TRAIN_SEED)
    spec = make_spec(arch, len(LABELS), WINDOW, FEAT, PARAMS[arch])
    model = compile_same(build_from_spec(spec), steps)
    t0 = time.time()
    keep = KeepBest()
    # Every model runs the full cosine schedule (no early stopping, so no model is
    # cut off while its learning rate is still high); the best validation epoch is kept.
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2, callbacks=[keep])
    keep.restore(model)
    HISTORIES[arch], MODELS[arch], SPECS[arch] = hist.history, model, spec
    RESULTS[arch] = {"train_minutes": round((time.time() - t0) / 60, 1),
                     "epochs_run": len(hist.history["loss"]), "best_epoch": keep.best_epoch}
    print(f"{arch} done in {RESULTS[arch]['train_minutes']} min")
"""),
    (MD, r"""## Evaluation on unseen signers"""),
    (PY, r"""from sklearn.metrics import f1_score

Xte = (X[te].astype(np.float32) - scaler.mean_) / scaler.scale_
Xte = Xte.astype(np.float32)
Yte = Y[te]

def cpu_latency_ms(arch: str, model, runs: int = 50) -> float:
    # one-window latency on CPU (like a laptop running the app), on a fresh CPU copy
    try:
        with tf.device("/CPU:0"):
            cpu_model = build_from_spec(SPECS[arch])
            cpu_model.set_weights(model.get_weights())
            one = Xte[:1]
            for _ in range(5):
                cpu_model.predict_on_batch(one)
            t0 = time.perf_counter()
            for _ in range(runs):
                cpu_model.predict_on_batch(one)
        return (time.perf_counter() - t0) / runs * 1000
    except Exception as exc:  # timing is a nice-to-have; never lose a finished run over it
        print(f"  CPU latency not measured for {arch}: {type(exc).__name__}: {exc}"[:300])
        return float("nan")

PRED = {}
for arch, model in MODELS.items():
    probs = model.predict(Xte, batch_size=512, verbose=0)
    pred = probs.argmax(1)
    top5 = np.mean([y in p for y, p in zip(Yte, np.argsort(probs, 1)[:, -5:])])
    PRED[arch] = pred
    RESULTS[arch].update({
        "top1": float(np.mean(pred == Yte)), "top5": float(top5),
        "macro_f1": float(f1_score(Yte, pred, average="macro", labels=np.unique(Yte))),
        "params": int(model.count_params()), "cpu_ms": round(cpu_latency_ms(arch, model), 1),
    })
    if len(YCT):   # ASL Citizen's own test signers: a second, independent test
        pc = model.predict(((XCT.astype(np.float32) - scaler.mean_) / scaler.scale_).astype(np.float32),
                           batch_size=512, verbose=0)
        RESULTS[arch].update({"aslc_top1": float(np.mean(pc.argmax(1) == YCT)),
                              "aslc_top5": float(np.mean([y in p for y, p in zip(YCT, np.argsort(pc, 1)[:, -5:])]))})

table = pd.DataFrame(RESULTS).T[["top1", "top5", "macro_f1"] + (["aslc_top1", "aslc_top5"] if len(YCT) else [])
                                + ["params", "cpu_ms", "best_epoch", "train_minutes"]]
table = table.sort_values("top1", ascending=False)
print(table.to_string(float_format=lambda v: f"{v:.4f}"))
BEST = table.index[0]
print("\nbest:", BEST)
"""),
    (PY, r"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for arch, h in HISTORIES.items():
    axes[0].plot(h["val_accuracy"], label=arch)
    axes[1].plot(h["val_loss"], label=arch)
axes[0].set_title("validation accuracy (unseen signers)"); axes[1].set_title("validation loss")
for ax in axes: ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "comparison_curves.png", dpi=140)

fig, ax = plt.subplots(figsize=(7, 4))
order = list(table.index)
ax.bar(order, [RESULTS[a]["top1"] * 100 for a in order], color=["#1E5A45", "#A8731E", "#6B6B6B"][:len(order)])
for i, a in enumerate(order):
    ax.text(i, RESULTS[a]["top1"] * 100 + 0.5, f"{RESULTS[a]['top1'] * 100:.1f}%", ha="center")
ax.set_ylabel("top-1 accuracy on unseen signers (%)"); ax.set_ylim(0, 100)
fig.tight_layout(); fig.savefig(OUT / "comparison_top1.png", dpi=140)

cm = confusion_matrix(Yte, PRED[BEST], labels=np.arange(len(LABELS)), normalize="true")
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cm, cmap="Greens"); ax.set_title(f"{BEST}: confusion matrix (row-normalized)")
ax.set_xlabel("predicted"); ax.set_ylabel("true")
fig.tight_layout(); fig.savefig(OUT / f"confusion_{BEST}.png", dpi=120)
plt.close("all")
"""),
    (MD, r"""## Live-pipeline check (real clips through the app's recognizer)
Offline accuracy feeds each test clip to the model directly. The live app instead sees a
**stream of frames**: it has to notice when a sign starts and ends (from hand motion), cut it
out, and decide whether it is confident enough to answer. Here 1,000 random test clips (unseen
signers) are streamed frame by frame — with empty frames before and after, as if the signer
lowered their hands — through SignBridge's own `SignToTextPipeline`, exactly as in the app.

* **correct** = the app answered with exactly the right sign
* **wrong** = it answered with a different sign
* **not sure** = it detected a sign but was not confident enough to answer
* **missed** = it did not detect a sign at all

The threshold table shows the trade-off behind `inference.segment_confidence`."""),
    (PY, r"""from sign_bridge.capture.holistic import HolisticResult
from sign_bridge.inference.sign_to_text import SignToTextPipeline

LIVE_N, LIVE_FPS = 1000, 30.0
live_rows, LIVE = [], {}
try:
    live_idx = np.random.default_rng(SEED).choice(np.flatnonzero(te), size=min(LIVE_N, int(te.sum())),
                                                  replace=False)
    pipe = SignToTextPipeline(MODELS[BEST], LABELS, scaler, load_config(),
                              feature_mode=FEATURE_MODE, mode="segment")
    pipe.LIVE_EVERY = float("inf")   # only the answers given when a sign ends (no live guesses)
    seen = []
    _classify = pipe._classify_segment
    def _spy(seg):
        out = _classify(seg)
        seen.append(out)
        return out
    pipe._classify_segment = _spy
    blank = np.zeros(RAW_DIM, np.float32)
    necks = []
    for n, i in enumerate(live_idx, 1):
        raw = load_raw(meta["path"].iloc[i])
        pose = raw[:, 126:162].reshape(len(raw), 12, 3)
        ok = np.any(pose[:, 4, :2] != 0, 1) & np.any(pose[:, 5, :2] != 0, 1) & np.any(pose[:, 0, :2] != 0, 1)
        if ok.any():
            mid = (pose[ok, 4, :2] + pose[ok, 5, :2]) / 2
            necks.append(np.median((mid[:, 1] - pose[ok, 0, 1]) / np.linalg.norm(pose[ok, 4, :2] - pose[ok, 5, :2], axis=1)))
        pipe.reset(); seen.clear()
        stream = [blank] * 15 + list(raw) + [blank] * 60
        locked = []
        for k, f in enumerate(stream):
            left, right = bool(np.any(f[:63])), bool(np.any(f[63:126]))
            o = pipe.process_frame(HolisticResult(features=f, hands_visible=left or right,
                                                  left_hand_visible=left, right_hand_visible=right,
                                                  raw=None), now=k / LIVE_FPS)
            if o.is_locked:
                locked.append(o.locked_sign)
        truth = LABELS[Y[i]]
        best = max(seen, key=lambda r: r[1]) if seen else None
        live_rows.append({"truth": truth, "segments": len(seen), "answers": locked,
                          "best": best[0] if best else None, "conf": best[1] if best else 0.0,
                          "margin": best[2] if best else 0.0})
        if n % 250 == 0:
            print(f"  {n}/{len(live_idx)} clips streamed", flush=True)
    lr = pd.DataFrame(live_rows)
    status = np.where(lr["segments"] == 0, "missed",
             np.where(lr["answers"].map(len) == 0, "not sure",
             np.where(lr.apply(lambda r: r["answers"] == [r["truth"]], axis=1), "correct", "wrong")))
    lr["status"] = status
    counts = lr["status"].value_counts(normalize=True).reindex(["correct", "wrong", "not sure", "missed"]).fillna(0)
    offline = float(np.mean(MODELS[BEST].predict(((X[live_idx].astype(np.float32) - scaler.mean_) / scaler.scale_).astype(np.float32), verbose=0).argmax(1) == Y[live_idx]))
    inf = load_config().inference
    sweep = []
    for c in (0.0, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7):
        ans = (lr["segments"] > 0) & (lr["conf"] >= c) & (lr["margin"] >= inf.segment_margin)
        right = ans & (lr["best"] == lr["truth"])
        sweep.append({"confidence ≥": c, "answered": ans.mean(), "correct": right.mean(),
                      "wrong": (ans & ~right).mean(),
                      "precision": right.sum() / max(1, ans.sum())})
    sweep = pd.DataFrame(sweep)
    LIVE = {"clips": int(len(lr)), "model": BEST, "offline_top1_same_clips": offline,
            **{k: float(v) for k, v in counts.items()},
            "one_segment_per_clip": float(np.mean(lr["segments"] == 1)),
            "segment_confidence": inf.segment_confidence, "segment_margin": inf.segment_margin,
            "median_neck_ratio_raw": float(np.median(necks)) if necks else None}
    print(f"\nLive-pipeline check — {BEST}, {len(lr)} unseen-signer clips")
    print(f"offline top-1 on the same clips: {offline:.1%}")
    for k in ("correct", "wrong", "not sure", "missed"):
        print(f"  {k:>9}: {LIVE[k]:.1%}")
    print(f"exactly one sign detected per clip: {LIVE['one_segment_per_clip']:.1%}")
    print("\nconfidence threshold trade-off (margin ≥", inf.segment_margin, ")")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if necks:
        print(f"\nraw nose→shoulder distance / shoulder width (camera shape cue): median {np.median(necks):.2f}")
except Exception as exc:  # a diagnostic — never lose the trained models over it
    print("live-pipeline check skipped:", type(exc).__name__, exc)
"""),
    (MD, r"""## Models answering together (earlier runs attached)
If earlier SignBridge models are attached as an input (their `.keras`, `.scaler.joblib` and
`.labels.json` files), each is scored on the **same test signers**, and so is the average of
their answers with this run's model. Equal weights, decided in advance: picking weights on the
test signers would overstate the gain. The best combination also goes through the live check."""),
    (PY, r"""from sign_bridge.models.ensemble import AveragedModel, IdentityScaler
from sign_bridge.models.evaluator import load_model_from_stem

ENSEMBLE, ENS_LIVE = {}, {}
try:
    found = {}
    for k in sorted(SRC_SEARCH.rglob("asl_conv_transformer_*.keras")):   # the best architecture only
        stem = k.with_suffix("")
        if OUT in k.parents or not Path(f"{stem}.scaler.joblib").is_file():
            continue
        found.setdefault(stem.name, stem)                    # one copy of each model
    EARLIER = {}
    for name, stem in found.items():
        m, labels_m, sc = load_model_from_stem(stem)
        if labels_m != LABELS or tuple(m.input_shape[1:]) != (WINDOW, FEAT):
            print("not comparable, skipped:", name); continue
        EARLIER[name] = (m, sc)
    print("earlier models attached:", list(EARLIER) or "none")
    if EARLIER:
        def probs_of(m, sc, x):
            xs = ((x.astype(np.float32) - sc.mean_) / sc.scale_).astype(np.float32)
            return m.predict(xs, batch_size=512, verbose=0)
        NEW = f"this run ({BEST})"
        P = {NEW: MODELS[BEST].predict(Xte, batch_size=512, verbose=0)}
        PC = {NEW: probs_of(MODELS[BEST], scaler, XCT)} if len(YCT) else {}
        for name, (m, sc) in EARLIER.items():
            P[name] = probs_of(m, sc, X[te])
            if len(YCT): PC[name] = probs_of(m, sc, XCT)
        def score(p, y):
            return float(np.mean(p.argmax(1) == y)), float(np.mean([t in r for t, r in zip(y, np.argsort(p, 1)[:, -5:])]))
        from itertools import combinations
        rows = []
        for size in range(1, len(P) + 1):
            for combo in combinations(P, size):
                t1, t5 = score(sum(P[c] for c in combo) / size, Yte)
                row = {"models": " + ".join(combo), "n": size, "top1": t1, "top5": t5}
                if PC:
                    row["aslc_top1"], row["aslc_top5"] = score(sum(PC[c] for c in combo) / size, YCT)
                rows.append(row)
        ens_table = pd.DataFrame(rows).sort_values("top1", ascending=False)
        print(ens_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        best_combo = ens_table.iloc[0]["models"].split(" + ")
        ENSEMBLE = {"rows": rows, "best": best_combo}

        # the best combination through the live recognizer, on the same 1,000 clips as above
        if len(best_combo) > 1 and LIVE:
            members = [(MODELS[BEST], scaler) if c == NEW else EARLIER[c] for c in best_combo]
            epipe = SignToTextPipeline(AveragedModel(members), LABELS, IdentityScaler(), load_config(),
                                       feature_mode=FEATURE_MODE, mode="segment")
            epipe.LIVE_EVERY = float("inf")
            erows = []
            for n, i in enumerate(live_idx, 1):
                raw = load_raw(meta["path"].iloc[i])
                epipe.reset()
                locked = []
                for k, f in enumerate([blank] * 15 + list(raw) + [blank] * 60):
                    left, right = bool(np.any(f[:63])), bool(np.any(f[63:126]))
                    o = epipe.process_frame(HolisticResult(features=f, hands_visible=left or right,
                                                           left_hand_visible=left, right_hand_visible=right,
                                                           raw=None), now=k / LIVE_FPS)
                    if o.is_locked:
                        locked.append(o.locked_sign)
                erows.append({"truth": LABELS[Y[i]], "answers": locked})
                if n % 250 == 0:
                    print(f"  {n}/{len(live_idx)} clips streamed (models together)", flush=True)
            er = pd.DataFrame(erows)
            est = np.where(er["answers"].map(len) == 0, "not sure / missed",
                  np.where(er.apply(lambda r: r["answers"] == [r["truth"]], axis=1), "correct", "wrong"))
            ec = pd.Series(est).value_counts(normalize=True)
            ENS_LIVE = {"models": best_combo, "clips": int(len(er)),
                        **{k: float(ec.get(k, 0.0)) for k in ("correct", "wrong", "not sure / missed")}}
            print("\nLive check, models together:", " + ".join(best_combo))
            for k in ("correct", "wrong", "not sure / missed"):
                print(f"  {k:>17}: {ENS_LIVE[k]:.1%}")
            print(f"  (this run alone: correct {LIVE['correct']:.1%}, wrong {LIVE['wrong']:.1%})")
except Exception as exc:  # a comparison — never lose the trained model over it
    print("models-together check skipped:", type(exc).__name__, exc)
"""),
    (MD, r"""## Export for the SignBridge app
Every model is written in SignBridge's model format; `asl_latest.json` points at the best one.
Unzip `signbridge_asl_models.zip` into the project's `models/` folder and run `sign-bridge`."""),
    (PY, r"""import joblib, zipfile
from datetime import datetime, timezone

export = OUT / "signbridge_models"
shutil.rmtree(export, ignore_errors=True); export.mkdir()
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
per_class = {}
for arch, model in MODELS.items():
    stem = export / f"asl_{arch}_{stamp}"
    model.save(f"{stem}.keras")
    model.save_weights(f"{stem}.weights.h5")
    Path(f"{stem}.spec.json").write_text(json.dumps(SPECS[arch], indent=2))
    Path(f"{stem}.labels.json").write_text(json.dumps(LABELS, indent=2))
    joblib.dump(scaler, f"{stem}.scaler.joblib")
    f1s = f1_score(Yte, PRED[arch], average=None, labels=np.arange(len(LABELS)), zero_division=0)
    per_class[arch] = f1s
    pd.DataFrame({"sign": LABELS, "f1": f1s}).sort_values("f1").to_csv(f"{stem}.per_class_f1.csv", index=False)
    Path(f"{stem}.metrics.json").write_text(json.dumps({
        "language": "asl", "architecture": arch, "feature_mode": FEATURE_MODE,
        "dataset": "Google ASL Signs (Kaggle asl-signs)" + (" + ASL Citizen train/val (kept glosses)" if len(YC) else ""),
        "split": "signer-independent",
        "test_participants": [int(p) for p in test_p], "val_participants": [int(p) for p in val_p],
        "test_accuracy": RESULTS[arch]["top1"], "test_top5": RESULTS[arch]["top5"],
        "test_macro_f1": RESULTS[arch]["macro_f1"], "n_classes": len(LABELS),
        "n_train": int(len(Y_TRAIN)), "n_train_aslc": int(len(YC)), "n_val": int(va.sum()), "n_test": int(te.sum()),
        **({"aslc_test_top1": RESULTS[arch]["aslc_top1"], "aslc_test_top5": RESULTS[arch]["aslc_top5"],
            "n_test_aslc": int(len(YCT))} if len(YCT) else {}),
        "params": RESULTS[arch]["params"], "cpu_latency_ms": RESULTS[arch]["cpu_ms"],
        "window_size": WINDOW, "epochs_run": RESULTS[arch]["epochs_run"],
        "best_epoch": RESULTS[arch]["best_epoch"], "clip_mode": "whole_sign",
        "timestamp_utc": stamp}, indent=2))

best_stem = f"asl_{BEST}_{stamp}"
(export / "asl_latest.json").write_text(json.dumps({
    "model": f"{best_stem}.keras", "stem": best_stem, "feature_mode": FEATURE_MODE,
    "architecture": BEST, "window_size": WINDOW, "window_seconds": WINDOW_SECONDS,
    "clip_mode": "whole_sign"}, indent=2))

for name in ("comparison_curves.png", "comparison_top1.png", f"confusion_{BEST}.png"):
    shutil.copy(OUT / name, export / name)
table.to_csv(export / "results_table.csv")
(export / "results_table.md").write_text(
    "| model | top-1 | top-5 | macro-F1 | params | CPU ms |\n|---|---|---|---|---|---|\n" +
    "\n".join(f"| {a} | {r['top1']*100:.2f}% | {r['top5']*100:.2f}% | {r['macro_f1']:.4f} | "
              f"{r['params']:,} | {r['cpu_ms']} |" for a, r in ((a, RESULTS[a]) for a in table.index)) +
    f"\n\nGoogle ASL Signs, {len(LABELS)} signs, signer-independent split: "
    f"{len(train_p)} training / {len(val_p)} validation / {len(test_p)} test signers "
    f"({int(te.sum()):,} test clips). Window {WINDOW} frames, features {FEATURE_MODE} ({FEAT}).\n" +
    (f"\nTraining also used {len(YC):,} ASL Citizen clips (train + val signers, kept glosses). "
     f"ASL Citizen test signers ({len(YCT):,} clips): " +
     ", ".join(f"{a} top-1 {RESULTS[a]['aslc_top1']:.2%} / top-5 {RESULTS[a]['aslc_top5']:.2%}" for a in table.index) +
     ".\n" if len(YCT) else "") +
    (f"\nLive-pipeline check ({LIVE['model']}, {LIVE['clips']} test clips streamed frame by frame "
     f"through the app's recognizer): correct {LIVE['correct']:.1%}, wrong {LIVE['wrong']:.1%}, "
     f"not sure {LIVE['not sure']:.1%}, missed {LIVE['missed']:.1%} "
     f"(offline top-1 on the same clips {LIVE['offline_top1_same_clips']:.1%}).\n" if LIVE else ""))
(export / "environment.json").write_text(json.dumps({
    "tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__,
    "signbridge": sign_bridge.__version__, "epochs": EPOCHS, "batch": BATCH,
    "learning_rate": LEARNING_RATE, "label_smoothing": LABEL_SMOOTHING, "seed": SEED,
    "train_seed": TRAIN_SEED,
    "params": PARAMS}, indent=2))

if ENSEMBLE:
    (export / "ensemble_results.json").write_text(json.dumps({**ENSEMBLE, "live": ENS_LIVE}, indent=2))
    ens_table.to_csv(export / "ensemble_results.csv", index=False)
if LIVE:
    (export / "live_check.json").write_text(json.dumps(LIVE, indent=2))
    sweep.to_csv(export / "live_check_thresholds.csv", index=False)
    lr.to_csv(export / "live_check_clips.csv", index=False)

zip_path = OUT / "signbridge_asl_models.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(export.iterdir()):
        zf.write(f, f.name)
print("wrote", zip_path, f"({zip_path.stat().st_size / 1e6:.1f} MB)")
print(open(export / "results_table.md").read())
"""),
    (MD, r"""## Reporting these results
* Quote the numbers from `results_table.md` exactly as produced; they are measured on signers the models never saw.
* Cite the dataset as instructed on the competition's data page, and check its data-use terms before publishing.
* `environment.json`, the split participants in each `*.metrics.json` and the fixed seed make the run reproducible."""),
]


#: Run C: only the best model, trained twice as long, with stronger augmentation and
#: 2-frame motion. Each entry swaps a default in the settings cell.
RUN_C = {
    '"SB_ARCHS", "lstm,transformer,conv_transformer"': '"SB_ARCHS", "conv_transformer"',
    '"SB_EPOCHS", 100': '"SB_EPOCHS", 200',
    '"SB_STRONG_AUG", "0"': '"SB_STRONG_AUG", "1"',
    '"SB_MOTION_LAGS", "1"': '"SB_MOTION_LAGS", "1,2"',
}
RUN_C_NOTE = (MD, r"""## Run C
This copy is set up for **run C**: only the conv-transformer (the best model in runs A and B),
**200 epochs** instead of 100 (in run A its best epoch was 93 of 100 — still improving), **stronger
augmentation** (a hand or the lips hidden now and then, a few frames blanked) and **2-frame motion**
features. Attach the same inputs as run B (competition, project, private ASL Citizen import).
About 3 h on GPU T4 x2.""")


#: Run D: run C's recipe with another training seed, scored alone and together with the
#: earlier runs' models (attach them as an input).
RUN_D = {**RUN_C, '"SB_TRAIN_SEED", SEED': '"SB_TRAIN_SEED", 7'}
RUN_D_NOTE = (MD, r"""## Run D
This copy is set up for **run D**: exactly run C's recipe (conv-transformer, 200 epochs, stronger
augmentation, 2-frame motion) with a **different training seed** (7 instead of 42). The signer
split stays the same, so all runs are scored on the same people. Two models trained alike but
from different starting points make different mistakes; averaging their answers usually beats
either one. Attach the same inputs as run C **plus** a private dataset holding the run B and
run C model files, and the notebook also scores every combination. About 3 h on GPU T4 x2.""")

PROFILES = {"C": (RUN_C, RUN_C_NOTE), "D": (RUN_D, RUN_D_NOTE)}


def cells(profile: str = "") -> list[tuple[str, str]]:
    """The cells, with a profile's settings applied (``""`` = defaults, ``"C"``/``"D"`` = runs)."""
    out = list(CELLS)
    if profile:
        swaps, note = PROFILES[profile]
        settings = out[1][1]
        for old, new in swaps.items():
            assert old in settings, old
            settings = settings.replace(old, new)
        out = [out[0], note, (PY, settings), *out[2:]]
    return out


def build(path: Path, profile: str = "") -> None:
    """Write the notebook file."""
    nb = {
        "cells": [
            {"cell_type": kind, "metadata": {},
             **({"outputs": [], "execution_count": None} if kind == PY else {}),
             "source": src.strip("\n").splitlines(keepends=True)}
            for kind, src in cells(profile)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def code_cells(profile: str = "") -> list[str]:
    """The code cells in order (used to execute the notebook in tests)."""
    return [src for kind, src in cells(profile) if kind == PY]


if __name__ == "__main__":
    out = Path(__file__).with_name("train_asl_signs_kaggle.ipynb")
    build(out)
    print("wrote", out)
    out_c = Path(__file__).with_name("train_asl_run_c.ipynb")
    build(out_c, "C")
    print("wrote", out_c)
    out_d = Path(__file__).with_name("train_asl_run_d.ipynb")
    build(out_d, "D")
    print("wrote", out_d)
