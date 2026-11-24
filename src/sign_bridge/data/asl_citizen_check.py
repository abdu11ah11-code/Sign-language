"""``sign-check-aslcitizen`` — is each imported ASL Citizen sign really the model's sign?

The importer files ASL Citizen videos under SignBridge's sign names. Most match
by name (MOM is MOM), but some are only *candidates*:

* **aliases** — a different name that may be the same sign (FATHER for DAD);
* **variants** — numbered entries that the importer merges (NUT1 and NUT2 both
  become NUT), which can be two different signs that share an English word.

An English word is not proof of the same sign, so this asks the trained model:
run it on a candidate's videos, and if it recognises them as the sign they were
filed under about as well as it recognises the signs that matched exactly, the
candidate is kept. Otherwise it is dropped. Nothing is guessed.

Only ASL Citizen's **train and val** signers are used for the decision, so its
test signers stay untouched for a later benchmark. As a by-product the check
reports how well the current model, which never saw ASL Citizen, recognises
its webcam videos — split by split, test included.

Output: ``<data>/check.json`` — every gloss with its numbers and a keep/drop
decision, plus ``keep_glosses`` for the training notebook.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sign_bridge.data.asl_citizen import iter_clips, normalize_gloss
from sign_bridge.logging_conf import configure_logging, get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)

#: Splits the keep/drop decision may look at (never ``test``).
DECISION_SPLITS = ("train", "val")
#: A candidate needs at least this many videos to be judged at all.
MIN_CLIPS = 5
#: A candidate is kept when its top-5 agreement reaches this percentile of the
#: exact matches' per-sign top-5 agreement — no worse than the weakest tenth
#: of signs that certainly are the same sign.
KEEP_PERCENTILE = 10.0
#: Clips predicted per model call.
BATCH = 256


@dataclass
class GlossResult:
    """How the model saw one ASL Citizen gloss (decision splits only)."""

    gloss: str
    sign: str
    kind: str                 # exact / variant / alias
    clips: int
    top1: float
    top5: float
    most_predicted: str
    most_predicted_share: float
    keep: bool = True


def gloss_kind(gloss: str, sign: str) -> str:
    """``exact`` (same name), ``variant`` (CAT1 → CAT) or ``alias`` (FATHER → DAD)."""
    if normalize_gloss(gloss) != sign:
        return "alias"
    return "exact" if re.sub(r"[^A-Z0-9]", "", gloss.strip().upper()) == sign else "variant"


def predict(model: Any, scaler: Any, clips: NDArray, mode: str, window: int) -> NDArray:
    """Class probabilities for raw ``(N, T, 282)`` clips, prepared as the app does."""
    from sign_bridge.data.preprocess import normalize_window, resample_clip

    out = []
    for start in range(0, len(clips), BATCH):
        part = np.stack([normalize_window(resample_clip(c.astype(np.float32), window), mode)
                         for c in clips[start:start + BATCH]])
        flat = scaler.transform(part.reshape(-1, part.shape[-1]))
        part = flat.reshape(part.shape).astype(np.float32)
        out.append(np.asarray(model.predict(part, batch_size=BATCH, verbose=0)))
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def score(rows: list[dict[str, str]], probs: NDArray, labels: list[str]) -> list[GlossResult]:
    """Per-gloss agreement between the model and the sign each video is filed under."""
    index = {normalize_gloss(lbl): i for i, lbl in enumerate(labels)}
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if row["sign"] in index:
            groups.setdefault(row["gloss"], []).append(i)
    results = []
    for gloss, members in sorted(groups.items()):
        sign = rows[members[0]]["sign"]
        target = index[sign]
        p = probs[members]
        top5 = np.argsort(p, axis=1)[:, ::-1][:, :5]
        predicted = top5[:, 0]
        values, counts = np.unique(predicted, return_counts=True)
        best = int(values[np.argmax(counts)])
        results.append(GlossResult(
            gloss=gloss, sign=sign, kind=gloss_kind(gloss, sign), clips=len(members),
            top1=float(np.mean(predicted == target)),
            top5=float(np.mean(np.any(top5 == target, axis=1))),
            most_predicted=labels[best], most_predicted_share=float(counts.max() / len(members))))
    return results


def decide(results: list[GlossResult], percentile: float = KEEP_PERCENTILE,
           min_clips: int = MIN_CLIPS) -> float:
    """Mark each candidate keep/drop; returns the top-5 threshold used.

    Exact matches are always kept: they *are* the reference, and dropping a
    real sign because the model is weak at it would bias the data. Variants
    and aliases must reach the threshold.
    """
    reference = [r.top5 for r in results if r.kind == "exact" and r.clips >= min_clips]
    if not reference:
        raise ValueError("No exact matches to compare against — is this an ASL Citizen import?")
    threshold = float(np.percentile(reference, percentile))
    for r in results:
        if r.kind != "exact":
            r.keep = r.clips >= min_clips and r.top5 >= threshold
    return threshold


def split_accuracy(rows: list[dict[str, str]], probs: NDArray,
                   labels: list[str]) -> dict[str, dict[str, float]]:
    """Top-1/top-5 per split, exact matches only: a clean webcam benchmark."""
    index = {normalize_gloss(lbl): i for i, lbl in enumerate(labels)}
    out: dict[str, dict[str, float]] = {}
    for split in sorted({r["split"] for r in rows}):
        members = [i for i, r in enumerate(rows) if r["split"] == split
                   and r["sign"] in index and gloss_kind(r["gloss"], r["sign"]) == "exact"]
        if not members:
            continue
        target = np.array([index[rows[i]["sign"]] for i in members])
        top5 = np.argsort(probs[members], axis=1)[:, ::-1][:, :5]
        out[split] = {"clips": len(members),
                      "top1": float(np.mean(top5[:, 0] == target)),
                      "top5": float(np.mean(np.any(top5 == target[:, None], axis=1)))}
    return out


def _model_stem(stem: Path | None, config_path: str | None) -> Path:
    if stem is not None:
        return stem
    from sign_bridge.config import load_config

    cfg = load_config(config_path)
    pointer = cfg.resolve(cfg.paths.models_dir) / "asl_latest.json"
    if not pointer.is_file():
        raise FileNotFoundError("No trained ASL model (models/asl_latest.json). Pass --model.")
    return pointer.parent / str(json.loads(pointer.read_text(encoding="utf-8"))["stem"])


def _model_meta(stem: Path) -> tuple[str, int]:
    """Feature mode and window the model was trained with."""
    for path in (Path(f"{stem}.metrics.json"), Path(f"{stem}.spec.json")):
        if path.is_file():
            meta = json.loads(path.read_text(encoding="utf-8"))
            if "feature_mode" in meta and "window_size" in meta:
                return str(meta["feature_mode"]), int(meta["window_size"])
    raise FileNotFoundError(f"Cannot tell how {stem.name} prepares its input: no "
                            "feature_mode/window_size in its metrics or spec file.")


def check(data: Path, stem: Path | None = None, config_path: str | None = None,
          percentile: float = KEEP_PERCENTILE, loaded: tuple[Any, list[str], Any] | None = None,
          meta: tuple[str, int] | None = None) -> dict[str, Any]:
    """Run the check on an import folder and write ``check.json``."""
    from sign_bridge.models.evaluator import load_model_from_stem

    model_stem = _model_stem(stem, config_path) if loaded is None else Path(stem or "given")
    model, labels, scaler = loaded or load_model_from_stem(model_stem)
    mode, window = meta or _model_meta(model_stem)

    pairs = list(iter_clips(data))
    rows = [row for _, row in pairs]
    clips = np.stack([clip for clip, _ in pairs]) if pairs else np.zeros((0, window, 282))
    log.info("aslc_check_predicting", clips=len(rows), feature_mode=mode, window=window)
    probs = predict(model, scaler, clips, mode, window)

    decision = [i for i, r in enumerate(rows) if r["split"] in DECISION_SPLITS]
    results = score([rows[i] for i in decision], probs[decision], labels)
    threshold = decide(results, percentile)
    report = {
        "model": model_stem.name, "feature_mode": mode, "window": window,
        "decision_splits": list(DECISION_SPLITS), "keep_percentile": percentile,
        "top5_threshold": threshold,
        "accuracy_exact_matches": split_accuracy(rows, probs, labels),
        "keep_glosses": sorted(r.gloss for r in results if r.keep),
        "drop_glosses": sorted(r.gloss for r in results if not r.keep),
        "glosses": [asdict(r) for r in results],
    }
    (data / "check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Entry point for ``sign-check-aslcitizen``."""
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(
        prog="sign-check-aslcitizen",
        description="Test each imported ASL Citizen sign against the trained model and "
                    "decide which name matches to keep.")
    parser.add_argument("--data", type=Path, default=Path("data/asl_citizen"),
                        help="the import folder (default: data/asl_citizen)")
    parser.add_argument("--model", type=Path,
                        help="model file stem, without extension (default: the latest ASL model)")
    parser.add_argument("--percentile", type=float, default=KEEP_PERCENTILE,
                        help=f"keep threshold, as a percentile of exact matches' top-5 "
                             f"agreement (default: {KEEP_PERCENTILE:g})")
    args = parser.parse_args()

    report = check(args.data, args.model, percentile=args.percentile)
    print(f"\nmodel: {report['model']}  ({report['feature_mode']}, window {report['window']})")
    print("\nHow the model does on ASL Citizen's webcam videos (exact name matches only):")
    for split, acc in report["accuracy_exact_matches"].items():
        print(f"  {split:5s} {acc['clips']:5,} clips   top-1 {acc['top1']:6.1%}   "
              f"top-5 {acc['top5']:6.1%}")
    print(f"\nKeep threshold: top-5 agreement >= {report['top5_threshold']:.1%} "
          f"(the {report['keep_percentile']:g}th percentile of exact matches, "
          f"train + val signers only)\n")
    print(f"  {'ASL Citizen':16s} {'filed as':12s} {'kind':8s} {'clips':>5s} "
          f"{'top-1':>7s} {'top-5':>7s}  most predicted       decision")
    for r in report["glosses"]:
        if r["kind"] == "exact":
            continue
        print(f"  {r['gloss']:16s} {r['sign']:12s} {r['kind']:8s} {r['clips']:5d} "
              f"{r['top1']:7.1%} {r['top5']:7.1%}  "
              f"{r['most_predicted']:12s} {r['most_predicted_share']:5.0%}   "
              f"{'KEEP' if r['keep'] else 'drop'}")
    print(f"\nKept {len(report['keep_glosses'])} glosses, dropped "
          f"{len(report['drop_glosses'])}. Written to {args.data / 'check.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()
