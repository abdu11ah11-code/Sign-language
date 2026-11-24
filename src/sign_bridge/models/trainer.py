"""``sign-train`` — train a sign classifier for one language.

Usage:
    sign-train --language asl --architecture lstm
    sign-train --language asl --architecture conv_transformer --features holistic_lips
    sign-train --language psl --architecture transformer
    sign-train --language asl --skip-cv --epochs 40     # quick iteration
    sign-train --language psl --features hand_shape     # imported photo dataset

Pipeline: load CSVs → normalize → augment (train folds only) → 5-fold
stratified CV for an honest accuracy estimate → final fit on a train/val
split with early stopping + LR schedule → evaluate on a held-out test set.

Artifacts (timestamped — existing models are never overwritten):
    models/<lang>_<arch>_<ts>.keras       trained model
    models/<lang>_<arch>_<ts>.weights.h5  weights (fallback if .keras won't load)
    models/<lang>_<arch>_<ts>.spec.json   architecture + hyperparameters
    models/<lang>_<arch>_<ts>.labels.json class-index → gloss
    models/<lang>_<arch>_<ts>.scaler.joblib
    models/<lang>_<arch>_<ts>.curves.png  loss/accuracy curves
    models/<lang>_<arch>_<ts>.metrics.json
    models/<lang>_latest.json             pointer consumed by inference
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sign_bridge.config import AppConfig, load_config
from sign_bridge.data.augment import augment_dataset
from sign_bridge.data.preprocess import (
    FEATURE_MODES,
    Dataset,
    apply_scaler,
    canonical_hand_window,
    fit_scaler,
    load_dataset,
    save_labels,
    save_scaler,
)
from sign_bridge.logging_conf import configure_logging, get_logger
from sign_bridge.models.registry import ARCHITECTURES
from sign_bridge.typing import NDArray

log = get_logger(__name__)


def _build(
    architecture: str, num_classes: int, cfg: AppConfig, window_size: int, feature_dim: int
) -> tuple[Any, dict[str, Any]]:
    """Construct the requested architecture; returns ``(model, spec)``."""
    from sign_bridge.models.registry import build_from_spec, default_params, make_spec

    spec = make_spec(architecture, num_classes, window_size, feature_dim,
                     default_params(architecture, cfg))
    return build_from_spec(spec), spec


def _callbacks(cfg: AppConfig) -> list[Any]:
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=cfg.training.reduce_lr_patience,
            factor=cfg.training.reduce_lr_factor,
            verbose=0,
        ),
    ]


def _class_weights(y: NDArray, num_classes: int) -> dict[int, float]:
    """Inverse-frequency class weights for imbalanced sign counts.

    Every class gets an entry (1.0 when absent from ``y``) because Keras
    rejects a ``class_weight`` dict with missing class ids.
    """
    counts = np.bincount(y, minlength=num_classes)
    present = int((counts > 0).sum())
    total = counts.sum()
    return {
        i: float(total / (present * c)) if c > 0 else 1.0 for i, c in enumerate(counts)
    }


def _split(
    x: NDArray, y: NDArray, fraction: float, seed: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Stratified hold-out split that also works on very small datasets.

    sklearn refuses a stratified split when the hold-out set is smaller than
    the number of classes (common with a few clips per sign). We hold out at
    least one clip per class, and fall back to a plain shuffle split only
    when stratification is truly impossible.
    """
    from sklearn.model_selection import train_test_split

    counts = np.bincount(y)
    n_classes = int((counts > 0).sum())
    n_hold = max(int(round(fraction * len(y))), n_classes)
    if counts[counts > 0].min() >= 2 and len(y) - n_hold >= n_classes:
        parts = train_test_split(x, y, test_size=n_hold, stratify=y, random_state=seed)
    else:
        log.warning("split_not_stratified", reason="too few clips per sign")
        parts = train_test_split(
            x, y, test_size=max(1, int(round(fraction * len(y)))), random_state=seed
        )
    x_a, x_b, y_a, y_b = parts
    return x_a, x_b, y_a, y_b


def _check_dataset(ds: Dataset) -> None:
    """Fail early, with a useful message, when there is too little data."""
    counts = np.bincount(ds.y, minlength=len(ds.labels))
    too_few = [f"{lbl} ({n})" for lbl, n in zip(ds.labels, counts, strict=True) if n < 3]
    if too_few:
        raise SystemExit(
            "Each sign needs at least 3 recorded clips (30+ recommended). Too few: "
            + ", ".join(too_few)
        )
    if len(ds.labels) < 2:
        raise SystemExit("Record at least 2 different signs before training.")
    thin = [lbl for lbl, n in zip(ds.labels, counts, strict=True) if n < 10]
    if thin:
        log.warning("few_clips_per_sign", signs=thin, advice="record 30+ clips per sign")


def _augment(
    x: NDArray, y: NDArray, cfg: AppConfig, feature_mode: str = "holistic"
) -> tuple[NDArray, NDArray]:
    if not cfg.training.augment:
        return x, y
    a = cfg.augmentation
    x_aug, y_aug = augment_dataset(
        x, y,
        copies_per_sample=a.copies_per_sample,
        rotation_deg=a.rotation_deg,
        scale_pct=a.scale_pct,
        use_mirror=a.mirror,
        jitter_frames=a.temporal_jitter_frames,
        seed=cfg.training.seed,
        feature_mode=feature_mode,
    )
    if feature_mode == "hand_shape":
        # Mirroring swapped hand slots; the signing hand lives in the right slot.
        x_aug = np.stack([canonical_hand_window(w) for w in x_aug]).astype(np.float32)
    return x_aug, y_aug


def cross_validate(
    ds: Dataset, architecture: str, cfg: AppConfig, feature_mode: str = "holistic"
) -> list[float]:
    """K-fold CV accuracy. Augmentation/scaling are fit inside each fold —
    never on validation data — so the estimate is leak-free."""
    from sklearn.model_selection import StratifiedKFold

    # StratifiedKFold needs every class in every fold.
    folds = min(cfg.training.cv_folds, int(np.bincount(ds.y).min()))
    if folds < 2:
        log.warning("cv_skipped", reason="fewer than 2 clips in some class")
        return []
    if folds < cfg.training.cv_folds:
        log.warning("cv_folds_reduced", requested=cfg.training.cv_folds, used=folds)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.training.seed)
    scores: list[float] = []
    for fold, (tr, va) in enumerate(skf.split(ds.x, ds.y), start=1):
        x_tr, y_tr = _augment(ds.x[tr], ds.y[tr], cfg, feature_mode)
        scaler = fit_scaler(x_tr)
        x_tr, x_va = apply_scaler(x_tr, scaler), apply_scaler(ds.x[va], scaler)

        model, _ = _build(architecture, len(ds.labels), cfg, ds.x.shape[1], ds.x.shape[2])
        model.fit(
            x_tr, y_tr,
            validation_data=(x_va, ds.y[va]),
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            class_weight=(
                _class_weights(y_tr, len(ds.labels)) if cfg.training.use_class_weights else None
            ),
            callbacks=_callbacks(cfg),
            verbose=0,
        )
        _, acc = model.evaluate(x_va, ds.y[va], verbose=0)
        scores.append(float(acc))
        log.info("cv_fold_done", fold=fold, accuracy=round(acc, 4))
    return scores


def _plot_curves(history: Any, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "accuracy"), strict=True):
        ax.plot(history.history[metric], label="train", color="#1E5A45")
        ax.plot(history.history[f"val_{metric}"], label="val", color="#A8731E")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("SignBridge training curves")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train(
    language: str,
    architecture: str,
    config_path: str | None = None,
    skip_cv: bool = False,
    epochs: int | None = None,
    feature_mode: str = "holistic",
) -> Path:
    """Full training run. Returns the saved model path.

    Args:
        language: ``"asl"`` or ``"psl"``.
        architecture: ``"lstm"``, ``"transformer"`` or ``"conv_transformer"``.
        config_path: Optional config YAML.
        skip_cv: Skip k-fold cross-validation (much faster; use while
            iterating on data, not for reported numbers).
        epochs: Override ``training.epochs``.
        feature_mode: ``"holistic"`` (body + hands, default),
            ``"hand_shape"`` (one hand relative to its wrist; for close-up
            hand photos such as UAlpha40) or ``"holistic_lips"`` (hands +
            body + lips, 2D). Saved with the model.
    """
    cfg = load_config(config_path)
    if epochs is not None:
        cfg = cfg.model_copy(
            update={"training": cfg.training.model_copy(update={"epochs": epochs})}
        )
    np.random.seed(cfg.training.seed)
    import tensorflow as tf

    tf.random.set_seed(cfg.training.seed)

    ds = load_dataset(
        cfg.resolve(cfg.paths.data_dir) / language, cfg.landmarks.window_size, feature_mode
    )
    _check_dataset(ds)

    # 1) Honest CV estimate.
    t0 = time.monotonic()
    cv_scores = [] if skip_cv else cross_validate(ds, architecture, cfg, feature_mode)
    if cv_scores:
        log.info(
            "cv_summary",
            mean=round(float(np.mean(cv_scores)), 4),
            std=round(float(np.std(cv_scores)), 4),
        )

    # 2) Final model: train / val / test split.
    x_trval, x_te, y_trval, y_te = _split(ds.x, ds.y, cfg.training.test_split, cfg.training.seed)
    x_tr, x_va, y_tr, y_va = _split(
        x_trval, y_trval, cfg.training.validation_split, cfg.training.seed
    )
    x_tr, y_tr = _augment(x_tr, y_tr, cfg, feature_mode)
    scaler = fit_scaler(x_tr)
    x_tr, x_va, x_te = (apply_scaler(a, scaler) for a in (x_tr, x_va, x_te))

    model, spec = _build(architecture, len(ds.labels), cfg, ds.x.shape[1], ds.x.shape[2])
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_va, y_va),
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        class_weight=(
            _class_weights(y_tr, len(ds.labels)) if cfg.training.use_class_weights else None
        ),
        callbacks=_callbacks(cfg),
        verbose=2,
    )

    # 3) Save timestamped artifacts.
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    models_dir = cfg.resolve(cfg.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    stem = models_dir / f"{language}_{architecture}_{stamp}"

    model_path = stem.with_suffix(".keras")
    model.save(model_path)
    model.save_weights(f"{stem}.weights.h5")
    Path(f"{stem}.spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    save_labels(ds.labels, Path(f"{stem}.labels.json"))
    save_scaler(scaler, Path(f"{stem}.scaler.joblib"))
    _plot_curves(history, Path(f"{stem}.curves.png"))

    # 4) Held-out evaluation (confusion matrix + per-class F1).
    from sign_bridge.models.evaluator import evaluate

    report = evaluate(model, x_te, y_te, ds.labels, output_stem=stem)
    metrics = {
        "language": language,
        "architecture": architecture,
        "feature_mode": feature_mode,
        "timestamp_utc": stamp,
        "cv_accuracy_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_accuracy_std": float(np.std(cv_scores)) if cv_scores else None,
        "cv_folds": len(cv_scores),
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro_f1"],
        "n_samples": int(ds.x.shape[0]),
        "n_classes": len(ds.labels),
        "train_seconds": round(time.monotonic() - t0, 1),
    }
    Path(f"{stem}.metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 5) Update the "latest" pointer used by inference.
    pointer = models_dir / f"{language}_latest.json"
    pointer.write_text(
        json.dumps(
            {"model": model_path.name, "stem": stem.name, "feature_mode": feature_mode,
             "architecture": architecture, "window_size": int(ds.x.shape[1]),
             "window_seconds": cfg.inference.window_seconds},
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {k: v for k, v in metrics.items() if k != "language"}
    log.info("training_complete", model=str(model_path), **summary)
    print(f"\nSaved {model_path}")
    cv_text = (
        f"CV accuracy {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f} | "
        if cv_scores else "CV skipped | "
    )
    print(f"{cv_text}test accuracy {metrics['test_accuracy']:.3f}"
          f" | macro-F1 {metrics['test_macro_f1']:.3f}")
    return model_path


def main() -> None:
    """Console entry point for ``sign-train``."""
    configure_logging()
    parser = argparse.ArgumentParser(prog="sign-train", description="Train a sign classifier.")
    parser.add_argument("--language", choices=["asl", "psl"], required=True)
    parser.add_argument("--architecture", choices=list(ARCHITECTURES), default=None,
                        help="lstm | transformer | conv_transformer (Conv1D + Transformer)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip cross-validation (fast iteration; not for reported numbers)")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--features", choices=FEATURE_MODES, default="holistic",
                        help="holistic = hands + body (sign-collect clips); "
                             "hand_shape = one hand vs. its wrist (close-up photo datasets); "
                             "holistic_lips = hands + body + lips (best for ASL)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(args.language, args.architecture or cfg.model.architecture, args.config,
          skip_cv=args.skip_cv, epochs=args.epochs, feature_mode=args.features)


if __name__ == "__main__":
    main()
