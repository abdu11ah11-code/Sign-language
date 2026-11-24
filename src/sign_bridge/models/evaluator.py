"""``sign-eval`` — evaluation: confusion matrix, per-class F1, reports.

Used two ways:
    * called by the trainer on the held-out test split, and
    * standalone, to re-evaluate any saved model on current data:
      ``sign-eval --language asl``  (uses the *_latest.json pointer)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from sign_bridge.config import load_config
from sign_bridge.logging_conf import configure_logging, get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)


class EvalReport(TypedDict):
    """Summary metrics returned by :func:`evaluate`."""

    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]


def _plot_confusion(cm: NDArray, labels: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5),) * 2)
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("SignBridge confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() / 2 else "#181611")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(
    model: Any,
    x: NDArray,
    y: NDArray,
    labels: list[str],
    output_stem: Path | None = None,
) -> EvalReport:
    """Evaluate a model; optionally write PNG/CSV artifacts.

    Args:
        model: Compiled Keras model.
        x: ``(N, T, F)`` inputs (already scaled).
        y: ``(N,)`` true class indices.
        labels: Index → gloss mapping.
        output_stem: If given, writes ``<stem>.confusion.png`` and
            ``<stem>.per_class_f1.csv``.

    Returns:
        :class:`EvalReport` with accuracy, macro-F1, per-class F1.
    """
    from sklearn.metrics import confusion_matrix, f1_score

    probs = model.predict(x, verbose=0)
    preds = probs.argmax(axis=1)

    accuracy = float((preds == y).mean())
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    per_class = f1_score(y, preds, average=None, zero_division=0,
                         labels=list(range(len(labels))))
    per_class_f1 = {lbl: float(f) for lbl, f in zip(labels, per_class, strict=True)}

    if output_stem is not None:
        cm = confusion_matrix(y, preds, labels=list(range(len(labels))))
        _plot_confusion(cm, labels, Path(f"{output_stem}.confusion.png"))
        pd.DataFrame(
            {"sign": labels, "f1": per_class}
        ).sort_values("f1").to_csv(f"{output_stem}.per_class_f1.csv", index=False)
        log.info("eval_artifacts_written", stem=str(output_stem))

    log.info("evaluation", accuracy=round(accuracy, 4), macro_f1=round(macro_f1, 4))
    return EvalReport(accuracy=accuracy, macro_f1=macro_f1, per_class_f1=per_class_f1)


def load_latest_model(language: str, config_path: str | None = None) -> tuple[Any, list[str], Any]:
    """Load the model/labels/scaler referenced by ``models/<lang>_latest.json``.

    Returns:
        ``(model, labels, scaler)``.

    Raises:
        FileNotFoundError: When no trained model exists for the language.
    """
    cfg = load_config(config_path)
    models_dir = cfg.resolve(cfg.paths.models_dir)
    pointer = models_dir / f"{language}_latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(
            f"No trained {language.upper()} model. Run: sign-train --language {language}"
        )
    meta = json.loads(pointer.read_text(encoding="utf-8"))
    stems = [models_dir / s for s in (meta.get("ensemble") or [meta["stem"]])]
    for stem in stems:
        if not Path(f"{stem}.keras").is_file():
            raise FileNotFoundError(
                f"{pointer.name} points to {stem.name}.keras, which is missing. "
                f"Retrain with: sign-train --language {language}"
            )
    if len(stems) == 1:
        return load_model_from_stem(stems[0])
    return load_ensemble(stems, meta.get("ensemble_weights"))


def load_ensemble(stems: list[Path], weights: list[float] | None = None
                  ) -> tuple[Any, list[str], Any]:
    """Several saved models answering together (see :mod:`sign_bridge.models.ensemble`).

    Returns:
        ``(averaged model, labels, identity scaler)``: the members scale their own input.

    Raises:
        ValueError: The models answer over different label lists.
    """
    from sign_bridge.models.ensemble import AveragedModel, IdentityScaler

    loaded = [load_model_from_stem(s) for s in stems]
    labels = loaded[0][1]
    for stem, (_, other, _) in zip(stems, loaded, strict=True):
        if other != labels:
            raise ValueError(f"{stem.name} answers over different signs than {stems[0].name}")
    return AveragedModel([(m, sc) for m, _, sc in loaded], weights), labels, IdentityScaler()


def load_model_from_stem(stem: Path) -> tuple[Any, list[str], Any]:
    """Load ``<stem>.keras`` (or its spec + weights), labels and scaler.

    Returns:
        ``(model, labels, scaler)``.
    """
    from tensorflow import keras

    from sign_bridge.data.preprocess import load_labels, load_scaler
    from sign_bridge.models.registry import build_from_spec, register_custom_objects

    # Registers the custom layers/losses; without it saved transformer and
    # conv-transformer models cannot be deserialized.
    register_custom_objects()
    try:
        # Inference needs no optimizer/loss: compile=False also avoids
        # version-specific optimizer configs (models trained on Kaggle).
        model = keras.models.load_model(f"{stem}.keras", compile=False)
    except Exception as exc:  # noqa: BLE001 — fall back to spec + weights
        spec_path, weights_path = Path(f"{stem}.spec.json"), Path(f"{stem}.weights.h5")
        if not (spec_path.is_file() and weights_path.is_file()):
            raise
        log.warning("model_rebuilt_from_spec", reason=str(exc)[:200])
        model = build_from_spec(json.loads(spec_path.read_text(encoding="utf-8")))
        model.load_weights(weights_path)
    labels = load_labels(Path(f"{stem}.labels.json"))
    scaler = load_scaler(Path(f"{stem}.scaler.joblib"))
    return model, labels, scaler


def load_model_meta(language: str, config_path: str | None = None) -> dict[str, Any]:
    """Metadata of the latest model (feature mode, window size/seconds, …).

    Reads ``models/<lang>_latest.json``; returns ``{}`` when there is none.
    """
    cfg = load_config(config_path)
    pointer = cfg.resolve(cfg.paths.models_dir) / f"{language}_latest.json"
    if not pointer.is_file():
        return {}
    meta: dict[str, Any] = json.loads(pointer.read_text(encoding="utf-8"))
    return meta


def load_feature_mode(language: str, config_path: str | None = None) -> str:
    """Feature mode the latest model was trained with (``"holistic"`` if unset).

    Models trained before feature modes existed have no entry and were
    trained with the holistic features.
    """
    cfg = load_config(config_path)
    pointer = cfg.resolve(cfg.paths.models_dir) / f"{language}_latest.json"
    if not pointer.is_file():
        return "holistic"
    return str(json.loads(pointer.read_text(encoding="utf-8")).get("feature_mode", "holistic"))


def main() -> None:
    """Console entry point for ``sign-eval``: re-evaluate latest model."""
    configure_logging()
    parser = argparse.ArgumentParser(prog="sign-eval")
    parser.add_argument("--language", choices=["asl", "psl"], required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    from sign_bridge.data.preprocess import apply_scaler, load_dataset

    cfg = load_config(args.config)
    model, labels, scaler = load_latest_model(args.language, args.config)
    ds = load_dataset(cfg.resolve(cfg.paths.data_dir) / args.language,
                      cfg.landmarks.window_size,
                      load_feature_mode(args.language, args.config))
    x_raw, y = ds.x, ds.y
    if ds.labels != labels:
        # Class indices differ between disk and model: map by gloss name and
        # drop signs the model was never trained on.
        log.warning("label_mismatch", disk=ds.labels, model=labels)
        model_index = {lbl: i for i, lbl in enumerate(labels)}
        keep = np.array([ds.labels[c] in model_index for c in ds.y], dtype=bool)
        if not keep.any():
            raise SystemExit("None of the recorded signs are in the trained model.")
        x_raw = ds.x[keep]
        y = np.array([model_index[ds.labels[c]] for c in ds.y[keep]], dtype=np.int64)
    x = apply_scaler(x_raw, scaler)
    report = evaluate(model, x, y, labels,
                      output_stem=cfg.resolve(cfg.paths.models_dir) / f"{args.language}_reval")
    print(json.dumps({"accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
