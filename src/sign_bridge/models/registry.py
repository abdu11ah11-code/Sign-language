"""One place that knows how to build every SignBridge architecture.

A *spec* is a small JSON-able dict that fully describes a model::

    {"architecture": "conv_transformer", "num_classes": 250,
     "window_size": 64, "feature_dim": 188, "params": {...}}

The trainer and the Kaggle notebook save it next to each model
(``<stem>.spec.json``) together with the weights. If a ``.keras`` file
saved by one Keras version cannot be deserialized by another, the loader
rebuilds the model from the spec and loads the weights instead.
"""

from __future__ import annotations

from typing import Any

from sign_bridge.config import AppConfig

ARCHITECTURES = ("lstm", "transformer", "conv_transformer")


def default_params(architecture: str, cfg: AppConfig) -> dict[str, Any]:
    """Hyperparameters for ``architecture`` taken from the config."""
    m, t = cfg.model, cfg.training
    if architecture == "lstm":
        return {"lstm_units": list(m.lstm_units), "dense_units": m.dense_units,
                "dropout": m.dropout, "learning_rate": t.learning_rate}
    if architecture == "transformer":
        tr = m.transformer
        return {"d_model": tr.d_model, "num_heads": tr.num_heads, "ff_dim": tr.ff_dim,
                "num_blocks": tr.num_blocks, "dropout": m.dropout,
                "learning_rate": t.learning_rate}
    if architecture == "conv_transformer":
        ct = m.conv_transformer
        return {"dim": ct.dim, "kernel_size": ct.kernel_size, "num_stacks": ct.num_stacks,
                "convs_per_stack": ct.convs_per_stack, "num_heads": ct.num_heads,
                "expand": ct.expand, "drop_path": ct.drop_path,
                "late_dropout": ct.late_dropout, "weight_decay": ct.weight_decay,
                "label_smoothing": ct.label_smoothing, "learning_rate": t.learning_rate}
    raise ValueError(f"Unknown architecture {architecture!r}; expected one of {ARCHITECTURES}")


def make_spec(architecture: str, num_classes: int, window_size: int, feature_dim: int,
              params: dict[str, Any]) -> dict[str, Any]:
    """Bundle everything needed to rebuild a model."""
    return {"architecture": architecture, "num_classes": int(num_classes),
            "window_size": int(window_size), "feature_dim": int(feature_dim),
            "params": dict(params)}


def build_from_spec(spec: dict[str, Any]) -> Any:
    """Construct (and compile) the model a spec describes."""
    arch = spec["architecture"]
    common = {"num_classes": spec["num_classes"], "window_size": spec["window_size"],
              "feature_dim": spec["feature_dim"]}
    params = dict(spec.get("params", {}))
    if arch == "lstm":
        from sign_bridge.models.lstm_classifier import build_lstm

        params["lstm_units"] = tuple(params.get("lstm_units", (128, 64)))
        return build_lstm(**common, **params)
    if arch == "transformer":
        from sign_bridge.models.transformer_clf import build_transformer

        return build_transformer(**common, **params)
    if arch == "conv_transformer":
        from sign_bridge.models.conv_transformer import build_conv_transformer

        return build_conv_transformer(**common, **params)
    raise ValueError(f"Unknown architecture {arch!r}; expected one of {ARCHITECTURES}")


def register_custom_objects() -> None:
    """Import every module that registers custom Keras layers/losses."""
    import sign_bridge.models.conv_transformer  # noqa: F401
    import sign_bridge.models.transformer_clf  # noqa: F401
