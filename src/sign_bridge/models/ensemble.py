"""Several trained models answering together.

The answer is the average of the members' probabilities. Each member keeps its
own scaler (models trained in different runs fit their scalers on differently
augmented data), so the ensemble takes *unscaled* input and is paired with
:class:`IdentityScaler` wherever the app would normally scale first.

Measured on the 2,703 ASL Citizen test clips (11 signers never seen in
training): run B 91.64%, run C 92.64%, both averaged 93.23% top-1. See the
results log.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from sign_bridge.typing import NDArray


class IdentityScaler:
    """A scaler that leaves the input as it is (the members scale for themselves)."""

    def transform(self, x: NDArray) -> NDArray:
        return x


class AveragedModel:
    """Average of several ``(model, scaler)`` members' probabilities.

    Args:
        members: ``(keras model, fitted scaler)`` pairs; every model must take the
            same input shape and answer over the same labels, in the same order.
        weights: How much each member counts (default: equally). Normalized to sum 1.

    Raises:
        ValueError: No members, mismatched input shapes, or a bad weight list.
    """

    def __init__(self, members: Sequence[tuple[Any, Any]],
                 weights: Sequence[float] | None = None) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one model")
        shapes = {tuple(m.input_shape) for m, _ in members}
        if len(shapes) != 1:
            raise ValueError(f"members take different inputs: {sorted(shapes)}")
        w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=np.float64)
        if w.shape != (len(members),) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be one non-negative number per member")
        self._members = list(members)
        self._weights = w / w.sum()
        self.input_shape = next(iter(shapes))

    def __len__(self) -> int:
        return len(self._members)

    def _scaled(self, x: NDArray, scaler: Any) -> NDArray:
        flat = scaler.transform(np.asarray(x).reshape(-1, x.shape[-1]))
        return np.asarray(flat, dtype=np.float32).reshape(x.shape)

    def predict_on_batch(self, x: NDArray) -> NDArray:
        """Averaged probabilities for one batch of *unscaled* windows."""
        out = None
        for (model, scaler), w in zip(self._members, self._weights, strict=True):
            p = w * np.asarray(model.predict_on_batch(self._scaled(x, scaler)), dtype=np.float64)
            out = p if out is None else out + p
        assert out is not None
        result: NDArray = out.astype(np.float32)
        return result

    def predict(self, x: NDArray, batch_size: int = 256, verbose: int = 0) -> NDArray:
        """Like :meth:`predict_on_batch`, for any number of windows."""
        out = None
        for (model, scaler), w in zip(self._members, self._weights, strict=True):
            p = model.predict(self._scaled(x, scaler), batch_size=batch_size, verbose=verbose)
            p = w * np.asarray(p, dtype=np.float64)
            out = p if out is None else out + p
        assert out is not None
        result: NDArray = out.astype(np.float32)
        return result
