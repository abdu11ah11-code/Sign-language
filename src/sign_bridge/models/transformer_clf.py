"""Transformer-encoder sign classifier — the attention-based alternative.

Architecture: linear projection to d_model → fixed sinusoidal positional
encoding → N pre-norm encoder blocks (MHA + feed-forward, residual) →
global average pooling → softmax. With 30-frame windows the sequence is
short, so a small 2-block encoder matches the LSTM's speed while sometimes
generalizing better on motion-heavy signs. Select with
``sign-train --architecture transformer``.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="sign_bridge")
class PositionalEncoding(keras.layers.Layer):
    """Adds fixed sinusoidal positional encodings (Vaswani et al., 2017)."""

    def __init__(self, max_len: int, d_model: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        pos = np.arange(max_len)[:, None].astype(np.float32)
        i = np.arange(d_model)[None, :].astype(np.float32)
        angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
        enc = np.zeros((max_len, d_model), dtype=np.float32)
        enc[:, 0::2] = np.sin(angle[:, 0::2])
        enc[:, 1::2] = np.cos(angle[:, 1::2])
        self._encoding = tf.constant(enc[None, ...])

    def call(self, x: tf.Tensor) -> tf.Tensor:  # noqa: D102 (Keras API)
        return x + self._encoding[:, : tf.shape(x)[1], :]

    def get_config(self) -> dict[str, object]:  # noqa: D102 (Keras API)
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg  # type: ignore[no-any-return]


def _encoder_block(
    x: tf.Tensor, d_model: int, num_heads: int, ff_dim: int, dropout: float, idx: int
) -> tf.Tensor:
    """One pre-norm transformer encoder block with residual connections."""
    norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{idx}")(x)
    attn = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, name=f"mha_{idx}"
    )(norm1, norm1)
    attn = keras.layers.Dropout(dropout, name=f"drop_attn_{idx}")(attn)
    x = keras.layers.Add(name=f"res_attn_{idx}")([x, attn])

    norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{idx}")(x)
    ff = keras.layers.Dense(ff_dim, activation="relu", name=f"ff1_{idx}")(norm2)
    ff = keras.layers.Dense(d_model, name=f"ff2_{idx}")(ff)
    ff = keras.layers.Dropout(dropout, name=f"drop_ff_{idx}")(ff)
    return keras.layers.Add(name=f"res_ff_{idx}")([x, ff])


def build_transformer(
    num_classes: int,
    window_size: int = 30,
    feature_dim: int = 162,
    d_model: int = 96,
    num_heads: int = 4,
    ff_dim: int = 192,
    num_blocks: int = 2,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile the transformer classifier.

    Args mirror :func:`sign_bridge.models.lstm_classifier.build_lstm`,
    plus the encoder hyperparameters (config §model.transformer).

    Returns:
        A compiled ``keras.Model``.
    """
    inputs = keras.Input(shape=(window_size, feature_dim), name="landmark_window")
    x = keras.layers.Dense(d_model, name="proj")(inputs)
    x = PositionalEncoding(window_size, d_model, name="pos_enc")(x)
    for i in range(num_blocks):
        x = _encoder_block(x, d_model, num_heads, ff_dim, dropout, i)
    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="drop_head")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="sign")(x)

    model = keras.Model(inputs, outputs, name="sign_transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
