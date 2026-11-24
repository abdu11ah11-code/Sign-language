"""LSTM sign classifier — the primary architecture.

Spec: LSTM(128) → Dropout(0.3) → LSTM(64) → Dense(64, relu) → Dropout(0.3)
→ Dense(N, softmax), consuming windows of shape ``(window_size, 162)``.
Small enough to train on CPU with a few hundred clips per class and fast
enough (<10 ms on modern laptops) for the every-5-frames inference cadence.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


def build_lstm(
    num_classes: int,
    window_size: int = 30,
    feature_dim: int = 162,
    lstm_units: tuple[int, ...] = (128, 64),
    dense_units: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile the LSTM classifier.

    Args:
        num_classes: Output vocabulary size for this language.
        window_size: Frames per input window.
        feature_dim: Features per frame (162 = hands + upper pose).
        lstm_units: Units per stacked LSTM layer; all but the last return
            sequences.
        dense_units: Width of the pre-softmax dense layer.
        dropout: Dropout rate applied after each LSTM and the dense layer.
        learning_rate: Adam learning rate.

    Returns:
        A compiled ``keras.Model`` (sparse categorical crossentropy).
    """
    inputs = keras.Input(shape=(window_size, feature_dim), name="landmark_window")
    x: tf.Tensor = inputs
    for i, units in enumerate(lstm_units):
        last = i == len(lstm_units) - 1
        x = keras.layers.LSTM(units, return_sequences=not last, name=f"lstm_{i}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_lstm_{i}")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(dropout, name="drop_dense")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="sign")(x)

    model = keras.Model(inputs, outputs, name="sign_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
