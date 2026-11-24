"""Conv1D-Transformer sign classifier — SignBridge's strongest model.

A hybrid in the style of the 1st-place solution to Google's *Isolated Sign
Language Recognition* Kaggle competition (ASL Signs, 250 signs, MediaPipe
landmarks), which beat pure Transformer and recurrent models there:

    window (T, F)
      → motion features: concat(x, x[t] − x[t−k] for each lag k)   (T, (1+L)·F)
      → Dense(dim) + BatchNorm
      → stack × S:
            Conv1D block × C   depthwise temporal conv (kernel k) with
                               efficient channel attention (ECA), residual
            Transformer block  multi-head self-attention + feed-forward,
                               pre-norm, residual
      → Dense(2·dim, swish) → global average pool → late dropout → softmax

Why the hybrid: depthwise convolutions capture *local* motion (a finger
flick over a few frames) cheaply and also inject relative position, so no
positional encoding is needed; self-attention then relates *distant*
moments of the sign (start and end handshape). Regularization follows the
same recipe: stochastic-depth-style residual dropout, heavy late dropout,
label smoothing and AdamW weight decay.

Select with ``sign-train --architecture conv_transformer``.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="sign_bridge")
class TemporalDelta(keras.layers.Layer):
    """Append motion: ``concat(x, x_t - x_{t-k} for k in lags)``.

    ``lags=(1,)`` (the default, and what models saved before 0.14 use) adds the
    frame-to-frame change; ``(1, 2)`` also the change over two frames, which
    is less noisy for slow movements. The first frames repeat frame 0.
    """

    def __init__(self, lags: tuple[int, ...] | list[int] = (1,), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lags = tuple(int(k) for k in lags)

    def call(self, x: tf.Tensor) -> tf.Tensor:  # noqa: D102 (Keras API)
        parts = [x]
        for k in self.lags:
            prev = tf.concat([tf.repeat(x[:, :1], k, axis=1), x[:, :-k]], axis=1)
            parts.append(x - prev)
        return tf.concat(parts, axis=-1)

    def compute_output_shape(self, input_shape: Any) -> Any:  # noqa: D102
        f = input_shape[-1]
        return (*input_shape[:-1], None if f is None else (1 + len(self.lags)) * f)

    def get_config(self) -> dict[str, Any]:  # noqa: D102
        cfg: dict[str, Any] = super().get_config()
        cfg.update({"lags": list(self.lags)})
        return cfg


@keras.utils.register_keras_serializable(package="sign_bridge")
class DepthwiseTemporalConv(keras.layers.Layer):
    """Depthwise 1-D convolution over time (one kernel per channel).

    Implemented on ``tf.nn.depthwise_conv2d`` rather than Keras'
    ``DepthwiseConv1D``, which crashes (segmentation fault) with
    TensorFlow 2.16 on CPU — the configuration SignBridge runs live on.
    """

    def __init__(self, kernel_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape: Any) -> None:  # noqa: D102
        channels = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel", shape=(self.kernel_size, 1, channels, 1),
            initializer="glorot_uniform", trainable=True,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:  # noqa: D102
        y = tf.nn.depthwise_conv2d(x[:, :, None, :], self.kernel, [1, 1, 1, 1], "SAME")
        return y[:, :, 0, :]

    def get_config(self) -> dict[str, Any]:  # noqa: D102
        cfg: dict[str, Any] = super().get_config()
        cfg.update({"kernel_size": self.kernel_size})
        return cfg


@keras.utils.register_keras_serializable(package="sign_bridge")
class ECA(keras.layers.Layer):
    """Efficient Channel Attention (Wang et al., 2020) over a ``(B, T, C)`` tensor."""

    def __init__(self, kernel_size: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = keras.layers.Conv1D(1, kernel_size, padding="same", use_bias=False)

    def build(self, input_shape: Any) -> None:  # noqa: D102
        self.conv.build((None, input_shape[-1], 1))
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:  # noqa: D102
        w = tf.reduce_mean(x, axis=1)[..., None]          # (B, C, 1)
        w = tf.sigmoid(self.conv(w))[:, None, :, 0]        # (B, 1, C)
        return x * w

    def get_config(self) -> dict[str, Any]:  # noqa: D102
        cfg: dict[str, Any] = super().get_config()
        cfg.update({"kernel_size": self.kernel_size})
        return cfg


@keras.utils.register_keras_serializable(package="sign_bridge")
class SparseSmoothedCrossentropy(keras.losses.Loss):
    """Sparse categorical cross-entropy with label smoothing (integer labels)."""

    def __init__(self, label_smoothing: float = 0.1, name: str = "sparse_smoothed_ce",
                 **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:  # noqa: D102
        n = tf.shape(y_pred)[-1]
        onehot = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), n)
        return keras.losses.categorical_crossentropy(
            onehot, y_pred, label_smoothing=self.label_smoothing
        )

    def get_config(self) -> dict[str, Any]:  # noqa: D102
        cfg: dict[str, Any] = super().get_config()
        cfg.update({"label_smoothing": self.label_smoothing})
        return cfg


def _residual_dropout(rate: float, name: str) -> keras.layers.Layer:
    """Drop a whole residual branch per sample (stochastic-depth style)."""
    return keras.layers.Dropout(rate, noise_shape=(None, 1, 1), name=name)


def _conv_block(x: tf.Tensor, dim: int, kernel: int, drop: float, expand: int,
                name: str) -> tf.Tensor:
    skip = x
    h = keras.layers.Dense(dim * expand, activation="swish", name=f"{name}_expand")(x)
    h = DepthwiseTemporalConv(kernel, name=f"{name}_dwconv")(h)
    h = keras.layers.BatchNormalization(momentum=0.95, name=f"{name}_bn")(h)
    h = ECA(name=f"{name}_eca")(h)
    h = keras.layers.Dense(dim, name=f"{name}_project")(h)
    h = _residual_dropout(drop, f"{name}_droppath")(h)
    return keras.layers.Add(name=f"{name}_add")([skip, h])


def _transformer_block(x: tf.Tensor, dim: int, heads: int, expand: int, drop: float,
                       name: str) -> tf.Tensor:
    h = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    h = keras.layers.MultiHeadAttention(heads, key_dim=dim // heads, dropout=0.2,
                                        name=f"{name}_mha")(h, h)
    h = _residual_dropout(drop, f"{name}_droppath1")(h)
    x = keras.layers.Add(name=f"{name}_add1")([x, h])
    h = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    h = keras.layers.Dense(dim * expand, activation="swish", name=f"{name}_ff1")(h)
    h = keras.layers.Dense(dim, name=f"{name}_ff2")(h)
    h = _residual_dropout(drop, f"{name}_droppath2")(h)
    return keras.layers.Add(name=f"{name}_add2")([x, h])


def build_conv_transformer(
    num_classes: int,
    window_size: int = 64,
    feature_dim: int = 188,
    dim: int = 192,
    kernel_size: int = 17,
    num_stacks: int = 2,
    convs_per_stack: int = 3,
    num_heads: int = 4,
    expand: int = 2,
    drop_path: float = 0.2,
    late_dropout: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    motion_lags: tuple[int, ...] | list[int] = (1,),
) -> keras.Model:
    """Build and compile the Conv1D-Transformer classifier.

    Args:
        num_classes: Output vocabulary size.
        window_size: Frames per input window.
        feature_dim: Values per frame (188 for ``holistic_lips``).
        dim: Channel width of every block.
        kernel_size: Temporal kernel of the depthwise convolutions.
        num_stacks: Number of (conv × C + transformer) stacks.
        convs_per_stack: Conv1D blocks before each transformer block.
        num_heads: Attention heads.
        expand: Width multiplier inside conv/feed-forward blocks.
        drop_path: Per-sample residual-branch dropout.
        late_dropout: Dropout before the classifier.
        learning_rate / weight_decay: AdamW settings.
        label_smoothing: Target smoothing for the loss.
        motion_lags: Frame gaps of the motion features (see :class:`TemporalDelta`).

    Returns:
        A compiled ``keras.Model``.
    """
    inputs = keras.Input(shape=(window_size, feature_dim), name="landmark_window")
    x = TemporalDelta(motion_lags, name="motion")(inputs)
    x = keras.layers.Dense(dim, use_bias=False, name="stem")(x)
    x = keras.layers.BatchNormalization(momentum=0.95, name="stem_bn")(x)
    for s in range(num_stacks):
        for c in range(convs_per_stack):
            x = _conv_block(x, dim, kernel_size, drop_path, expand, f"s{s}_conv{c}")
        x = _transformer_block(x, dim, num_heads, expand, drop_path, f"s{s}_attn")
    x = keras.layers.Dense(dim * 2, activation="swish", name="top")(x)
    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = keras.layers.Dropout(late_dropout, name="late_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="sign")(x)

    model = keras.Model(inputs, outputs, name="sign_conv_transformer")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=SparseSmoothedCrossentropy(label_smoothing),
        metrics=["accuracy"],
    )
    return model
