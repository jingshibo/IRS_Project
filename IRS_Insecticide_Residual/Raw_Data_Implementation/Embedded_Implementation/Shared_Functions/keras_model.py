from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf


@dataclass(frozen=True)
class SharedBackboneConfig:
    """Architecture defaults matching Models/Model_Structure.py."""

    input_length: int = 540
    in_channels: int = 3
    num_classes: int = 3
    conv_channels: Sequence[int] = (32, 64, 128, 256)
    kernel_sizes: Sequence[int] = (5, 5, 5, 5)
    strides: Sequence[int] = (2, 2, 2, 2)
    dilations: Sequence[int] = (1, 1, 1, 1)
    use_pool: Sequence[bool] = (True, True, True, True)
    pool_kernel_sizes: Sequence[int] = (3, 3, 3, 3)
    pool_strides: Sequence[int] = (2, 2, 2, 2)
    classifier_hidden: Sequence[int] = (512, 256)
    dropout: float = 0.1
    leaky_relu_slope: float = 0.05
    batch_norm_epsilon: float = 1e-5
    batch_norm_momentum: float = 0.9


def pytorch_same_padding(kernel_size: int, stride: int, dilation: int) -> int:
    """Return the symmetric padding used by Model_Structure._same_padding."""
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def _validate_config(config: SharedBackboneConfig) -> None:
    expected_blocks = 4
    fields = {
        "conv_channels": config.conv_channels,
        "kernel_sizes": config.kernel_sizes,
        "strides": config.strides,
        "dilations": config.dilations,
        "use_pool": config.use_pool,
        "pool_kernel_sizes": config.pool_kernel_sizes,
        "pool_strides": config.pool_strides,
    }
    for name, values in fields.items():
        if len(values) != expected_blocks:
            raise ValueError(f"{name} must contain {expected_blocks} values, got {len(values)}")
    if len(config.classifier_hidden) != 2:
        raise ValueError(f"classifier_hidden must contain 2 values, got {len(config.classifier_hidden)}")


def conv_bn_lrelu_pool(
    x: tf.Tensor,
    out_channels: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    use_pool: bool,
    pool_kernel_size: int,
    pool_stride: int,
    leaky_relu_slope: float,
    batch_norm_epsilon: float,
    batch_norm_momentum: float,
    block_idx: int,
) -> tf.Tensor:
    """Keras equivalent of the PyTorch Conv1d-BN-LeakyReLU-Pool block.
    This code uses the Keras Functional API, which defines the data flow directly, without needing to define a class.
    It is different from PyTorch, which typically defines a class for the model first. """
    pad = pytorch_same_padding(kernel_size=kernel_size, stride=stride, dilation=dilation)
    if pad > 0:
        x = tf.keras.layers.ZeroPadding1D(padding=pad, name=f"features_{block_idx}_pad")(x)
    x = tf.keras.layers.Conv1D(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=dilation,
        padding="valid",
        use_bias=True,
        name=f"features_{block_idx}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        epsilon=batch_norm_epsilon,
        momentum=batch_norm_momentum,
        name=f"features_{block_idx}_bn",
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_slope, name=f"features_{block_idx}_leaky_relu")(x)
    if use_pool:
        x = tf.keras.layers.MaxPooling1D(
            pool_size=pool_kernel_size,
            strides=pool_stride,
            padding="valid",
            name=f"features_{block_idx}_pool",
        )(x)
    return x


def build_shared_backbone_keras_model(
    config: SharedBackboneConfig | None = None,
    include_softmax: bool = False,
    match_pytorch_flatten: bool = False,
) -> tf.keras.Model:
    """Build the Keras equivalent model structure of PyTorch OneDCNNClassifier.
     It can either: receive copied PyTorch weights in Pytorch_to_Keras_Method, or
     be trained directly in Retrain_Keras_Method, then exported to .TFLite.

    The Keras input layout is [batch, length, channels]. The PyTorch model uses
    [batch, channels, length]. Set `match_pytorch_flatten=True` only when
    transferring PyTorch Dense weights exactly; the direct Keras deployment path
    leaves the tensor in native Keras layout to avoid an extra TFLite transpose.
    """
    config = config or SharedBackboneConfig() # Uses the provided model settings, or falls back to defaults.
    _validate_config(config) # checks that the model config is structurally valid.

    inputs = tf.keras.Input( # Instantiate a Keras tensor as the Keras model's input
        shape=(config.input_length, config.in_channels),
        name="signal", # name="signal" gives the Keras input a readable name, not an automatic name like: input_1
    )
    x = inputs # The input shape of Keras should be [batch, length, channels].

    # Builds the convolution feature extractor. Each block is roughly: Conv1D -> BatchNorm -> LeakyReLU -> MaxPool to mirror the Pytroch CNN backdone.
    # Note that the tensor order in Keras is: [batch, length, channels]. But in Pytorch, the tensor order is: [batch, channels, length].
    # To match the convolution weights when transferring from PyTorch to Keras, we need to manually transpose the Pytorch Conv1D weights.
    for block_idx, out_channels in enumerate(config.conv_channels):
        x = conv_bn_lrelu_pool(
            x=x,
            out_channels=out_channels,
            kernel_size=config.kernel_sizes[block_idx],
            stride=config.strides[block_idx],
            dilation=config.dilations[block_idx],
            use_pool=config.use_pool[block_idx],
            pool_kernel_size=config.pool_kernel_sizes[block_idx],
            pool_stride=config.pool_strides[block_idx],
            leaky_relu_slope=config.leaky_relu_slope,
            batch_norm_epsilon=config.batch_norm_epsilon,
            batch_norm_momentum=config.batch_norm_momentum,
            block_idx=block_idx,
        )

    # The problem is the Flatten before Dense layer. Flatten does not just remove input dimensions. It also fixes a specific ordering of the values.
    # if you copy PyTorch Dense weights directly into Keras, the Dense layer expects the flattened input vector to be in the same order as PyTorch.
    # So we need to transpose the Keras' input order to match PyTorch: [batch, channels, length] to make the flattened vector consistent.
    if match_pytorch_flatten:
        x = tf.keras.layers.Permute((2, 1), name="features_pytorch_layout")(x) # exchange length and channels order
    x = tf.keras.layers.Flatten(name="classifier_flatten")(x) # Now Keras flatten order matches PyTorch flatten order.

    # Build the remaining classifier head. Each block is roughly: Dense -> BatchNorm -> LeakyReLU -> Dropout to mirror the Pytroch CNN classifier.
    # Dense block 1
    x = tf.keras.layers.Dense(config.classifier_hidden[0], use_bias=True, name="classifier_dense_0")(x)
    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        epsilon=config.batch_norm_epsilon,
        momentum=config.batch_norm_momentum,
        name="classifier_bn_0",
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=config.leaky_relu_slope, name="classifier_leaky_relu_0")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="classifier_dropout_0")(x)

    # Dense block 2
    x = tf.keras.layers.Dense(config.classifier_hidden[1], use_bias=True, name="classifier_dense_1")(x)
    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        epsilon=config.batch_norm_epsilon,
        momentum=config.batch_norm_momentum,
        name="classifier_bn_1",
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=config.leaky_relu_slope, name="classifier_leaky_relu_1")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="classifier_dropout_1")(x)

    # Final output layer, produces raw class logits. If include_softmax=True, it returns probabilities.
    logits = tf.keras.layers.Dense(config.num_classes, use_bias=True, name="classifier_logits")(x)
    outputs = tf.keras.layers.Softmax(name="probabilities")(logits) if include_softmax else logits
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="shared_backbone_keras")


def torch_to_keras_input(x):
    """Convert a NumPy/PyTorch-like [N, C, L] array to Keras [N, L, C]."""
    return x.transpose(0, 2, 1)
