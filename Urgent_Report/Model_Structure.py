from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


DEFAULT_CHANNELS = (32, 64, 128, 256)
DEFAULT_KERNEL_SIZES = (5, 5, 5, 5)
DEFAULT_STRIDES = (2, 2, 2, 2)
DEFAULT_DILATIONS = (1, 1, 1, 1)
DEFAULT_USE_POOL = (True, True, True, True)
DEFAULT_POOL_TYPES = ("max", "max", "max", "max")
DEFAULT_POOL_KERNEL_SIZES = (3, 3, 3, 3)
DEFAULT_POOL_STRIDES = (2, 2, 2, 2)
DEFAULT_POOL_PADDINGS = (0, 0, 0, 0)
DEFAULT_CLASSIFIER_HIDDEN = (512, 256)


def _same_padding(kernel_size: int, stride: int, dilation: int) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def _expand_to_depth(name: str, values: Sequence, depth: int) -> tuple:
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if len(values) < depth:
        raise ValueError(f"{name} must contain at least {depth} values, got {len(values)}")
    return tuple(values[:depth])


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    use_pool: bool,
    pool_type: str,
    pool_kernel_size: int,
    pool_stride: int,
    pool_padding: int,
    leaky_relu_slope: float,
) -> nn.Sequential:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")

    layers: list[nn.Module] = [
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=_same_padding(kernel_size=kernel_size, stride=stride, dilation=dilation),
        ),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
    ]

    if use_pool:
        if pool_type == "max":
            layers.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding))
        elif pool_type == "avg":
            layers.append(nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding))
        else:
            raise ValueError(f"Unsupported pool_type='{pool_type}'. Use 'max' or 'avg'.")

    return nn.Sequential(*layers)


class FlexibleCNNFeatureExtractor(nn.Module):
    """1D CNN feature extractor with configurable number of convolution blocks."""

    def __init__(
        self,
        in_channels: int,
        cnn_layers: int = 4,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        kernel_sizes: Sequence[int] = DEFAULT_KERNEL_SIZES,
        strides: Sequence[int] = DEFAULT_STRIDES,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        use_pool: Sequence[bool] = DEFAULT_USE_POOL,
        pool_types: Sequence[str] = DEFAULT_POOL_TYPES,
        pool_kernel_sizes: Sequence[int] = DEFAULT_POOL_KERNEL_SIZES,
        pool_strides: Sequence[int] = DEFAULT_POOL_STRIDES,
        pool_paddings: Sequence[int] = DEFAULT_POOL_PADDINGS,
        leaky_relu_slope: float = 0.05,
    ) -> None:
        super().__init__()
        channels = _expand_to_depth("channels", channels, cnn_layers)
        kernel_sizes = _expand_to_depth("kernel_sizes", kernel_sizes, cnn_layers)
        strides = _expand_to_depth("strides", strides, cnn_layers)
        dilations = _expand_to_depth("dilations", dilations, cnn_layers)
        use_pool = _expand_to_depth("use_pool", use_pool, cnn_layers)
        pool_types = _expand_to_depth("pool_types", pool_types, cnn_layers)
        pool_kernel_sizes = _expand_to_depth("pool_kernel_sizes", pool_kernel_sizes, cnn_layers)
        pool_strides = _expand_to_depth("pool_strides", pool_strides, cnn_layers)
        pool_paddings = _expand_to_depth("pool_paddings", pool_paddings, cnn_layers)

        blocks = []
        previous_channels = in_channels
        for idx in range(cnn_layers):
            blocks.append(
                _conv_block(
                    in_channels=previous_channels,
                    out_channels=channels[idx],
                    kernel_size=kernel_sizes[idx],
                    stride=strides[idx],
                    dilation=dilations[idx],
                    use_pool=use_pool[idx],
                    pool_type=pool_types[idx],
                    pool_kernel_size=pool_kernel_sizes[idx],
                    pool_stride=pool_strides[idx],
                    pool_padding=pool_paddings[idx],
                    leaky_relu_slope=leaky_relu_slope,
                )
            )
            previous_channels = channels[idx]

        self.features = nn.Sequential(*blocks)
        self.out_channels = previous_channels
        self.cnn_layers = cnn_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class FlexibleOneDCNNClassifier(nn.Module):
    """One-dimensional CNN classifier with adjustable CNN depth.

    Use ``cnn_layers`` to select how many convolution blocks are used. The default
    value of 4 mirrors the current project model.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 3,
        cnn_layers: int = 4,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        kernel_sizes: Sequence[int] = DEFAULT_KERNEL_SIZES,
        strides: Sequence[int] = DEFAULT_STRIDES,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        use_pool: Sequence[bool] = DEFAULT_USE_POOL,
        pool_types: Sequence[str] = DEFAULT_POOL_TYPES,
        pool_kernel_sizes: Sequence[int] = DEFAULT_POOL_KERNEL_SIZES,
        pool_strides: Sequence[int] = DEFAULT_POOL_STRIDES,
        pool_paddings: Sequence[int] = DEFAULT_POOL_PADDINGS,
        classifier_hidden: Sequence[int] = DEFAULT_CLASSIFIER_HIDDEN,
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.05,
    ) -> None:
        super().__init__()
        self.features = FlexibleCNNFeatureExtractor(
            in_channels=in_channels,
            cnn_layers=cnn_layers,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            dilations=dilations,
            use_pool=use_pool,
            pool_types=pool_types,
            pool_kernel_sizes=pool_kernel_sizes,
            pool_strides=pool_strides,
            pool_paddings=pool_paddings,
            leaky_relu_slope=leaky_relu_slope,
        )

        classifier_layers: list[nn.Module] = [nn.Flatten()]
        for hidden_units in classifier_hidden:
            classifier_layers.extend(
                [
                    nn.LazyLinear(hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
        classifier_layers.append(nn.LazyLinear(num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = FlexibleOneDCNNClassifier(in_channels=3, num_classes=3, cnn_layers=3)
    x = torch.randn(8, 3, 820)
    y = model(x)
    print(y.shape)
