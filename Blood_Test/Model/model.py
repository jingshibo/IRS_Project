from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    dropout: float,
    negative_slope: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        ),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Dropout(dropout),
    )


class BloodConcentrationCNN1D(nn.Module):
    """Small single-head 1D CNN for three-class concentration prediction."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_sizes: tuple[int, ...] = (5, 5, 3),
        strides: tuple[int, ...] = (2, 2, 1),
        dropout: float = 0.3,
        negative_slope: float = 0.05,
        classifier_hidden: int = 64,
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(
                "channels, kernel_sizes, and strides must have the same length"
            )

        blocks = []
        current_channels = in_channels
        for output_channels, kernel_size, stride in zip(
            channels,
            kernel_sizes,
            strides,
        ):
            blocks.append(
                _conv_block(
                    current_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            )
            current_channels = output_channels

        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(classifier_hidden),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

