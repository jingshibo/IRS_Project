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
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Dropout(dropout),
    )


class AdamWellcomeCNN1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        kernel_sizes: tuple[int, int, int, int] = (7, 5, 5, 3),
        strides: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout: float = 0.1,
        negative_slope: float = 0.05,
        classifier_hidden: int = 256,
    ):
        super().__init__()
        if len(channels) != 4 or len(kernel_sizes) != 4 or len(strides) != 4:
            raise ValueError("channels, kernel_sizes, and strides must all have length 4")

        blocks = []
        current_in = in_channels
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            blocks.append(
                _conv_block(
                    current_in,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            )
            current_in = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(classifier_hidden),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

