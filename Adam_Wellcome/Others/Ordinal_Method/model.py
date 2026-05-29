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


class OrdinalAdamWellcomeCNN1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_liquids: int,
        num_concentrations: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_sizes: tuple[int, ...] = (5, 5, 3, 3),
        strides: tuple[int, ...] = (1, 1, 1, 1),
        dropout: float = 0.1,
        negative_slope: float = 0.05,
        classifier_hidden: int = 128,
    ):
        super().__init__()
        if num_liquids < 2:
            raise ValueError("num_liquids must be >= 2")
        if num_concentrations < 2:
            raise ValueError("num_concentrations must be >= 2")
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("channels, kernel_sizes, and strides must have the same length")

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
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(classifier_hidden),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(dropout),
        )
        self.liquid_head = nn.Linear(classifier_hidden, num_liquids)
        self.concentration_head = nn.Linear(classifier_hidden, num_concentrations - 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.embedding(x)
        return self.liquid_head(x), self.concentration_head(x)
