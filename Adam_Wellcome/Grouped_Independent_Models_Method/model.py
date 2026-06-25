from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    dilation: int,
    dropout: float,
    negative_slope: float,
) -> nn.Sequential:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        ),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Dropout(dropout),
    )


class IndependentAdamWellcomeCNN1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        input_length: int,
        num_classes: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_sizes: tuple[int, ...] = (5, 5, 3, 3),
        strides: tuple[int, ...] = (2, 2, 1, 1),
        dilations: tuple[int, ...] = (1, 1, 1, 1),
        dropout: float = 0.1,
        negative_slope: float = 0.05,
        classifier_hidden: int = 128,
    ):
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be >= 1")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if not (len(channels) == len(kernel_sizes) == len(strides) == len(dilations)):
            raise ValueError("channels, kernel_sizes, strides, and dilations must have the same length")

        blocks = []
        current_in = in_channels
        for out_channels, kernel_size, stride, dilation in zip(channels, kernel_sizes, strides, dilations):
            blocks.append(
                _conv_block(
                    current_in,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            )
            current_in = out_channels

        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            was_training = self.features.training
            self.features.eval()
            feature_example = torch.zeros(1, in_channels, input_length)
            flattened_dim = int(self.features(feature_example).flatten(start_dim=1).shape[1])
            self.features.train(was_training)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, classifier_hidden),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
