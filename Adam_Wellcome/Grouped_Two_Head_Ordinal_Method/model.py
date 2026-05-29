from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GroupedOrdinalAdamWellcomeCNN1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_liquids: int,
        num_concentration_groups: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_sizes: tuple[int, ...] = (5, 5, 3, 3),
        strides: tuple[int, ...] = (2, 2, 1, 1),
        dropout: float = 0.3,
        negative_slope: float = 0.05,
        classifier_hidden: int = 128,
        strict_ordinal: bool = False,
    ):
        super().__init__()
        if num_liquids < 2:
            raise ValueError("num_liquids must be >= 2")
        if num_concentration_groups < 2:
            raise ValueError("num_concentration_groups must be >= 2")
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
        self.strict_ordinal = strict_ordinal
        self.num_concentration_groups = num_concentration_groups
        self.liquid_head = nn.Linear(classifier_hidden, num_liquids)
        if strict_ordinal:
            self.concentration_score_head = nn.Linear(classifier_hidden, 1)
            self.concentration_cutpoint_start = nn.Parameter(torch.tensor(-0.5))
            if num_concentration_groups > 2:
                self.concentration_cutpoint_deltas = nn.Parameter(
                    torch.full((num_concentration_groups - 2,), 0.54132485)
                )
            else:
                self.register_parameter("concentration_cutpoint_deltas", None)
        else:
            self.concentration_head = nn.Linear(classifier_hidden, num_concentration_groups - 1)

    def _ordered_concentration_cutpoints(self) -> torch.Tensor:
        if self.concentration_cutpoint_deltas is None:
            return self.concentration_cutpoint_start.reshape(1)
        positive_deltas = F.softplus(self.concentration_cutpoint_deltas) + 1e-6
        cutpoints = self.concentration_cutpoint_start + torch.cumsum(positive_deltas, dim=0)
        return torch.cat((self.concentration_cutpoint_start.reshape(1), cutpoints))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.embedding(x)
        liquid_logits = self.liquid_head(x)
        if self.strict_ordinal:
            concentration_score = self.concentration_score_head(x)
            cutpoints = self._ordered_concentration_cutpoints()
            concentration_logits = concentration_score - cutpoints.reshape(1, -1)
        else:
            concentration_logits = self.concentration_head(x)
        return liquid_logits, concentration_logits
