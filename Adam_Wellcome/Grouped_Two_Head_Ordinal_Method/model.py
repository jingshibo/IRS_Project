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
    dilation: int,
    dropout: float,
    negative_slope: float,
) -> nn.Sequential:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Dropout(dropout),
    )


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class ResidualSEConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
        negative_slope: float,
        se_reduction: int,
    ):
        super().__init__()
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.se = SqueezeExcitation1D(out_channels, reduction=se_reduction)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = self.norm(self.conv(x))
        x = self.activation(x + residual)
        x = self.se(x)
        x = self.pool(x)
        return self.dropout(x)


class GroupedOrdinalAdamWellcomeCNN1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        input_length: int,
        num_liquids: int,
        num_concentration_groups: int,
        num_joint_classes: int | None = None,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_sizes: tuple[int, ...] = (5, 5, 3, 3),
        strides: tuple[int, ...] = (2, 2, 1, 1),
        dilations: tuple[int, ...] = (2, 2, 1, 1),
        dropout: float = 0.1,
        negative_slope: float = 0.05,
        classifier_hidden: int = 128,
        feature_block_type: str = "plain",
        se_reduction: int = 8,
        strict_ordinal: bool = False,
    ):
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be >= 1")
        if num_liquids < 2:
            raise ValueError("num_liquids must be >= 2")
        if num_concentration_groups < 2:
            raise ValueError("num_concentration_groups must be >= 2")
        if num_joint_classes is not None and num_joint_classes < 2:
            raise ValueError("num_joint_classes must be >= 2 when provided")
        if not (len(channels) == len(kernel_sizes) == len(strides) == len(dilations)):
            raise ValueError("channels, kernel_sizes, strides, and dilations must have the same length")
        if feature_block_type not in {"plain", "residual_se"}:
            raise ValueError("feature_block_type must be 'plain' or 'residual_se'")
        if se_reduction <= 0:
            raise ValueError("se_reduction must be >= 1")

        blocks = []
        current_in = in_channels
        for out_channels, kernel_size, stride, dilation in zip(channels, kernel_sizes, strides, dilations):
            if feature_block_type == "plain":
                block = _conv_block(
                    current_in,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            else:
                block = ResidualSEConvBlock(
                    current_in,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dropout=dropout,
                    negative_slope=negative_slope,
                    se_reduction=se_reduction,
                )
            blocks.append(block)
            current_in = out_channels

        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            feature_example = torch.zeros(1, in_channels, input_length)
            flattened_dim = int(self.features(feature_example).flatten(start_dim=1).shape[1])
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, classifier_hidden),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout(dropout),
        )
        self.strict_ordinal = strict_ordinal
        self.num_concentration_groups = num_concentration_groups
        self.liquid_head = nn.Linear(classifier_hidden, num_liquids)
        self.joint_head = nn.Linear(classifier_hidden, num_joint_classes) if num_joint_classes is not None else None
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_joint_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.embedding(x)
        liquid_logits = self.liquid_head(x)
        if self.strict_ordinal:
            concentration_score = self.concentration_score_head(x)
            cutpoints = self._ordered_concentration_cutpoints()
            concentration_logits = concentration_score - cutpoints.reshape(1, -1)
        else:
            concentration_logits = self.concentration_head(x)
        if return_joint_logits:
            if self.joint_head is None:
                raise RuntimeError("joint_head is not configured")
            return liquid_logits, concentration_logits, self.joint_head(x)
        return liquid_logits, concentration_logits
