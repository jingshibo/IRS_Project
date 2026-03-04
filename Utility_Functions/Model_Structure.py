import torch
import torch.nn as nn

# =========================
# Centralized Model Configs
# =========================
# Keep only OneDCNNClassifier-related defaults here.
one_d_kernel_sizes = (5, 5, 5, 5)
one_d_feature_extractor_channels = (32, 64, 128, 256)
one_d_in_channels = 2
one_d_num_classes = 3
one_d_dropout = 0
one_d_leaky_relu_slope = 0.3
one_d_classifier_hidden = (512, 256)
one_d_pool_kernel_size = 3
one_d_pool_stride = 2
one_d_pool_padding = 0


def _validate_kernel_sizes(name: str, kernel_sizes, expected_len: int) -> None:
    if len(kernel_sizes) != expected_len:
        raise ValueError(f"{name} expects {expected_len} kernel sizes, got {len(kernel_sizes)}")


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 5,
    leaky_relu_slope: float = one_d_leaky_relu_slope,
    pool_kernel_size: int = one_d_pool_kernel_size,
    pool_stride: int = one_d_pool_stride,
    pool_padding: int = one_d_pool_padding,
) -> nn.Sequential:
    """Conv-BN-ReLU-Pool block used across model variants."""
    stride = 2
    dilation = 1
    pad = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad,
        ),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
        nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
    )


class CNNFeatureExtractor(nn.Module):
    """Shared CNN feature extractor block used by single/dual-branch models."""

    def __init__(
        self,
        in_channels: int,
        kernel_sizes=one_d_kernel_sizes,
        leaky_relu_slope: float = one_d_leaky_relu_slope,
    ):
        super().__init__()
        _validate_kernel_sizes("CNNFeatureExtractor", kernel_sizes, expected_len=4)
        c1, c2, c3, c4 = one_d_feature_extractor_channels
        self.features = nn.Sequential(
            _conv_block(in_channels, c1, kernel_sizes[0], leaky_relu_slope=leaky_relu_slope),
            _conv_block(c1, c2, kernel_sizes[1], leaky_relu_slope=leaky_relu_slope),
            _conv_block(c2, c3, kernel_sizes[2], leaky_relu_slope=leaky_relu_slope),
            _conv_block(c3, c4, kernel_sizes[3], leaky_relu_slope=leaky_relu_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class OneDCNNClassifier(nn.Module):
    """1D CNN model for multi-class classification."""

    def __init__(
        self,
        in_channels: int = one_d_in_channels,
        num_classes: int = one_d_num_classes,
        dropout: float = one_d_dropout,
        kernel_sizes=one_d_kernel_sizes,
        leaky_relu_slope: float = one_d_leaky_relu_slope,
    ):
        super().__init__()
        _validate_kernel_sizes("OneDCNNClassifier", kernel_sizes, expected_len=4)
        hidden1, hidden2 = one_d_classifier_hidden
        self.features = CNNFeatureExtractor(
            in_channels=in_channels,
            kernel_sizes=kernel_sizes,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.Dropout(p=dropout),
            nn.LazyLinear(hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class DualBranchOneDCNNClassifier(nn.Module):
    """Dual-branch CNN: raw and diff signals use separate feature extractors."""

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.3,
        kernel_sizes=(5, 5, 5, 5),
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        _validate_kernel_sizes("DualBranchOneDCNNClassifier", kernel_sizes, expected_len=4)
        hidden1, hidden2 = (1024, 256)
        self.raw_branch = CNNFeatureExtractor(
            in_channels=1,
            kernel_sizes=kernel_sizes,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.diff_branch = CNNFeatureExtractor(
            in_channels=1,
            kernel_sizes=kernel_sizes,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, 2, L], split into raw and diff branches.
        if x.size(1) != 2:
            raise ValueError(f"DualBranchOneDCNNClassifier expects 2 channels [raw, diff], got {x.size(1)}")

        x_raw = x[:, 0:1, :]
        x_diff = x[:, 1:2, :]

        feat_raw = self.raw_branch(x_raw)
        feat_diff = self.diff_branch(x_diff)
        fused = torch.cat([feat_raw, feat_diff], dim=1)
        logits = self.classifier(fused)
        return logits


class DualBranchFusionCNNClassifier(nn.Module):
    """Two-branch early feature extraction, then fused shared CNN trunk."""

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.3,
        branch_kernel_sizes=(5, 5, 5),
        fusion_kernel_sizes=(5,),
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        _validate_kernel_sizes("DualBranchFusionCNNClassifier (branch)", branch_kernel_sizes, expected_len=3)
        _validate_kernel_sizes("DualBranchFusionCNNClassifier (fusion)", fusion_kernel_sizes, expected_len=1)
        b1, b2, b3 = (32, 64, 128)
        hidden1, hidden2 = (1024, 256)
        fusion_channels_in = b3 * 2
        # First 3 conv blocks per branch.
        self.raw_branch = nn.Sequential(
            _conv_block(1, b1, branch_kernel_sizes[0], leaky_relu_slope=leaky_relu_slope),
            _conv_block(b1, b2, branch_kernel_sizes[1], leaky_relu_slope=leaky_relu_slope),
            _conv_block(b2, b3, branch_kernel_sizes[2], leaky_relu_slope=leaky_relu_slope),
        )
        self.diff_branch = nn.Sequential(
            _conv_block(1, b1, branch_kernel_sizes[0], leaky_relu_slope=leaky_relu_slope),
            _conv_block(b1, b2, branch_kernel_sizes[1], leaky_relu_slope=leaky_relu_slope),
            _conv_block(b2, b3, branch_kernel_sizes[2], leaky_relu_slope=leaky_relu_slope),
        )
        # After concatenation: channels 128 + 128 = 256.
        self.shared_fusion_trunk = nn.Sequential(
            _conv_block(fusion_channels_in, fusion_channels_in, fusion_kernel_sizes[0], leaky_relu_slope=leaky_relu_slope),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.Dropout(p=dropout),
            nn.LazyLinear(hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, 2, L], with channel 0 = raw and channel 1 = diff.
        if x.size(1) != 2:
            raise ValueError(f"DualBranchFusionCNNClassifier expects 2 channels [raw, diff], got {x.size(1)}")

        x_raw = x[:, 0:1, :]
        x_diff = x[:, 1:2, :]

        feat_raw = self.raw_branch(x_raw)
        feat_diff = self.diff_branch(x_diff)
        fused = torch.cat([feat_raw, feat_diff], dim=1)  # concatenate feature maps as channels
        fused = self.shared_fusion_trunk(fused)
        logits = self.classifier(fused)
        return logits
