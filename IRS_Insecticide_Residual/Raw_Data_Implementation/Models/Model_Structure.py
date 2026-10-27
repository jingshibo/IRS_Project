import torch
import torch.nn as nn

# =========================
# Centralized Model Configs
# =========================
# Keep model defaults centralized here.
one_d_kernel_sizes = (5, 5, 5, 5)
one_d_feature_extractor_channels = (32, 64, 128, 256)
one_d_in_channels = 2
one_d_num_classes = 3
one_d_dropout = 0.1
one_d_leaky_relu_slope = 0.05
one_d_classifier_hidden = (512, 256)
one_d_strides = (2, 2, 2, 2)
one_d_dilations = (1, 1, 1, 1)
one_d_use_pool = (True, True, True, True)
one_d_pool_types = ("max", "max", "max", "max")
one_d_pool_kernel_sizes = (3, 3, 3, 3)
one_d_pool_strides = (2, 2, 2, 2)
one_d_pool_paddings = (0, 0, 0, 0)
one_d_pool_kernel_size = 3
one_d_pool_stride = 2
one_d_pool_padding = 0

multi_scale_num_classes = 3
multi_scale_in_channels = 2
multi_scale_branch_kernel_sizes = (11, 31, 51)
multi_scale_branch_out_channels = 16
multi_scale_kernel_sizes = (5, 5, 5)
multi_scale_dropout = 0.3
multi_scale_leaky_relu_slope = 0.05
multi_scale_classifier_hidden = (512, 256)
multi_scale_strides = (2, 2, 2, 2)
multi_scale_dilations = (1, 1, 1, 1)
multi_scale_use_pool = (True, True, True, True)
multi_scale_pool_types = ("max", "max", "max", "max")
multi_scale_pool_kernel_sizes = (3, 3, 3, 3)
multi_scale_pool_strides = (2, 2, 2, 2)
multi_scale_pool_paddings = (0, 0, 0, 0)
multi_scale_feature_channels = (64, 128, 256)

dual_branch_num_classes = 3
dual_branch_dropout = 0.3
dual_branch_kernel_sizes = (5, 5, 5, 5)
dual_branch_leaky_relu_slope = 0.01
dual_branch_classifier_hidden = (1024, 256)

fusion_num_classes = 3
fusion_dropout = 0.3
fusion_branch_kernel_sizes = (5, 5, 5)
fusion_kernel_sizes = (5,)
fusion_branch_channels = (32, 64, 128)
fusion_leaky_relu_slope = 0.01

tcn_channels = (64, 128, 128, 256)
tcn_kernel_size = 5
tcn_dropout = 0.2
tcn_downsample_kernel_size = 3
tcn_downsample_stride = 2
tcn_pool_kernel_size = 2
tcn_pool_stride = 2
tcn_pool_type = "max"


def _validate_kernel_sizes(name: str, kernel_sizes, expected_len: int) -> None:
    if len(kernel_sizes) != expected_len:
        raise ValueError(f"{name} expects {expected_len} kernel sizes, got {len(kernel_sizes)}")


def _validate_block_values(name: str, values, expected_len: int) -> None:
    if len(values) != expected_len:
        raise ValueError(f"{name} expects {expected_len} values, got {len(values)}")


def _same_padding(kernel_size: int, stride: int, dilation: int) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 5,
    stride: int = 2,
    dilation: int = 1,
    leaky_relu_slope: float = one_d_leaky_relu_slope,
    pool_kernel_size: int = one_d_pool_kernel_size,
    pool_stride: int = one_d_pool_stride,
    pool_padding: int = one_d_pool_padding,
    pool_type: str = "max",
    use_pool: bool = True,
) -> nn.Sequential:
    """Conv-BN-ReLU-Pool block used across model variants."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    pad = _same_padding(kernel_size=kernel_size, stride=stride, dilation=dilation)
    layers = [
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
    ]
    if use_pool:
        if pool_type == "max":
            layers.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding))
        elif pool_type == "avg":
            layers.append(nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding))
        else:
            raise ValueError(f"Unsupported pool_type='{pool_type}'. Use 'max' or 'avg'.")
    return nn.Sequential(*layers)


class CNNFeatureExtractor(nn.Module):
    """Shared CNN feature extractor block used by single/dual-branch models."""

    def __init__(
        self,
        in_channels: int,
        kernel_sizes=one_d_kernel_sizes,
        strides=one_d_strides,
        dilations=one_d_dilations,
        use_pool=one_d_use_pool,
        pool_types=one_d_pool_types,
        pool_kernel_sizes=one_d_pool_kernel_sizes,
        pool_strides=one_d_pool_strides,
        pool_paddings=one_d_pool_paddings,
        leaky_relu_slope: float = one_d_leaky_relu_slope,
    ):
        super().__init__()
        _validate_kernel_sizes("CNNFeatureExtractor", kernel_sizes, expected_len=4)
        _validate_block_values("CNNFeatureExtractor strides", strides, expected_len=4)
        _validate_block_values("CNNFeatureExtractor dilations", dilations, expected_len=4)
        _validate_block_values("CNNFeatureExtractor use_pool", use_pool, expected_len=4)
        _validate_block_values("CNNFeatureExtractor pool_types", pool_types, expected_len=4)
        _validate_block_values("CNNFeatureExtractor pool_kernel_sizes", pool_kernel_sizes, expected_len=4)
        _validate_block_values("CNNFeatureExtractor pool_strides", pool_strides, expected_len=4)
        _validate_block_values("CNNFeatureExtractor pool_paddings", pool_paddings, expected_len=4)
        c1, c2, c3, c4 = one_d_feature_extractor_channels
        self.features = nn.Sequential(
            _conv_block(
                in_channels,
                c1,
                kernel_sizes[0],
                stride=strides[0],
                dilation=dilations[0],
                leaky_relu_slope=leaky_relu_slope,
                pool_kernel_size=pool_kernel_sizes[0],
                pool_stride=pool_strides[0],
                pool_padding=pool_paddings[0],
                pool_type=pool_types[0],
                use_pool=use_pool[0],
            ),
            _conv_block(
                c1,
                c2,
                kernel_sizes[1],
                stride=strides[1],
                dilation=dilations[1],
                leaky_relu_slope=leaky_relu_slope,
                pool_kernel_size=pool_kernel_sizes[1],
                pool_stride=pool_strides[1],
                pool_padding=pool_paddings[1],
                pool_type=pool_types[1],
                use_pool=use_pool[1],
            ),
            _conv_block(
                c2,
                c3,
                kernel_sizes[2],
                stride=strides[2],
                dilation=dilations[2],
                leaky_relu_slope=leaky_relu_slope,
                pool_kernel_size=pool_kernel_sizes[2],
                pool_stride=pool_strides[2],
                pool_padding=pool_paddings[2],
                pool_type=pool_types[2],
                use_pool=use_pool[2],
            ),
            _conv_block(
                c3,
                c4,
                kernel_sizes[3],
                stride=strides[3],
                dilation=dilations[3],
                leaky_relu_slope=leaky_relu_slope,
                pool_kernel_size=pool_kernel_sizes[3],
                pool_stride=pool_strides[3],
                pool_padding=pool_paddings[3],
                pool_type=pool_types[3],
                use_pool=use_pool[3],
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _MultiScaleStem(nn.Module):
    """Parallel first-layer convolutions fused by channel concatenation."""

    def __init__(
        self,
        in_channels: int,
        branch_out_channels: int,
        branch_kernel_sizes=multi_scale_branch_kernel_sizes,
        stride: int = 2,
        dilation: int = 1,
        use_pool: bool = True,
        pool_type: str = "max",
        pool_kernel_size: int = one_d_pool_kernel_size,
        pool_stride: int = one_d_pool_stride,
        pool_padding: int = one_d_pool_padding,
        leaky_relu_slope: float = one_d_leaky_relu_slope,
    ):
        super().__init__()
        if len(branch_kernel_sizes) == 0:
            raise ValueError("branch_kernel_sizes must contain at least one kernel size.")
        if branch_out_channels < 1:
            raise ValueError(f"branch_out_channels must be >= 1, got {branch_out_channels}")
        branches = []
        for kernel_size in branch_kernel_sizes:
            branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        branch_out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding=_same_padding(kernel_size=kernel_size, stride=stride, dilation=dilation),
                    ),
                    nn.BatchNorm1d(branch_out_channels),
                    nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.use_pool = use_pool
        if use_pool:
            if pool_type == "max":
                self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
            elif pool_type == "avg":
                self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
            else:
                raise ValueError(f"Unsupported pool_type='{pool_type}'. Use 'max' or 'avg'.")
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.pool(x)


class MultiScaleCNNFeatureExtractor(nn.Module):
    """CNN feature extractor with a multi-scale first layer."""

    def __init__(
        self,
        in_channels: int,
        branch_kernel_sizes=multi_scale_branch_kernel_sizes,
        branch_out_channels: int = multi_scale_branch_out_channels,
        kernel_sizes=multi_scale_kernel_sizes,
        strides=multi_scale_strides,
        dilations=multi_scale_dilations,
        use_pool=multi_scale_use_pool,
        pool_types=multi_scale_pool_types,
        pool_kernel_sizes=multi_scale_pool_kernel_sizes,
        pool_strides=multi_scale_pool_strides,
        pool_paddings=multi_scale_pool_paddings,
        leaky_relu_slope: float = multi_scale_leaky_relu_slope,
    ):
        super().__init__()
        _validate_block_values("MultiScaleCNNFeatureExtractor strides", strides, expected_len=4)
        _validate_block_values("MultiScaleCNNFeatureExtractor dilations", dilations, expected_len=4)
        _validate_block_values("MultiScaleCNNFeatureExtractor use_pool", use_pool, expected_len=4)
        _validate_block_values("MultiScaleCNNFeatureExtractor pool_types", pool_types, expected_len=4)
        _validate_block_values("MultiScaleCNNFeatureExtractor pool_kernel_sizes", pool_kernel_sizes, expected_len=4)
        _validate_block_values("MultiScaleCNNFeatureExtractor pool_strides", pool_strides, expected_len=4)
        _validate_block_values("MultiScaleCNNFeatureExtractor pool_paddings", pool_paddings, expected_len=4)
        _validate_kernel_sizes("MultiScaleCNNFeatureExtractor", kernel_sizes, expected_len=3)
        c2, c3, c4 = multi_scale_feature_channels
        stem_out_channels = branch_out_channels * len(branch_kernel_sizes)
        self.features = nn.Sequential(
            _MultiScaleStem(
                in_channels=in_channels,
                branch_out_channels=branch_out_channels,
                branch_kernel_sizes=branch_kernel_sizes,
                stride=strides[0],
                dilation=dilations[0],
                use_pool=use_pool[0],
                pool_type=pool_types[0],
                pool_kernel_size=pool_kernel_sizes[0],
                pool_stride=pool_strides[0],
                pool_padding=pool_paddings[0],
                leaky_relu_slope=leaky_relu_slope,
            ),
            _conv_block(
                stem_out_channels,
                c2,
                kernel_sizes[0],
                stride=strides[1],
                dilation=dilations[1],
                leaky_relu_slope=leaky_relu_slope,
                pool_kernel_size=pool_kernel_sizes[1],
                pool_stride=pool_strides[1],
                pool_padding=pool_paddings[1],
                pool_type=pool_types[1],
                use_pool=use_pool[1],
            ),
            _conv_block(
                c2,
                c3,
                kernel_sizes[1],
                stride=strides[2],
                dilation=dilations[2],
                leaky_relu_slope=leaky_relu_slope,
                pool_kernel_size=pool_kernel_sizes[2],
                pool_stride=pool_strides[2],
                pool_padding=pool_paddings[2],
                pool_type=pool_types[2],
                use_pool=use_pool[2],
            ),
            _conv_block(
                c3,
                c4,
                kernel_sizes[2],
                stride=strides[3],
                dilation=dilations[3],
                leaky_relu_slope=leaky_relu_slope,
                pool_kernel_size=pool_kernel_sizes[3],
                pool_stride=pool_strides[3],
                pool_padding=pool_paddings[3],
                pool_type=pool_types[3],
                use_pool=use_pool[3],
            ),
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
        strides=one_d_strides,
        dilations=one_d_dilations,
        use_pool=one_d_use_pool,
        pool_types=one_d_pool_types,
        pool_kernel_sizes=one_d_pool_kernel_sizes,
        pool_strides=one_d_pool_strides,
        pool_paddings=one_d_pool_paddings,
        leaky_relu_slope: float = one_d_leaky_relu_slope,
    ):
        super().__init__()
        _validate_kernel_sizes("OneDCNNClassifier", kernel_sizes, expected_len=4)
        hidden1, hidden2 = one_d_classifier_hidden
        self.features = CNNFeatureExtractor(
            in_channels=in_channels,
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
            nn.LazyLinear(num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class MultiScaleOneDCNNClassifier(nn.Module):
    """1D CNN with a multi-scale first layer followed by a shared CNN trunk."""

    def __init__(
        self,
        in_channels: int = multi_scale_in_channels,
        num_classes: int = multi_scale_num_classes,
        dropout: float = multi_scale_dropout,
        branch_kernel_sizes=multi_scale_branch_kernel_sizes,
        branch_out_channels: int = multi_scale_branch_out_channels,
        kernel_sizes=multi_scale_kernel_sizes,
        strides=multi_scale_strides,
        dilations=multi_scale_dilations,
        use_pool=multi_scale_use_pool,
        pool_types=multi_scale_pool_types,
        pool_kernel_sizes=multi_scale_pool_kernel_sizes,
        pool_strides=multi_scale_pool_strides,
        pool_paddings=multi_scale_pool_paddings,
        leaky_relu_slope: float = multi_scale_leaky_relu_slope,
    ):
        super().__init__()
        hidden1, hidden2 = multi_scale_classifier_hidden
        self.features = MultiScaleCNNFeatureExtractor(
            in_channels=in_channels,
            branch_kernel_sizes=branch_kernel_sizes,
            branch_out_channels=branch_out_channels,
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
            nn.LazyLinear(num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class DualBranchOneDCNNClassifier(nn.Module):
    """Dual-branch CNN: raw and diff signals use separate feature extractors."""

    def __init__(
        self,
        num_classes: int = dual_branch_num_classes,
        dropout: float = dual_branch_dropout,
        kernel_sizes=dual_branch_kernel_sizes,
        leaky_relu_slope: float = dual_branch_leaky_relu_slope,
    ):
        super().__init__()
        _validate_kernel_sizes("DualBranchOneDCNNClassifier", kernel_sizes, expected_len=4)
        hidden1, hidden2 = dual_branch_classifier_hidden
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
            nn.LazyLinear(hidden2),
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
        num_classes: int = fusion_num_classes,
        dropout: float = fusion_dropout,
        branch_kernel_sizes=fusion_branch_kernel_sizes,
        fusion_kernel_sizes=fusion_kernel_sizes,
        leaky_relu_slope: float = fusion_leaky_relu_slope,
    ):
        super().__init__()
        _validate_kernel_sizes("DualBranchFusionCNNClassifier (branch)", branch_kernel_sizes, expected_len=3)
        _validate_kernel_sizes("DualBranchFusionCNNClassifier (fusion)", fusion_kernel_sizes, expected_len=1)
        b1, b2, b3 = fusion_branch_channels
        hidden1, hidden2 = dual_branch_classifier_hidden
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


class _Chomp1d(nn.Module):
    """Remove right-side padding to keep causal sequence length unchanged."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class _TemporalBlock(nn.Module):
    """Residual block used by TCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        downsample_kernel_size: int = tcn_downsample_kernel_size,
        downsample_stride: int = tcn_downsample_stride,
        pool_kernel_size: int = tcn_pool_kernel_size,
        pool_stride: int = tcn_pool_stride,
        pool_type: str = tcn_pool_type,
        leaky_relu_slope: float = one_d_leaky_relu_slope,
    ):
        super().__init__()
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2 for TCN blocks, got {kernel_size}")
        if downsample_kernel_size < 1 or downsample_stride < 1 or pool_kernel_size < 1 or pool_stride < 1:
            raise ValueError(
                "downsample/pool kernel_size and stride must be >= 1, "
                f"got downsample=({downsample_kernel_size}, {downsample_stride}), "
                f"pool=({pool_kernel_size}, {pool_stride})"
            )
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            nn.Dropout(dropout),
        )
        self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.out_act = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
        self.downsample_conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=downsample_kernel_size,
            stride=downsample_stride,
            padding=(downsample_kernel_size - 1) // 2,
        )
        if pool_type == "max":
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        elif pool_type == "avg":
            self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        else:
            raise ValueError(f"Unsupported pool_type='{pool_type}'. Use 'max' or 'avg'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out_act(self.net(x) + self.residual_proj(x))
        x = self.downsample_conv(x)
        return self.pool(x)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network classifier for [N, C, L] signals."""

    def __init__(
        self,
        in_channels: int = one_d_in_channels,
        num_classes: int = one_d_num_classes,
        channels=tcn_channels,
        kernel_size: int = tcn_kernel_size,
        dropout: float = tcn_dropout,
        downsample_kernel_size: int = tcn_downsample_kernel_size,
        downsample_stride: int = tcn_downsample_stride,
        pool_kernel_size: int = tcn_pool_kernel_size,
        pool_stride: int = tcn_pool_stride,
        pool_type: str = tcn_pool_type,
        leaky_relu_slope: float = one_d_leaky_relu_slope,
    ):
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must contain at least one stage for TCNClassifier.")
        hidden1, hidden2 = one_d_classifier_hidden
        layers = []
        prev = in_channels
        for stage_idx, out_ch in enumerate(channels):
            dilation = 2 ** stage_idx
            layers.append(
                _TemporalBlock(
                    in_channels=prev,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    downsample_kernel_size=downsample_kernel_size,
                    downsample_stride=downsample_stride,
                    pool_kernel_size=pool_kernel_size,
                    pool_stride=pool_stride,
                    pool_type=pool_type,
                    leaky_relu_slope=leaky_relu_slope,
                )
            )
            prev = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
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
        x = self.tcn(x)
        return self.head(x)
