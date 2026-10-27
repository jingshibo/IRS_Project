import copy
from collections import Counter
import gc
import itertools
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class _SignalDataset(Dataset):
    """Dataset wrapper for normalized CV fold arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _conv_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    """Conv-BN-Activation-Pool block (legacy wrapper defaults)."""
    return _conv_block_cfg(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        conv_stride=2,
        conv_dilation=1,
        avgpool_kernel_size=2,
        avgpool_stride=2,
        pool_type="avg",
        leaky_relu_alpha=0.01,
    )


def _conv_block_cfg(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    conv_stride: int,
    conv_dilation: int,
    avgpool_kernel_size: int,
    avgpool_stride: int,
    pool_type: str,
    leaky_relu_alpha: float = 0.01,
) -> nn.Sequential:
    """Conv-BN-LeakyReLU-Pool block used by all three models."""
    if conv_stride < 1 or conv_dilation < 1 or avgpool_kernel_size < 1 or avgpool_stride < 1:
        raise ValueError("conv/pool stride, dilation and kernel sizes must be >= 1")
    if leaky_relu_alpha < 0:
        raise ValueError("leaky_relu_alpha must be >= 0")
    # General approximate "same" padding formula for Conv1d.
    pad = ((conv_stride - 1) + conv_dilation * (kernel_size - 1)) // 2
    act = nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True)
    if pool_type == "avg":
        pool = nn.AvgPool1d(kernel_size=avgpool_kernel_size, stride=avgpool_stride)
    elif pool_type == "max":
        pool = nn.MaxPool1d(kernel_size=avgpool_kernel_size, stride=avgpool_stride)
    else:
        raise ValueError(f"Unsupported pool_type='{pool_type}'. Use 'avg' or 'max'.")
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=pad,
            dilation=conv_dilation,
        ),
        nn.BatchNorm1d(out_channels),
        act,
        pool,
    )


class _SharedBackbone2Ch(nn.Module):
    """Single shared CNN over all input channels."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        c0: int,
        k: int,
        l1: int,
        l2: int,
        dropout: float,
        conv_stride: int,
        conv_dilation: int,
        avgpool_kernel_size: int,
        avgpool_stride: int,
        pool_type: str,
        leaky_relu_alpha: float,
    ):
        super().__init__()
        c1, c2, c3 = c0 * 2, c0 * 4, c0 * 8
        act1 = nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True)
        act2 = nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True)
        self.features = nn.Sequential(
            _conv_block_cfg(in_channels, c0, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c0, c1, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c1, c2, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c2, c3, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(l1),
            nn.BatchNorm1d(l1),
            act1,
            nn.Dropout(dropout),
            nn.Linear(l1, l2),
            nn.BatchNorm1d(l2),
            act2,
            nn.Dropout(dropout),
            nn.Linear(l2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class _TwoTowerLateFusion(nn.Module):
    """Two separate towers, fuse right before MLP."""

    def __init__(
        self,
        num_classes: int,
        c0: int,
        k: int,
        l1: int,
        l2: int,
        dropout: float,
        conv_stride: int,
        conv_dilation: int,
        avgpool_kernel_size: int,
        avgpool_stride: int,
        pool_type: str,
        leaky_relu_alpha: float,
    ):
        super().__init__()
        c1, c2, c3 = c0 * 2, c0 * 4, c0 * 8
        act1 = nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True)
        act2 = nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True)
        self.raw_tower = nn.Sequential(
            _conv_block_cfg(1, c0, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c0, c1, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c1, c2, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c2, c3, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
        )
        self.diff_tower = nn.Sequential(
            _conv_block_cfg(1, c0, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c0, c1, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c1, c2, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c2, c3, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(l1),
            nn.BatchNorm1d(l1),
            act1,
            nn.Dropout(dropout),
            nn.Linear(l1, l2),
            nn.BatchNorm1d(l2),
            act2,
            nn.Dropout(dropout),
            nn.Linear(l2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 2:
            raise ValueError(f"two_tower_late_fusion expects x shape [N, 2, L], got channel count {x.size(1)}")
        xr = self.raw_tower(x[:, 0:1, :])
        xd = self.diff_tower(x[:, 1:2, :])
        return self.classifier(torch.cat([xr, xd], dim=1))


class _TwoTowerMidFusionCNN(nn.Module):
    """Two towers for first 3 blocks, concatenate, then 2 shared CNN blocks."""

    def __init__(
        self,
        num_classes: int,
        c0: int,
        k: int,
        l1: int,
        l2: int,
        dropout: float,
        conv_stride: int,
        conv_dilation: int,
        avgpool_kernel_size: int,
        avgpool_stride: int,
        pool_type: str,
        leaky_relu_alpha: float,
    ):
        super().__init__()
        c1, c2, c3 = c0 * 2, c0 * 4, c0 * 8
        act1 = nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True)
        act2 = nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True)
        self.raw_tower = nn.Sequential(
            _conv_block_cfg(1, c0, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c0, c1, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c1, c2, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
        )
        self.diff_tower = nn.Sequential(
            _conv_block_cfg(1, c0, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c0, c1, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c1, c2, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
        )
        # Fused channels = c2 + c2
        self.shared_trunk = nn.Sequential(
            _conv_block_cfg(c2 * 2, c3, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
            _conv_block_cfg(c3, c3, k, conv_stride, conv_dilation, avgpool_kernel_size, avgpool_stride, pool_type, leaky_relu_alpha),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(l1),
            nn.BatchNorm1d(l1),
            act1,
            nn.Dropout(dropout),
            nn.Linear(l1, l2),
            nn.BatchNorm1d(l2),
            act2,
            nn.Dropout(dropout),
            nn.Linear(l2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 2:
            raise ValueError(f"two_tower_mid_fusion_cnn expects x shape [N, 2, L], got channel count {x.size(1)}")
        xr = self.raw_tower(x[:, 0:1, :])
        xd = self.diff_tower(x[:, 1:2, :])
        fused = torch.cat([xr, xd], dim=1)
        return self.classifier(self.shared_trunk(fused))


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
    """Residual TCN block with conv downsample and pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        downsample_kernel_size: int,
        downsample_stride: int,
        pool_kernel_size: int,
        pool_stride: int,
        pool_type: str,
        leaky_relu_alpha: float,
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
            nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
            nn.Dropout(dropout),
        )
        self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.out_act = nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True)
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


class _TCNClassifier(nn.Module):
    """TCN classifier that mirrors Model_Structure.TCNClassifier."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        c0: int,
        k: int,
        l1: int,
        l2: int,
        dropout: float,
        conv_stride: int,
        avgpool_kernel_size: int,
        avgpool_stride: int,
        pool_type: str,
        leaky_relu_alpha: float,
    ):
        super().__init__()
        channels = (c0, c0 * 2, c0 * 2, c0 * 4)
        layers = []
        prev = in_channels
        for stage_idx, out_ch in enumerate(channels):
            dilation = 2 ** stage_idx
            layers.append(
                _TemporalBlock(
                    in_channels=prev,
                    out_channels=out_ch,
                    kernel_size=k,
                    dilation=dilation,
                    dropout=dropout,
                    downsample_kernel_size=3,
                    downsample_stride=conv_stride,
                    pool_kernel_size=avgpool_kernel_size,
                    pool_stride=avgpool_stride,
                    pool_type=pool_type,
                    leaky_relu_alpha=leaky_relu_alpha,
                )
            )
            prev = out_ch
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(l1),
            nn.BatchNorm1d(l1),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
            nn.Dropout(dropout),
            nn.LazyLinear(l2),
            nn.BatchNorm1d(l2),
            nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(l2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.tcn(x))


def _build_model(model_name: str, in_channels: int, num_classes: int, params: Dict[str, Any]) -> nn.Module:
    c0 = int(params["base_channels"])
    k = int(params["kernel_size"])
    l1 = int(params["l1"])
    l2 = int(params["l2"])
    p = float(params["dropout"])
    conv_stride = int(params["conv_stride"])
    conv_dilation = int(params["conv_dilation"])
    avgpool_kernel_size = int(params["avgpool_kernel_size"])
    avgpool_stride = int(params["avgpool_stride"])
    pool_type = str(params["pool_type"])
    leaky_relu_alpha = float(params["leaky_relu_alpha"])

    if model_name == "shared_backbone_2ch":
        return _SharedBackbone2Ch(
            in_channels=in_channels,
            num_classes=num_classes,
            c0=c0,
            k=k,
            l1=l1,
            l2=l2,
            dropout=p,
            conv_stride=conv_stride,
            conv_dilation=conv_dilation,
            avgpool_kernel_size=avgpool_kernel_size,
            avgpool_stride=avgpool_stride,
            pool_type=pool_type,
            leaky_relu_alpha=leaky_relu_alpha,
        )
    if model_name == "two_tower_late_fusion":
        return _TwoTowerLateFusion(
            num_classes=num_classes,
            c0=c0,
            k=k,
            l1=l1,
            l2=l2,
            dropout=p,
            conv_stride=conv_stride,
            conv_dilation=conv_dilation,
            avgpool_kernel_size=avgpool_kernel_size,
            avgpool_stride=avgpool_stride,
            pool_type=pool_type,
            leaky_relu_alpha=leaky_relu_alpha,
        )
    if model_name == "two_tower_mid_fusion_cnn":
        return _TwoTowerMidFusionCNN(
            num_classes=num_classes,
            c0=c0,
            k=k,
            l1=l1,
            l2=l2,
            dropout=p,
            conv_stride=conv_stride,
            conv_dilation=conv_dilation,
            avgpool_kernel_size=avgpool_kernel_size,
            avgpool_stride=avgpool_stride,
            pool_type=pool_type,
            leaky_relu_alpha=leaky_relu_alpha,
        )
    if model_name == "tcn_classifier":
        return _TCNClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            c0=c0,
            k=k,
            l1=l1,
            l2=l2,
            dropout=p,
            conv_stride=conv_stride,
            avgpool_kernel_size=avgpool_kernel_size,
            avgpool_stride=avgpool_stride,
            pool_type=pool_type,
            leaky_relu_alpha=leaky_relu_alpha,
        )
    raise ValueError(
        f"Unsupported model_name='{model_name}'. Use 'shared_backbone_2ch', 'two_tower_late_fusion', 'two_tower_mid_fusion_cnn', or 'tcn_classifier'."
    )


def _build_label_mapping(cv_folds, class_order: Optional[Sequence[str]]):
    if class_order is not None:
        return {label: i for i, label in enumerate(class_order)}
    labels = []
    for f in cv_folds:
        labels.extend(f["y_train"].tolist())
        labels.extend(f["y_val"].tolist())
    return {label: i for i, label in enumerate(sorted(set(labels)))}


def _encode_labels(labels: Iterable[str], label_to_idx: Dict[str, int]) -> np.ndarray:
    return np.asarray([label_to_idx[l] for l in labels], dtype=np.int64)


def _run_epoch(model, loader, criterion, device, optimizer=None) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                loss.backward()
                optimizer.step()
        bs = xb.size(0)
        loss_sum += loss.item() * bs
        correct += (torch.argmax(logits, dim=1) == yb).sum().item()
        total += bs

    return loss_sum / max(total, 1), correct / max(total, 1)


@dataclass
class GridTrialResult:
    trial_id: int
    model_name: str
    params: Dict[str, Any]
    mean_best_val_acc: float
    fold_best_val_accs: List[float]
    fold_best_epochs: List[int]
    duration_sec: float
    status: str = "ok"
    error: Optional[str] = None


def default_search_space() -> Dict[str, Sequence[Any]]:
    """Search space from the provided image."""
    return {
        "base_channels": [32, 64],
        "kernel_size": [3, 5, 9, 15],
        "l1": [1024],
        "l2": [256],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "dropout": [0.0, 0.3, 0.5],
        "label_smoothing": [0.0, 0.05, 0.1],
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "batch_size": [64, 128],
        "conv_stride": [2],
        "conv_dilation": [1],
        "avgpool_kernel_size": [2],
        "avgpool_stride": [2],
        "pool_type": ["avg", "max"],
        "leaky_relu_alpha": [0.01, 0.05],
        "scheduler_factor": [0.5],
    }


def estimate_total_trials(search_space: Dict[str, Sequence[Any]], model_names: Sequence[str]) -> int:
    total = 1
    for vals in search_space.values():
        total *= len(vals)
    return total * len(model_names)


def _iter_grid_jobs(
    search_space: Dict[str, Sequence[Any]],
    model_names: Sequence[str],
):
    """Yield (model_name, params) lazily instead of materializing all combinations."""
    param_keys = [
        "base_channels",
        "kernel_size",
        "l1",
        "l2",
        "weight_decay",
        "dropout",
        "label_smoothing",
        "lr",
        "batch_size",
        "conv_stride",
        "conv_dilation",
        "avgpool_kernel_size",
        "avgpool_stride",
        "pool_type",
        "leaky_relu_alpha",
        "scheduler_factor",
    ]
    for model_name in model_names:
        for combo in itertools.product(*(search_space[k] for k in param_keys)):
            yield model_name, dict(zip(param_keys, combo))


def run_grid_search_three_models(
    cv_folds,
    search_space: Optional[Dict[str, Sequence[Any]]] = None,
    model_names: Sequence[str] = ("shared_backbone_2ch", "two_tower_late_fusion", "two_tower_mid_fusion_cnn", "tcn_classifier"),
    class_order: Optional[Sequence[str]] = ("LOW", "TARGET", "HIGH"),
    epochs: int = 80,
    patience: int = 20,
    device: Optional[str] = None,
    num_workers: int = 0,
    verbose_train: bool = False,
    use_lr_scheduler: bool = True,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
    scheduler_min_lr: float = 1e-7,
    tensorboard_log_dir_root: Optional[str] = None,
    tensorboard_write_every_n: int = 10,
    max_trials: Optional[int] = None,
    continue_on_error: bool = True,
    print_progress: bool = True,
) -> Dict[str, Any]:
    """Grid search across model families on existing cv_folds.

    Notes:
    - `cv_folds` must already be normalized.
    - Two-tower models require exactly 2 channels in `cv_folds[*]["X_train"]`.
    - Full image search space is very large; use `max_trials` or a reduced space first.
    """
    if len(cv_folds) == 0:
        raise ValueError("cv_folds is empty.")
    search_space = search_space or default_search_space()

    # Basic validation
    required_keys = {
        "base_channels",
        "kernel_size",
        "l1",
        "l2",
        "weight_decay",
        "dropout",
        "label_smoothing",
        "lr",
        "batch_size",
        "conv_stride",
        "conv_dilation",
        "avgpool_kernel_size",
        "avgpool_stride",
        "pool_type",
        "leaky_relu_alpha",
        "scheduler_factor",
    }
    missing = required_keys.difference(search_space.keys())
    if missing:
        raise ValueError(f"search_space missing required keys: {sorted(missing)}")

    label_to_idx = _build_label_mapping(cv_folds, class_order)
    num_classes = len(label_to_idx)
    device_obj = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    total_jobs = estimate_total_trials(search_space, model_names)
    if max_trials is not None:
        total_jobs = min(total_jobs, int(max_trials))

    trial_results: List[GridTrialResult] = []
    started = time.perf_counter()

    job_iter = _iter_grid_jobs(search_space, model_names)
    if max_trials is not None:
        job_iter = itertools.islice(job_iter, int(max_trials))

    for trial_id, (model_name, params) in enumerate(job_iter, start=1):
        t0 = time.perf_counter()
        if print_progress:
            print(f"[GridSearch] {trial_id}/{total_jobs} model={model_name} params={params}")

        try:
            fold_best_accs: List[float] = []
            fold_best_epochs: List[int] = []

            for fold in cv_folds:
                x_train = np.asarray(fold["X_train"], dtype=np.float32)
                x_val = np.asarray(fold["X_val"], dtype=np.float32)
                y_train = _encode_labels(fold["y_train"], label_to_idx)
                y_val = _encode_labels(fold["y_val"], label_to_idx)

                train_ds = _SignalDataset(x_train, y_train)
                val_ds = _SignalDataset(x_val, y_val)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=int(params["batch_size"]),
                    shuffle=True,
                    num_workers=num_workers,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=int(params["batch_size"]),
                    shuffle=False,
                    num_workers=num_workers,
                )

                model = _build_model(model_name, in_channels=x_train.shape[1], num_classes=num_classes, params=params).to(device_obj)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(params["lr"]),
                    weight_decay=float(params["weight_decay"]),
                )
                criterion = nn.CrossEntropyLoss(label_smoothing=float(params["label_smoothing"]))
                scheduler = None
                if use_lr_scheduler:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=float(params["scheduler_factor"]),
                        patience=scheduler_patience,
                        min_lr=scheduler_min_lr,
                    )

                best_val_acc = -1.0
                best_epoch = -1
                best_state = None
                stale = 0
                writer = None
                if tensorboard_log_dir_root is not None:
                    from torch.utils.tensorboard import SummaryWriter

                    writer = SummaryWriter(
                        log_dir=(
                            f"{tensorboard_log_dir_root}/trial_{trial_id:04d}"
                            f"_{model_name}/fold_{int(fold['fold'])}"
                        )
                    )

                for epoch in range(int(epochs)):
                    train_loss, train_acc = _run_epoch(model, train_loader, criterion, device_obj, optimizer=optimizer)
                    val_loss, val_acc = _run_epoch(model, val_loader, criterion, device_obj, optimizer=None)

                    if writer is not None and (epoch + 1) % int(tensorboard_write_every_n) == 0:
                        writer.add_scalar("loss/train", train_loss, epoch)
                        writer.add_scalar("loss/val", val_loss, epoch)
                        writer.add_scalar("acc/train", train_acc, epoch)
                        writer.add_scalar("acc/val", val_acc, epoch)

                    if verbose_train and (epoch + 1) % int(tensorboard_write_every_n) == 0:
                        lr_now = optimizer.param_groups[0]["lr"]
                        print(
                            f"trial={trial_id} fold={int(fold['fold'])} epoch={epoch + 1}/{epochs} "
                            f"lr={lr_now:.6g} train_acc={train_acc:.4f} train_loss={train_loss:.4f} "
                            f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
                        )

                    if scheduler is not None:
                        scheduler.step(val_loss)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        best_state = copy.deepcopy(model.state_dict())
                        stale = 0
                    else:
                        stale += 1

                    if stale >= int(patience):
                        break

                if best_state is not None:
                    model.load_state_dict(best_state)
                if writer is not None:
                    writer.close()

                fold_best_accs.append(float(best_val_acc))
                fold_best_epochs.append(int(best_epoch))

                # Explicit cleanup helps long-running grid searches on GPU.
                del best_state, writer, scheduler, criterion, optimizer, model, train_loader, val_loader, train_ds, val_ds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            mean_best_val_acc = float(np.mean(fold_best_accs))
            trial_results.append(
                GridTrialResult(
                    trial_id=trial_id,
                    model_name=model_name,
                    params=params,
                    mean_best_val_acc=mean_best_val_acc,
                    fold_best_val_accs=fold_best_accs,
                    fold_best_epochs=fold_best_epochs,
                    duration_sec=time.perf_counter() - t0,
                )
            )

            if print_progress:
                print(
                    f"[GridSearch] done trial={trial_id} model={model_name} "
                    f"mean_best_val_acc={mean_best_val_acc:.4f} time={time.perf_counter() - t0:.1f}s"
                )

            # Trial-level cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as exc:  # pragma: no cover - runtime safety path
            result = GridTrialResult(
                trial_id=trial_id,
                model_name=model_name,
                params=params,
                mean_best_val_acc=float("-inf"),
                fold_best_val_accs=[],
                fold_best_epochs=[],
                duration_sec=time.perf_counter() - t0,
                status="error",
                error=str(exc),
            )
            trial_results.append(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if print_progress:
                print(f"[GridSearch] failed trial={trial_id} model={model_name} error={exc}")
            if not continue_on_error:
                raise

    valid_trials = [t for t in trial_results if t.status == "ok"]
    if not valid_trials:
        raise RuntimeError("All grid-search trials failed.")

    valid_trials_sorted = sorted(valid_trials, key=lambda t: t.mean_best_val_acc, reverse=True)

    return {
        "all_trials": trial_results,
        "valid_trials": valid_trials,
        "valid_trials_sorted": valid_trials_sorted,
        "best_trial": valid_trials_sorted[0],
        "label_to_idx": label_to_idx,
        "device": str(device_obj),
        "total_duration_sec": time.perf_counter() - started,
        "total_jobs": total_jobs,
    }


def print_top_grid_results(grid_out: Dict[str, Any], top_k: int = 100) -> None:
    """Print the top-k successful grid-search results."""
    top = grid_out["valid_trials_sorted"][: max(1, top_k)]
    print(f"Top {len(top)} / {len(grid_out['valid_trials'])} successful trials")
    for rank, trial in enumerate(top, start=1):
        print(
            f"#{rank} trial={trial.trial_id} model={trial.model_name} "
            f"mean_best_val_acc={trial.mean_best_val_acc:.4f} "
            f"time={trial.duration_sec:.1f}s params={trial.params}"
        )


def summarize_top_param_frequencies(
    grid_out: Dict[str, Any],
    top_k: int = 100,
    include_model_name: bool = True,
) -> Dict[str, Dict[Any, int]]:
    """Count how often each parameter value appears in the top-k valid trials."""
    top_trials = grid_out["valid_trials_sorted"][: max(1, top_k)]
    if not top_trials:
        return {}

    counters: Dict[str, Counter] = {}
    for trial in top_trials:
        if include_model_name:
            counters.setdefault("model_name", Counter())[trial.model_name] += 1
        for param_name, param_value in trial.params.items():
            counters.setdefault(param_name, Counter())[param_value] += 1

    result = {name: dict(counter.most_common()) for name, counter in counters.items()}

    # Also include the joint frequency of (l1, l2) combinations in the same summary dict.
    l1_l2_counter: Counter = Counter()
    for trial in top_trials:
        l1 = trial.params.get("l1")
        l2 = trial.params.get("l2")
        if l1 is not None and l2 is not None:
            l1_l2_counter[(l1, l2)] += 1
    if l1_l2_counter:
        result["l1_l2_combo"] = dict(l1_l2_counter.most_common())

    return result


def print_top_param_frequencies(
    grid_out: Dict[str, Any],
    top_k: int = 100,
    include_model_name: bool = True,
) -> None:
    """Print most frequent parameter values in the top-k valid trials."""
    top_trials = grid_out["valid_trials_sorted"][: max(1, top_k)]
    print(f"Parameter frequencies in top {len(top_trials)} valid trials")
    freq_map = summarize_top_param_frequencies(grid_out, top_k=top_k, include_model_name=include_model_name)
    for param_name, freq_dict in freq_map.items():
        print(f"[{param_name}]")
        for value, count in freq_dict.items():
            print(f"  {value}: {count}")


def summarize_top_l1_l2_combinations(
    grid_out: Dict[str, Any],
    top_k: int = 100,
) -> Dict[Tuple[Any, Any], int]:
    """Count (l1, l2) pair occurrences in the top-k valid trials."""
    top_trials = grid_out["valid_trials_sorted"][: max(1, top_k)]
    counter: Counter = Counter()
    for trial in top_trials:
        l1 = trial.params.get("l1")
        l2 = trial.params.get("l2")
        if l1 is not None and l2 is not None:
            counter[(l1, l2)] += 1
    return dict(counter.most_common())


def print_top_l1_l2_combinations(
    grid_out: Dict[str, Any],
    top_k: int = 100,
) -> None:
    """Print (l1, l2) pair frequencies in the top-k valid trials."""
    combo_freq = summarize_top_l1_l2_combinations(grid_out, top_k=top_k)
    n_trials = min(max(1, top_k), len(grid_out["valid_trials_sorted"]))
    print(f"(l1, l2) combination frequencies in top {n_trials} valid trials")
    for (l1, l2), count in combo_freq.items():
        print(f"  (l1={l1}, l2={l2}): {count}")


def save_grid_search_results(grid_out: Dict[str, Any], output_dir: str, prefix: str = "grid_search") -> Dict[str, str]:
    """Save all grid-search results to disk (CSV + JSON summary).

    Files written:
    - `{prefix}_all_trials.csv`
    - `{prefix}_valid_trials_sorted.csv`
    - `{prefix}_summary.json`
    """
    os.makedirs(output_dir, exist_ok=True)

    all_trials_csv = os.path.join(output_dir, f"{prefix}_all_trials.csv")
    valid_sorted_csv = os.path.join(output_dir, f"{prefix}_valid_trials_sorted.csv")
    summary_json = os.path.join(output_dir, f"{prefix}_summary.json")

    def _trial_to_row(trial: GridTrialResult) -> Dict[str, Any]:
        row = {
            "trial_id": trial.trial_id,
            "model_name": trial.model_name,
            "status": trial.status,
            "mean_best_val_acc": trial.mean_best_val_acc,
            "duration_sec": trial.duration_sec,
            "fold_best_val_accs": json.dumps(trial.fold_best_val_accs),
            "fold_best_epochs": json.dumps(trial.fold_best_epochs),
            "error": trial.error or "",
        }
        for k, v in trial.params.items():
            row[f"param__{k}"] = v
        return row

    all_rows = [_trial_to_row(t) for t in grid_out["all_trials"]]
    valid_rows = [_trial_to_row(t) for t in grid_out["valid_trials_sorted"]]

    # Use pandas if available for convenience; otherwise write CSV manually.
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(all_rows).to_csv(all_trials_csv, index=False)
        pd.DataFrame(valid_rows).to_csv(valid_sorted_csv, index=False)
    except Exception:
        import csv

        def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
            if not rows:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    f.write("")
                return
            fieldnames = sorted({k for row in rows for k in row.keys()})
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

        _write_csv(all_trials_csv, all_rows)
        _write_csv(valid_sorted_csv, valid_rows)

    best = grid_out["best_trial"]
    summary = {
        "total_jobs": grid_out.get("total_jobs"),
        "num_all_trials": len(grid_out.get("all_trials", [])),
        "num_valid_trials": len(grid_out.get("valid_trials", [])),
        "total_duration_sec": grid_out.get("total_duration_sec"),
        "device": grid_out.get("device"),
        "label_to_idx": grid_out.get("label_to_idx"),
        "best_trial": {
            "trial_id": best.trial_id,
            "model_name": best.model_name,
            "mean_best_val_acc": best.mean_best_val_acc,
            "duration_sec": best.duration_sec,
            "params": best.params,
            "fold_best_val_accs": best.fold_best_val_accs,
            "fold_best_epochs": best.fold_best_epochs,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "all_trials_csv": all_trials_csv,
        "valid_trials_sorted_csv": valid_sorted_csv,
        "summary_json": summary_json,
    }


if __name__ == "__main__":
    print("Utility module for grid search across CNN/TCN model families.")
    print("Import and call `run_grid_search_three_models(cv_folds, ...)` from your notebook/script.")
    print(
        "Default search-space jobs across 4 models: "
        f"{estimate_total_trials(default_search_space(), ('shared_backbone_2ch', 'two_tower_late_fusion', 'two_tower_mid_fusion_cnn', 'tcn_classifier'))}"
    )
