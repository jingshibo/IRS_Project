import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from IRS_Insecticide_Residual.Raw_Data_Implementation.Models.Model_Structure import (
    DualBranchFusionCNNClassifier,
    DualBranchOneDCNNClassifier,
    MultiScaleOneDCNNClassifier,
    OneDCNNClassifier,
    TCNClassifier,
)


class MicrowaveSignalDataset(Dataset):
    """Dataset for signal tensors and encoded class labels.

    x shape: [N, C, L]
    - N: number of samples
    - C: channels (here 2: raw + diff)
    - L: signal length
    y shape: [N], integer class IDs for CrossEntropyLoss
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        random_shift_max_points: int = 0,
        random_shift_fill_mode: str = "zero",
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.random_shift_max_points = int(random_shift_max_points)
        self.random_shift_fill_mode = random_shift_fill_mode
        if self.random_shift_max_points < 0:
            raise ValueError(f"random_shift_max_points must be >= 0, got {self.random_shift_max_points}")
        if self.random_shift_fill_mode not in {"zero", "edge", "wrap"}:
            raise ValueError(
                "random_shift_fill_mode must be one of {'zero', 'edge', 'wrap'}, "
                f"got '{self.random_shift_fill_mode}'."
            )

    def _apply_random_shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.random_shift_max_points == 0:
            return x

        shift = int(torch.randint(-self.random_shift_max_points, self.random_shift_max_points + 1, (1,)).item())
        if shift == 0:
            return x

        shifted = torch.roll(x, shifts=shift, dims=-1)
        if self.random_shift_fill_mode == "wrap":
            return shifted

        if shift > 0:
            if self.random_shift_fill_mode == "zero":
                shifted[:, :shift] = 0
            else:
                shifted[:, :shift] = x[:, :1].expand(-1, shift)
        else:
            tail_width = -shift
            if self.random_shift_fill_mode == "zero":
                shifted[:, shift:] = 0
            else:
                shifted[:, shift:] = x[:, -1:].expand(-1, tail_width)
        return shifted

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = self.x[idx]
        if self.random_shift_max_points > 0:
            x = self._apply_random_shift(x.clone())
        return x, self.y[idx]


class FocalLoss(nn.Module):
    """Cross-entropy focal loss with optional class weights and label smoothing."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - pt).pow(self.gamma)
        loss = focal_factor * cross_entropy
        if self.weight is not None:
            sample_weights = self.weight.gather(dim=0, index=targets)
            loss = loss * sample_weights
            return loss.sum() / sample_weights.sum().clamp_min(1e-12)
        return loss.mean()


@dataclass
class FoldResult:
    """Container for one fold's best model and validation outputs."""

    fold: int
    model: nn.Module
    best_val_acc: float
    best_epoch: int
    history: Dict[str, List[float]]
    y_true_idx: np.ndarray
    y_pred_idx: np.ndarray
    y_prob: np.ndarray
    y_true_label: List[str]
    y_pred_label: List[str]
    confusion_count: np.ndarray
    confusion_recall: np.ndarray


class TrainOutput(TypedDict):
    """Typed dictionary returned by training APIs."""

    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    device: str
    fold_results: List[FoldResult]
    mean_best_val_acc: float
    overall_confusion_count: np.ndarray
    overall_confusion_recall: np.ndarray


@dataclass
class TrainerConfig:
    """Configuration for cross-validation training."""

    class_order: Optional[Sequence[str]] = ("LOW", "TARGET", "HIGH")
    model_name: str = "shared_backbone_2ch"
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 15
    random_shift_max_points: int = 0
    random_shift_fill_mode: str = "zero"
    device: Optional[str] = None
    num_workers: int = 0
    tensorboard_log_dir: Optional[str] = None
    tensorboard_write_every_n: int = 10
    verbose: bool = True
    use_lr_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    imbalance_strategy: str = "none"
    focal_gamma: float = 2.0
    class_weight_beta: float = 1.0
    manual_class_weights: Optional[Dict[str, float]] = None
    train_class_sample_limits: Optional[Dict[str, int]] = None
    train_class_sample_seed: int = 42


class CNNTrainer:
    """Trainer that encapsulates CV training, logging, and predictions."""

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        if self.config.tensorboard_write_every_n <= 0:
            raise ValueError("tensorboard_write_every_n must be >= 1")
        valid_imbalance_strategies = {
            "none",
            "class_weight",
            "weighted_sampler",
            "class_weight_and_sampler",
            "focal_loss",
            "class_weight_focal_loss",
            "soft_class_weight",
            "manual_class_weight",
            "soft_class_weight_focal_loss",
        }
        if self.config.imbalance_strategy not in valid_imbalance_strategies:
            raise ValueError(
                f"imbalance_strategy must be one of {sorted(valid_imbalance_strategies)}, "
                f"got {self.config.imbalance_strategy!r}"
            )
        self.device = torch.device(
            self.config.device if self.config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Map legacy model names to canonical names."""
        alias_map = {
            "single_branch": "shared_backbone_2ch",
            "multi_scale": "multi_scale_1d_cnn",
            "dual_branch": "two_tower_late_fusion",
            "dual_branch_fusion_cnn": "two_tower_mid_fusion_cnn",
            "tcn": "tcn_classifier",
        }
        return alias_map.get(model_name, model_name)

    def _build_label_mapping(self, cv_folds: Sequence[Dict[str, np.ndarray]]) -> Dict[str, int]:
        """Build label->index map from class order or labels seen in folds."""
        if self.config.class_order is not None:
            return {label: i for i, label in enumerate(self.config.class_order)}

        labels: List[str] = []
        for fold_data in cv_folds:
            labels.extend(fold_data["y_train"].tolist())
            labels.extend(fold_data["y_val"].tolist())
        unique_labels = sorted(set(labels))
        return {label: i for i, label in enumerate(unique_labels)}

    def _encode_labels(self, labels: Iterable[str]) -> np.ndarray:
        """Convert string labels like LOW/TARGET/HIGH to integer IDs."""
        return np.asarray([self.label_to_idx[label] for label in labels], dtype=np.int64)

    def _compute_class_weights(self, y_encoded: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Return inverse-frequency class weights in label-index order."""
        if beta < 0:
            raise ValueError(f"class_weight_beta must be >= 0, got {beta}")
        num_classes = len(self.label_to_idx)
        counts = np.bincount(y_encoded, minlength=num_classes).astype(np.float32)
        if np.any(counts == 0):
            missing = [self.idx_to_label[idx] for idx, count in enumerate(counts) if count == 0]
            raise ValueError(f"Cannot compute class weights because train fold is missing classes: {missing}")
        weights = len(y_encoded) / (num_classes * counts)
        return np.power(weights, beta).astype(np.float32)

    def _manual_class_weights(self) -> np.ndarray:
        """Return user-provided class weights in label-index order."""
        if not self.config.manual_class_weights:
            raise ValueError("manual_class_weights must be provided when using manual_class_weight")

        missing = [
            label
            for label in self.label_to_idx
            if label not in self.config.manual_class_weights
        ]
        if missing:
            raise ValueError(f"manual_class_weights is missing labels: {missing}")

        weights = np.empty(len(self.label_to_idx), dtype=np.float32)
        for label, idx in self.label_to_idx.items():
            weight = float(self.config.manual_class_weights[label])
            if weight <= 0:
                raise ValueError(f"manual class weight for {label!r} must be > 0, got {weight}")
            weights[idx] = weight
        return weights

    def _apply_train_class_sample_limits(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        fold_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly cap per-class training samples for the current fold."""
        if not self.config.train_class_sample_limits:
            return x_train, y_train

        selected_indices = []
        rng = np.random.default_rng(int(self.config.train_class_sample_seed) + fold_id)

        for label, class_idx in self.label_to_idx.items():
            class_indices = np.flatnonzero(y_train == class_idx)
            limit = self.config.train_class_sample_limits.get(label)
            if limit is None or len(class_indices) <= limit:
                selected_indices.append(class_indices)
                continue
            if limit < 1:
                raise ValueError(f"train_class_sample_limits[{label!r}] must be >= 1, got {limit}")
            selected_indices.append(rng.choice(class_indices, size=int(limit), replace=False))

        keep_indices = np.concatenate(selected_indices)
        rng.shuffle(keep_indices)
        return x_train[keep_indices], y_train[keep_indices]

    @staticmethod
    def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute classification accuracy from logits and integer targets."""
        preds = torch.argmax(logits, dim=1)
        return (preds == targets).float().mean().item()

    def _compute_confusion_matrices(
        self,
        y_true_idx: np.ndarray,
        y_pred_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return confusion matrix count and row-wise recall ratio matrix."""
        num_classes = len(self.label_to_idx)
        count = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true_idx, y_pred_idx):
            count[int(t), int(p)] += 1

        row_sum = count.sum(axis=1, keepdims=True)
        recall = np.divide(
            count.astype(np.float64),
            row_sum,
            out=np.zeros_like(count, dtype=np.float64),
            where=row_sum != 0,
        )
        return count, recall

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        fold_id: int,
        epoch_idx: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[float, float, int]:
        """Run one train/validation epoch and return average (loss, acc)."""
        is_train = optimizer is not None
        model.train(is_train)

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        total_batches = len(loader)
        for batch_idx, (x_batch, y_batch) in enumerate(loader, start=1):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(is_train):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                if is_train:
                    loss.backward()
                    optimizer.step()

            batch_size = x_batch.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (torch.argmax(logits, dim=1) == y_batch).sum().item()
            running_total += batch_size

        avg_loss = running_loss / max(running_total, 1)
        avg_acc = running_correct / max(running_total, 1)
        return avg_loss, avg_acc, total_batches

    @torch.no_grad()
    def _collect_predictions(self, model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect y_true, y_pred, and softmax probabilities over a dataloader."""
        model.eval()
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        all_prob: List[np.ndarray] = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_true.append(y_batch.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

        y_true = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int64)
        y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.array([], dtype=np.int64)
        y_prob = np.concatenate(all_prob, axis=0) if all_prob else np.empty((0, 0), dtype=np.float32)
        return y_true, y_pred, y_prob

    def _create_model(self, in_channels: int) -> nn.Module:
        """Create and return a fresh CNN model on configured device."""
        model_name = self._normalize_model_name(self.config.model_name)
        if model_name == "shared_backbone_2ch":
            model = OneDCNNClassifier(
                in_channels=in_channels,
                num_classes=len(self.label_to_idx),
            )
        elif model_name == "multi_scale_1d_cnn":
            model = MultiScaleOneDCNNClassifier(
                in_channels=in_channels,
                num_classes=len(self.label_to_idx),
            )
        elif model_name == "two_tower_late_fusion":
            if in_channels != 2:
                raise ValueError(
                    f"Model '{model_name}' requires exactly 2 channels (raw + diff-style), got {in_channels}."
                )
            model = DualBranchOneDCNNClassifier(
                num_classes=len(self.label_to_idx),
            )
        elif model_name == "two_tower_mid_fusion_cnn":
            if in_channels != 2:
                raise ValueError(
                    f"Model '{model_name}' requires exactly 2 channels (raw + diff-style), got {in_channels}."
                )
            model = DualBranchFusionCNNClassifier(
                num_classes=len(self.label_to_idx),
            )
        elif model_name == "tcn_classifier":
            model = TCNClassifier(
                in_channels=in_channels,
                num_classes=len(self.label_to_idx),
            )
        else:
            raise ValueError(
                f"Unsupported model_name='{self.config.model_name}'. "
                f"Use 'shared_backbone_2ch', 'multi_scale_1d_cnn', 'two_tower_late_fusion', 'two_tower_mid_fusion_cnn', or 'tcn_classifier'."
            )
        return model.to(self.device)

    def _train_one_fold(self, fold_data: Dict[str, np.ndarray]) -> FoldResult:
        """Train one CV fold and return fold-level metrics and predictions."""
        fold_id = int(fold_data["fold"])
        x_train = np.asarray(fold_data["X_train"], dtype=np.float32)
        x_val = np.asarray(fold_data["X_val"], dtype=np.float32)
        y_train = self._encode_labels(fold_data["y_train"])
        y_val = self._encode_labels(fold_data["y_val"])
        x_train, y_train = self._apply_train_class_sample_limits(x_train, y_train, fold_id=fold_id)

        train_ds = MicrowaveSignalDataset(
            x_train,
            y_train,
            random_shift_max_points=self.config.random_shift_max_points,
            random_shift_fill_mode=self.config.random_shift_fill_mode,
        )
        val_ds = MicrowaveSignalDataset(x_val, y_val)

        class_weights = None
        weighted_loss_strategies = {
            "class_weight",
            "class_weight_and_sampler",
            "class_weight_focal_loss",
            "soft_class_weight",
            "manual_class_weight",
            "soft_class_weight_focal_loss",
        }
        weighted_sampler_strategies = {"weighted_sampler", "class_weight_and_sampler"}
        focal_loss_strategies = {"focal_loss", "class_weight_focal_loss", "soft_class_weight_focal_loss"}

        if self.config.imbalance_strategy in weighted_loss_strategies | weighted_sampler_strategies:
            if self.config.imbalance_strategy in {"soft_class_weight", "soft_class_weight_focal_loss"}:
                class_weights = self._compute_class_weights(y_train, beta=self.config.class_weight_beta)
            elif self.config.imbalance_strategy == "manual_class_weight":
                class_weights = self._manual_class_weights()
            else:
                class_weights = self._compute_class_weights(y_train)
        if self.config.imbalance_strategy in weighted_sampler_strategies:
            if class_weights is None:
                raise RuntimeError("class_weights must be computed before using weighted_sampler")
            sample_weights = class_weights[y_train]
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=self.config.batch_size,
                sampler=train_sampler,
                num_workers=self.config.num_workers,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                drop_last=True,
            )
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

        model = self._create_model(in_channels=x_train.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        loss_weights = None
        if self.config.imbalance_strategy in weighted_loss_strategies:
            if class_weights is None:
                raise RuntimeError("class_weights must be computed before using class_weight")
            loss_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        if self.config.imbalance_strategy in focal_loss_strategies:
            criterion = FocalLoss(
                weight=loss_weights,
                gamma=self.config.focal_gamma,
                label_smoothing=self.config.label_smoothing,
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=self.config.label_smoothing)
        scheduler = None
        if self.config.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr,
            )

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = -1.0
        best_epoch = -1
        best_state = None
        stale_epochs = 0

        writer = None
        if self.config.tensorboard_log_dir is not None:
            writer = SummaryWriter(log_dir=f"{self.config.tensorboard_log_dir}/fold_{fold_id}")

        for epoch in range(self.config.epochs):
            should_log_epoch = (epoch + 1) % self.config.tensorboard_write_every_n == 0
            train_loss, train_acc, train_total_batches = self._run_epoch(model, train_loader, criterion, fold_id=fold_id, epoch_idx=epoch, optimizer=optimizer)
            val_loss, val_acc, _ = self._run_epoch(model, val_loader, criterion, fold_id=fold_id, epoch_idx=epoch, optimizer=None)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if writer is not None and should_log_epoch:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("acc/train", train_acc, epoch)
                writer.add_scalar("acc/val", val_acc, epoch)

            if self.config.verbose and should_log_epoch:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"fold={fold_id} epoch={epoch + 1}/{self.config.epochs} "
                    f"lr={lr:.6g} "
                    f"training_accuracy={train_acc:.4f} training_loss={train_loss:.4f} "
                    f"val_accuracy={val_acc:.4f} val_loss={val_loss:.4f}"
                )

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= self.config.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        if writer is not None:
            writer.close()

        y_true_idx, y_pred_idx, y_prob = self._collect_predictions(model, val_loader)
        y_true_label = [self.idx_to_label[int(idx)] for idx in y_true_idx]
        y_pred_label = [self.idx_to_label[int(idx)] for idx in y_pred_idx]
        confusion_count, confusion_recall = self._compute_confusion_matrices(y_true_idx, y_pred_idx)

        return FoldResult(
            fold=fold_id,
            model=model,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            history=history,
            y_true_idx=y_true_idx,
            y_pred_idx=y_pred_idx,
            y_prob=y_prob,
            y_true_label=y_true_label,
            y_pred_label=y_pred_label,
            confusion_count=confusion_count,
            confusion_recall=confusion_recall,
        )

    def train_cv(self, cv_folds: Sequence[Dict[str, np.ndarray]]) -> TrainOutput:
        """Train across all folds and return aggregate results."""
        if len(cv_folds) == 0:
            raise ValueError("cv_folds is empty.")

        self.label_to_idx = self._build_label_mapping(cv_folds)
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        fold_results = [self._train_one_fold(fold_data) for fold_data in cv_folds]
        mean_val_acc = float(np.mean([res.best_val_acc for res in fold_results]))
        all_true = np.concatenate([res.y_true_idx for res in fold_results], axis=0)
        all_pred = np.concatenate([res.y_pred_idx for res in fold_results], axis=0)
        overall_confusion_count, overall_confusion_recall = self._compute_confusion_matrices(all_true, all_pred)

        return {
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "device": str(self.device),
            "fold_results": fold_results,
            "mean_best_val_acc": mean_val_acc,
            "overall_confusion_count": overall_confusion_count,
            "overall_confusion_recall": overall_confusion_recall,
        }

    @torch.no_grad()
    def predict_prob(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for input x with shape [N, C, L]."""
        model = model.to(self.device)
        model.eval()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def train_1d_cnn_cv(
    cv_folds: Sequence[Dict[str, np.ndarray]],
    class_order: Optional[Sequence[str]] = ("LOW", "TARGET", "HIGH"),
    model_name: str = "shared_backbone_2ch",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    patience: int = 15,
    random_shift_max_points: int = 0,
    random_shift_fill_mode: str = "zero",
    device: Optional[str] = None,
    num_workers: int = 0,
    tensorboard_log_dir: Optional[str] = None,
    tensorboard_write_every_n: int = 10,
    verbose: bool = True,
    use_lr_scheduler: bool = True,
    scheduler_factor: float = 0.7,
    scheduler_patience: int = 5,
    scheduler_min_lr: float = 1e-6,
    imbalance_strategy: str = "none",
    focal_gamma: float = 2.0,
    class_weight_beta: float = 1.0,
    manual_class_weights: Optional[Dict[str, float]] = None,
    train_class_sample_limits: Optional[Dict[str, int]] = None,
    train_class_sample_seed: int = 42,
) -> TrainOutput:
    """Compatibility wrapper around `CNNTrainer.fit_cv`."""
    config = TrainerConfig(
        class_order=class_order,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        patience=patience,
        random_shift_max_points=random_shift_max_points,
        random_shift_fill_mode=random_shift_fill_mode,
        device=device,
        num_workers=num_workers,
        tensorboard_log_dir=tensorboard_log_dir,
        tensorboard_write_every_n=tensorboard_write_every_n,
        verbose=verbose,
        use_lr_scheduler=use_lr_scheduler,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        scheduler_min_lr=scheduler_min_lr,
        imbalance_strategy=imbalance_strategy,
        focal_gamma=focal_gamma,
        class_weight_beta=class_weight_beta,
        manual_class_weights=manual_class_weights,
        train_class_sample_limits=train_class_sample_limits,
        train_class_sample_seed=train_class_sample_seed,
    )
    return CNNTrainer(config=config).train_cv(cv_folds)


@torch.no_grad()
def predict_prob(model: nn.Module, x: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """Compatibility wrapper around `CNNTrainer.predict_proba`."""
    trainer = CNNTrainer(config=TrainerConfig(device=device))
    return trainer.predict_prob(model, x)


def print_model_summary(
    signal_length: int = 4000,
    batch_size: int = 64,
    in_channels: int = 2,
    num_classes: int = 3,
    model_name: str = "shared_backbone_2ch",
) -> None:
    """Print model input/output shapes and parameter counts via torchinfo."""
    try:
        from torchinfo import summary
    except ImportError as exc:
        raise ImportError("torchinfo is required. Install with: pip install torchinfo") from exc

    normalized_model_name = CNNTrainer._normalize_model_name(model_name)
    if normalized_model_name == "shared_backbone_2ch":
        model = OneDCNNClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif normalized_model_name == "multi_scale_1d_cnn":
        model = MultiScaleOneDCNNClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif normalized_model_name == "two_tower_late_fusion":
        model = DualBranchOneDCNNClassifier(
            num_classes=num_classes,
        )
    elif normalized_model_name == "two_tower_mid_fusion_cnn":
        model = DualBranchFusionCNNClassifier(
            num_classes=num_classes,
        )
    elif normalized_model_name == "tcn_classifier":
        model = TCNClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            "model_name must be 'shared_backbone_2ch', 'multi_scale_1d_cnn', 'two_tower_late_fusion', 'two_tower_mid_fusion_cnn', or 'tcn_classifier'"
        )
    summary(
        model,
        input_size=(batch_size, in_channels, signal_length),
        col_names=("input_size", "output_size", "num_params"),
    )


if __name__ == "__main__":
    print_model_summary(
        signal_length=450,
        batch_size=64,
        in_channels=3,
        num_classes=3,
        model_name="shared_backbone_2ch",
    )
