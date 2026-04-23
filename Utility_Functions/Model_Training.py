import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from Utility_Functions.Model_Structure import (
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


@dataclass  # Function similar to C structure: A lightweight object designed primarily for storing and access various types of data.
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


@dataclass  # Function similar to C structure: A lightweight object designed primarily for storing and access various types of data.
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


class CNNTrainer:
    """Trainer that encapsulates CV training, logging, and predictions."""

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        if self.config.tensorboard_write_every_n <= 0:
            raise ValueError("tensorboard_write_every_n must be >= 1")
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

        train_ds = MicrowaveSignalDataset(
            x_train,
            y_train,
            random_shift_max_points=self.config.random_shift_max_points,
            random_shift_fill_mode=self.config.random_shift_fill_mode,
        )
        val_ds = MicrowaveSignalDataset(x_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

        model = self._create_model(in_channels=x_train.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
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
