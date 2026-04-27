import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from Feature_Implementation.Feature_Model_Structure import FeatureMLPClassifier

##  dataset for tabular feature vectors [N, F] and encoded class labels
class FeatureDataset(Dataset):
    """Dataset for tabular feature vectors [N, F] and encoded class labels."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"x must have shape [N, F], got {x.shape}")
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

##  result of training one CV fold, including best model state, best val acc, training history, and confusion matrices
@dataclass
class FeatureFoldResult:
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

##  overall training result across all CV folds, including label mappings, fold results, mean val acc, and overall confusion matrices
class FeatureTrainOutput(TypedDict):
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    device: str
    fold_results: List[FeatureFoldResult]
    mean_best_val_acc: float
    overall_confusion_count: np.ndarray
    overall_confusion_recall: np.ndarray


@dataclass
class FeatureTrainerConfig:
    class_order: Optional[Sequence[str]] = ("LOW", "TARGET", "HIGH")
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 15
    device: Optional[str] = None
    num_workers: int = 0
    tensorboard_log_dir: Optional[str] = None
    tensorboard_write_every_n: int = 10
    verbose: bool = True
    use_lr_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6


class FeatureTrainer:
    """Trainer for handcrafted tabular feature CV folds."""

    def __init__(self, config: Optional[FeatureTrainerConfig] = None):
        self.config = config or FeatureTrainerConfig()
        if self.config.tensorboard_write_every_n <= 0:
            raise ValueError("tensorboard_write_every_n must be >= 1")
        self.device = torch.device(
            self.config.device if self.config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}

    def _build_label_mapping(self, cv_folds: Sequence[Dict[str, np.ndarray]]) -> Dict[str, int]:
        if self.config.class_order is not None:
            return {label: i for i, label in enumerate(self.config.class_order)}

        labels: List[str] = []
        for fold_data in cv_folds:
            labels.extend(fold_data["y_train"].tolist())
            labels.extend(fold_data["y_val"].tolist())
        unique_labels = sorted(set(labels))
        return {label: i for i, label in enumerate(unique_labels)}

    def _encode_labels(self, labels: Iterable[str]) -> np.ndarray:
        return np.asarray([self.label_to_idx[label] for label in labels], dtype=np.int64)

    def _compute_confusion_matrices(
        self,
        y_true_idx: np.ndarray,
        y_pred_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[float, float]:
        is_train = optimizer is not None
        model.train(is_train)

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x_batch, y_batch in loader:
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
        return avg_loss, avg_acc

    @torch.no_grad()
    def _collect_predictions(self, model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _create_model(self, in_features: int) -> nn.Module:
        model = FeatureMLPClassifier(
            in_features=in_features,
            num_classes=len(self.label_to_idx),
        )
        return model.to(self.device)

    def _train_one_fold(self, fold_data: Dict[str, np.ndarray]) -> FeatureFoldResult:
        fold_id = int(fold_data["fold"])
        x_train = np.asarray(fold_data["X_train"], dtype=np.float32)
        x_val = np.asarray(fold_data["X_val"], dtype=np.float32)
        y_train = self._encode_labels(fold_data["y_train"])
        y_val = self._encode_labels(fold_data["y_val"])

        train_ds = FeatureDataset(x_train, y_train)
        val_ds = FeatureDataset(x_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

        model = self._create_model(in_features=x_train.shape[1])
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
            train_loss, train_acc = self._run_epoch(model, train_loader, criterion, optimizer=optimizer)
            val_loss, val_acc = self._run_epoch(model, val_loader, criterion, optimizer=None)

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

        return FeatureFoldResult(
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

    def train_cv(self, cv_folds: Sequence[Dict[str, np.ndarray]]) -> FeatureTrainOutput:
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
        model = model.to(self.device)
        model.eval()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def train_feature_mlp_cv(
    cv_folds: Sequence[Dict[str, np.ndarray]],
    class_order: Optional[Sequence[str]] = ("LOW", "TARGET", "HIGH"),
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    patience: int = 15,
    device: Optional[str] = None,
    num_workers: int = 0,
    tensorboard_log_dir: Optional[str] = None,
    tensorboard_write_every_n: int = 10,
    verbose: bool = True,
    use_lr_scheduler: bool = True,
    scheduler_factor: float = 0.7,
    scheduler_patience: int = 5,
    scheduler_min_lr: float = 1e-6,
) -> FeatureTrainOutput:
    config = FeatureTrainerConfig(
        class_order=class_order,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        patience=patience,
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
    return FeatureTrainer(config=config).train_cv(cv_folds)


@torch.no_grad()
def predict_prob(model: nn.Module, x: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    trainer = FeatureTrainer(config=FeatureTrainerConfig(device=device))
    return trainer.predict_prob(model, x)
