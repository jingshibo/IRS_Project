from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .dataset import SplitBundle
from .model import AdamWellcomeCNN1D


class SignalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


@dataclass
class TrainerConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 15
    device: Optional[str] = None
    num_workers: int = 0
    verbose: bool = True


@dataclass
class TrainResult:
    model: nn.Module
    history: Dict[str, List[float]]
    best_val_acc: float
    best_epoch: int
    test_acc: float
    val_acc: float
    y_val_true: np.ndarray
    y_val_pred: np.ndarray
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    val_confusion_matrix: np.ndarray
    confusion_matrix: np.ndarray
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]


class AdamWellcomeTrainer:
    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.device = torch.device(
            self.config.device if self.config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @staticmethod
    def _compute_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true_idx, pred_idx in zip(y_true, y_pred):
            confusion[int(true_idx), int(pred_idx)] += 1
        return confusion

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> tuple[float, float]:
        is_train = optimizer is not None
        model.train(is_train)

        total_loss = 0.0
        total_correct = 0
        total_items = 0

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

            batch_size = int(x_batch.size(0))
            total_loss += loss.item() * batch_size
            total_correct += int((torch.argmax(logits, dim=1) == y_batch).sum().item())
            total_items += batch_size

        return total_loss / max(total_items, 1), total_correct / max(total_items, 1)

    @torch.no_grad()
    def _predict(self, model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        y_true = []
        y_pred = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(preds)
        return np.concatenate(y_true), np.concatenate(y_pred)

    def fit(self, splits: SplitBundle) -> TrainResult:
        train_loader = DataLoader(
            SignalDataset(splits.x_train, splits.y_train),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        val_loader = DataLoader(
            SignalDataset(splits.x_val, splits.y_val),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        test_loader = DataLoader(
            SignalDataset(splits.x_test, splits.y_test),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        model = AdamWellcomeCNN1D(
            in_channels=int(splits.x_train.shape[1]),
            num_classes=len(splits.label_to_idx),
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = -np.inf
        best_epoch = -1
        stale_epochs = 0

        for epoch_idx in range(self.config.epochs):
            train_loss, train_acc = self._run_epoch(model, train_loader, criterion, optimizer=optimizer)
            val_loss, val_acc = self._run_epoch(model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if self.config.verbose:
                print(
                    f"Epoch {epoch_idx + 1:03d} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch_idx
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= self.config.patience:
                if self.config.verbose:
                    print(f"Early stopping at epoch {epoch_idx + 1}")
                break

        model.load_state_dict(best_state)
        y_val_true, y_val_pred = self._predict(model, val_loader)
        y_test_true, y_test_pred = self._predict(model, test_loader)
        val_acc = float((y_val_true == y_val_pred).mean()) if y_val_true.size else 0.0
        test_acc = float((y_test_true == y_test_pred).mean()) if y_test_true.size else 0.0
        val_confusion_matrix = self._compute_confusion_matrix(
            y_val_true,
            y_val_pred,
            num_classes=len(splits.label_to_idx),
        )
        confusion_matrix = self._compute_confusion_matrix(
            y_test_true,
            y_test_pred,
            num_classes=len(splits.label_to_idx),
        )

        return TrainResult(
            model=model,
            history=history,
            best_val_acc=float(best_val_acc),
            best_epoch=int(best_epoch),
            test_acc=test_acc,
            val_acc=val_acc,
            y_val_true=y_val_true,
            y_val_pred=y_val_pred,
            y_test_true=y_test_true,
            y_test_pred=y_test_pred,
            val_confusion_matrix=val_confusion_matrix,
            confusion_matrix=confusion_matrix,
            label_to_idx=splits.label_to_idx,
            idx_to_label=splits.idx_to_label,
        )
