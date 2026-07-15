from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Repeated_Measurements.dataset import RepeatedMeasurementSplit
from Repeated_Measurements.evaluation import compute_confusion_matrix

from .model import JointOrdinalCNN1D


class SignalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, ordinal_targets: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.ordinal_targets = torch.as_tensor(ordinal_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.ordinal_targets[index]


@dataclass
class Ordinal50ClassConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-1
    device: Optional[str] = None
    num_workers: int = 0
    verbose: bool = True
    use_lr_scheduler: bool = True
    scheduler_eta_min: float = 1e-5
    use_test_early_stopping: bool = True
    early_stopping_metric: str = "sample_joint_acc"
    early_stopping_patience: int = 100


@dataclass
class Ordinal50ClassFoldResult:
    fold: int
    model: nn.Module
    history: Dict[str, List[float]]
    measurement_joint_acc: float
    sample_joint_acc: float
    measurement_liquid_acc: float
    sample_liquid_acc: float
    measurement_concentration_acc: float
    sample_concentration_acc: float
    y_measurement_true: np.ndarray
    y_measurement_pred: np.ndarray
    y_sample_true: np.ndarray
    y_sample_pred: np.ndarray
    measurement_confusion_matrix: np.ndarray
    sample_confusion_matrix: np.ndarray
    joint_label_to_idx: Dict[str, int]
    idx_to_joint_label: Dict[int, str]


def _split_joint_label(label: str) -> tuple[str, str]:
    if "__" not in label:
        raise ValueError(
            "Ordinal-50 training requires joint labels formatted as 'liquid__concentration'."
        )
    liquid, concentration = label.split("__", maxsplit=1)
    return liquid, concentration


def _concentration_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("-"))


def _joint_sort_key(label: str) -> tuple[str, tuple[int, ...]]:
    liquid, concentration = _split_joint_label(label)
    return liquid, _concentration_sort_key(concentration)


def _ordinal_targets(y: np.ndarray, num_classes: int) -> np.ndarray:
    thresholds = np.arange(num_classes - 1, dtype=np.int64)
    return (y[:, None] > thresholds[None, :]).astype(np.float32, copy=False)


class Ordinal50ClassTrainer:
    def __init__(self, config: Optional[Ordinal50ClassConfig] = None):
        self.config = config or Ordinal50ClassConfig()
        self.device = torch.device(
            self.config.device if self.config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _validate_config(self) -> None:
        if self.config.epochs <= 0:
            raise ValueError("epochs must be >= 1")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if self.config.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.config.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.config.early_stopping_metric.lower() not in {
            "measurement_joint_acc",
            "sample_joint_acc",
            "measurement_liquid_acc",
            "sample_liquid_acc",
            "measurement_concentration_acc",
            "sample_concentration_acc",
        }:
            raise ValueError("Unsupported early_stopping_metric")
        if self.config.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be >= 1")

    def _create_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        if not self.config.use_lr_scheduler:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.scheduler_eta_min,
        )

    def _run_train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0

        for x_batch, y_batch, ordinal_target in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            ordinal_target = ordinal_target.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, ordinal_target)
            loss.backward()
            optimizer.step()

            preds = torch.sum(torch.sigmoid(logits) > 0.5, dim=1)
            batch_size = int(x_batch.size(0))
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == y_batch).sum().item())
            total_items += batch_size

        return total_loss / max(total_items, 1), total_correct / max(total_items, 1)

    @torch.no_grad()
    def _predict_logits(
        self,
        model: nn.Module,
        x: np.ndarray,
    ) -> np.ndarray:
        model.eval()
        loader = DataLoader(
            SignalDataset(
                x,
                np.zeros(len(x), dtype=np.int64),
                np.zeros((len(x), 1), dtype=np.float32),
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        outputs = []
        for x_batch, _, _ in loader:
            outputs.append(model(x_batch.to(self.device)).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32, copy=False)

    @staticmethod
    def _decode_ordinal(logits: np.ndarray) -> np.ndarray:
        return np.sum(logits > 0.0, axis=1).astype(np.int64, copy=False)

    @staticmethod
    def _labels_from_keys(keys: List[dict]) -> List[str]:
        return [f"{key['liquid']}__{key['concentration']}" for key in keys]

    def fit(self, split: RepeatedMeasurementSplit) -> Ordinal50ClassFoldResult:
        self._validate_config()

        train_keys = [split.measurement_keys[int(idx)] for idx in split.train_indices]
        test_keys = [split.measurement_keys[int(idx)] for idx in split.test_indices]
        train_joint_labels = self._labels_from_keys(train_keys)
        test_joint_labels = self._labels_from_keys(test_keys)

        label_set = sorted(set(train_joint_labels), key=_joint_sort_key)
        joint_label_to_idx = {label: idx for idx, label in enumerate(label_set)}
        idx_to_joint_label = {idx: label for label, idx in joint_label_to_idx.items()}

        y_train = np.asarray([joint_label_to_idx[label] for label in train_joint_labels], dtype=np.int64)
        ordinal_train_targets = _ordinal_targets(y_train, num_classes=len(label_set))

        train_loader = DataLoader(
            SignalDataset(split.x_train, y_train, ordinal_train_targets),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        model = JointOrdinalCNN1D(
            in_channels=int(split.x_train.shape[1]),
            num_classes=len(label_set),
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._create_lr_scheduler(optimizer)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_joint_acc": [],
            "stop_measurement_joint_acc": [],
            "stop_sample_joint_acc": [],
        }
        stop_metric = self.config.early_stopping_metric.lower()
        best_state = None
        best_score = -np.inf
        best_epoch = -1
        stale_epochs = 0

        for epoch_idx in range(self.config.epochs):
            train_loss, train_joint_acc = self._run_train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
            if scheduler is not None:
                scheduler.step()

            (
                measurement_joint_acc,
                sample_joint_acc,
                measurement_liquid_acc,
                sample_liquid_acc,
                measurement_concentration_acc,
                sample_concentration_acc,
                _,
                _,
                _,
                _,
            ) = self._evaluate_predictions(
                model=model,
                x_test=split.x_test,
                test_group_ids=split.test_group_ids,
                test_joint_labels=test_joint_labels,
                joint_label_to_idx=joint_label_to_idx,
                idx_to_joint_label=idx_to_joint_label,
            )

            history["train_loss"].append(train_loss)
            history["train_joint_acc"].append(train_joint_acc)
            history["stop_measurement_joint_acc"].append(measurement_joint_acc)
            history["stop_sample_joint_acc"].append(sample_joint_acc)

            score_map = {
                "measurement_joint_acc": measurement_joint_acc,
                "sample_joint_acc": sample_joint_acc,
                "measurement_liquid_acc": measurement_liquid_acc,
                "sample_liquid_acc": sample_liquid_acc,
                "measurement_concentration_acc": measurement_concentration_acc,
                "sample_concentration_acc": sample_concentration_acc,
            }
            current_score = float(score_map[stop_metric])

            if self.config.verbose:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"fold {split.fold + 1} | Epoch {epoch_idx + 1:03d} | "
                    f"lr={lr:.6g} | train_loss={train_loss:.4f} "
                    f"train_joint_acc={train_joint_acc:.4f} "
                    f"stop_joint_sample_acc={sample_joint_acc:.4f}"
                )

            if self.config.use_test_early_stopping:
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch_idx
                    best_state = copy.deepcopy(model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                if stale_epochs >= self.config.early_stopping_patience:
                    if self.config.verbose:
                        print(
                            f"fold {split.fold + 1} | Early stopping at epoch {epoch_idx + 1}; "
                            f"best epoch was {best_epoch + 1}"
                        )
                    break

        if self.config.use_test_early_stopping and best_state is not None:
            model.load_state_dict(best_state)

        (
            measurement_joint_acc,
            sample_joint_acc,
            measurement_liquid_acc,
            sample_liquid_acc,
            measurement_concentration_acc,
            sample_concentration_acc,
            y_measurement_pred,
            y_sample_true,
            y_sample_pred,
            measurement_confusion_matrix,
        ) = self._evaluate_predictions(
            model=model,
            x_test=split.x_test,
            test_group_ids=split.test_group_ids,
            test_joint_labels=test_joint_labels,
            joint_label_to_idx=joint_label_to_idx,
            idx_to_joint_label=idx_to_joint_label,
        )
        y_measurement_true = np.asarray([joint_label_to_idx[label] for label in test_joint_labels], dtype=np.int64)
        sample_confusion_matrix = compute_confusion_matrix(
            y_sample_true,
            y_sample_pred,
            num_classes=len(joint_label_to_idx),
        )

        return Ordinal50ClassFoldResult(
            fold=split.fold,
            model=model,
            history=history,
            measurement_joint_acc=measurement_joint_acc,
            sample_joint_acc=sample_joint_acc,
            measurement_liquid_acc=measurement_liquid_acc,
            sample_liquid_acc=sample_liquid_acc,
            measurement_concentration_acc=measurement_concentration_acc,
            sample_concentration_acc=sample_concentration_acc,
            y_measurement_true=y_measurement_true,
            y_measurement_pred=y_measurement_pred,
            y_sample_true=y_sample_true,
            y_sample_pred=y_sample_pred,
            measurement_confusion_matrix=measurement_confusion_matrix,
            sample_confusion_matrix=sample_confusion_matrix,
            joint_label_to_idx=joint_label_to_idx,
            idx_to_joint_label=idx_to_joint_label,
        )

    def _evaluate_predictions(
        self,
        *,
        model: nn.Module,
        x_test: np.ndarray,
        test_group_ids: np.ndarray,
        test_joint_labels: List[str],
        joint_label_to_idx: Dict[str, int],
        idx_to_joint_label: Dict[int, str],
    ) -> tuple[float, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logits = self._predict_logits(model, x_test)
        y_measurement_pred = self._decode_ordinal(logits)
        y_measurement_true = np.asarray([joint_label_to_idx[label] for label in test_joint_labels], dtype=np.int64)
        measurement_joint_acc = float((y_measurement_true == y_measurement_pred).mean())

        measurement_true_joint_labels = [idx_to_joint_label[int(idx)] for idx in y_measurement_true]
        measurement_pred_joint_labels = [idx_to_joint_label[int(idx)] for idx in y_measurement_pred]
        measurement_liquid_acc = float(
            np.mean([
                _split_joint_label(true_label)[0] == _split_joint_label(pred_label)[0]
                for true_label, pred_label in zip(measurement_true_joint_labels, measurement_pred_joint_labels)
            ])
        )
        measurement_concentration_acc = float(
            np.mean([
                _split_joint_label(true_label)[1] == _split_joint_label(pred_label)[1]
                for true_label, pred_label in zip(measurement_true_joint_labels, measurement_pred_joint_labels)
            ])
        )

        sample_true: List[int] = []
        sample_pred: List[int] = []
        for group_id in sorted(set(int(value) for value in test_group_ids)):
            mask = test_group_ids == group_id
            group_true_labels = [test_joint_labels[int(idx)] for idx in np.flatnonzero(mask)]
            if len(set(group_true_labels)) != 1:
                raise ValueError(f"Test group {group_id} contains multiple joint labels")

            mean_logit = logits[mask].mean(axis=0, keepdims=True)
            pred_idx = int(self._decode_ordinal(mean_logit)[0])
            sample_true.append(joint_label_to_idx[group_true_labels[0]])
            sample_pred.append(pred_idx)

        y_sample_true = np.asarray(sample_true, dtype=np.int64)
        y_sample_pred = np.asarray(sample_pred, dtype=np.int64)
        sample_joint_acc = float((y_sample_true == y_sample_pred).mean())

        sample_true_joint_labels = [idx_to_joint_label[int(idx)] for idx in y_sample_true]
        sample_pred_joint_labels = [idx_to_joint_label[int(idx)] for idx in y_sample_pred]
        sample_liquid_acc = float(
            np.mean([
                _split_joint_label(true_label)[0] == _split_joint_label(pred_label)[0]
                for true_label, pred_label in zip(sample_true_joint_labels, sample_pred_joint_labels)
            ])
        )
        sample_concentration_acc = float(
            np.mean([
                _split_joint_label(true_label)[1] == _split_joint_label(pred_label)[1]
                for true_label, pred_label in zip(sample_true_joint_labels, sample_pred_joint_labels)
            ])
        )

        measurement_confusion_matrix = compute_confusion_matrix(
            y_measurement_true,
            y_measurement_pred,
            num_classes=len(joint_label_to_idx),
        )

        return (
            measurement_joint_acc,
            sample_joint_acc,
            measurement_liquid_acc,
            sample_liquid_acc,
            measurement_concentration_acc,
            sample_concentration_acc,
            y_measurement_pred,
            y_sample_true,
            y_sample_pred,
            measurement_confusion_matrix,
        )
