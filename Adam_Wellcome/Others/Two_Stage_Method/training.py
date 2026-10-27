from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Model.model import AdamWellcomeCNN1D
from Repeated_Measurements.dataset import RepeatedMeasurementSplit
from Repeated_Measurements.evaluation import compute_confusion_matrix


class SignalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


@dataclass
class TwoStageConfig:
    epochs: int = 100
    liquid_batch_size: int = 32
    concentration_batch_size: int = 32
    liquid_lr: float = 1e-3
    concentration_lr: float = 1e-3
    liquid_weight_decay: float = 1e-2
    concentration_weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    device: Optional[str] = None
    num_workers: int = 0
    verbose: bool = True
    use_lr_scheduler: bool = True
    scheduler_eta_min: float = 1e-5
    use_test_early_stopping: bool = True
    early_stopping_patience: int = 100


@dataclass
class ClassifierResult:
    model: nn.Module
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    history: Dict[str, List[float]]


@dataclass
class TwoStageFoldResult:
    fold: int
    liquid_classifier: ClassifierResult
    concentration_classifiers: Dict[str, ClassifierResult]
    measurement_joint_acc: float
    sample_joint_acc: float
    measurement_liquid_acc: float
    sample_liquid_acc: float
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
            "Two-stage training requires joint labels formatted as 'liquid__concentration'."
        )
    liquid, concentration = label.split("__", maxsplit=1)
    return liquid, concentration


def _concentration_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("-"))


def _encode_labels(
    labels: List[str],
    *,
    sort_mode: str,
) -> tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    if sort_mode == "concentration":
        unique_labels = sorted(set(labels), key=_concentration_sort_key)
    else:
        unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y = np.asarray([label_to_idx[label] for label in labels], dtype=np.int64)
    return y, label_to_idx, idx_to_label


class TwoStageTrainer:
    def __init__(self, config: Optional[TwoStageConfig] = None):
        self.config = config or TwoStageConfig()
        self.device = torch.device(
            self.config.device if self.config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _validate_config(self) -> None:
        if self.config.epochs <= 0:
            raise ValueError("epochs must be >= 1")
        if self.config.liquid_batch_size <= 0:
            raise ValueError("liquid_batch_size must be >= 1")
        if self.config.concentration_batch_size <= 0:
            raise ValueError("concentration_batch_size must be >= 1")
        if self.config.liquid_lr <= 0:
            raise ValueError("liquid_lr must be > 0")
        if self.config.concentration_lr <= 0:
            raise ValueError("concentration_lr must be > 0")
        if self.config.liquid_weight_decay < 0:
            raise ValueError("liquid_weight_decay must be >= 0")
        if self.config.concentration_weight_decay < 0:
            raise ValueError("concentration_weight_decay must be >= 0")
        if not 0.0 <= self.config.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in the range [0, 1)")
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

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = int(x_batch.size(0))
            total_loss += float(loss.item()) * batch_size
            total_correct += int((torch.argmax(logits, dim=1) == y_batch).sum().item())
            total_items += batch_size

        return total_loss / max(total_items, 1), total_correct / max(total_items, 1)

    @torch.no_grad()
    def _predict_probabilities(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        model.eval()
        loader = DataLoader(
            SignalDataset(x, np.zeros(len(x), dtype=np.int64)),
            batch_size=self.config.liquid_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        probabilities = []
        for x_batch, _ in loader:
            logits = model(x_batch.to(self.device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probabilities).astype(np.float32, copy=False)

    def _fit_classifier(
        self,
        *,
        name: str,
        x_train: np.ndarray,
        train_labels: List[str],
        x_stop: np.ndarray,
        stop_labels: List[str],
        sort_mode: str,
        batch_size: int,
        lr: float,
        weight_decay: float,
    ) -> ClassifierResult:
        y_train, label_to_idx, idx_to_label = _encode_labels(train_labels, sort_mode=sort_mode)
        y_stop = np.asarray([label_to_idx[label] for label in stop_labels], dtype=np.int64)

        train_loader = DataLoader(
            SignalDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        model = AdamWellcomeCNN1D(
            in_channels=int(x_train.shape[1]),
            num_classes=len(label_to_idx),
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = self._create_lr_scheduler(optimizer)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "stop_acc": [],
        }
        best_state = None
        best_score = -np.inf
        best_epoch = -1
        stale_epochs = 0

        for epoch_idx in range(self.config.epochs):
            train_loss, train_acc = self._run_train_epoch(model, train_loader, criterion, optimizer)
            if scheduler is not None:
                scheduler.step()

            stop_probabilities = self._predict_probabilities(model, x_stop)
            stop_pred = np.argmax(stop_probabilities, axis=1)
            stop_acc = float((stop_pred == y_stop).mean()) if y_stop.size else 0.0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["stop_acc"].append(stop_acc)

            if self.config.verbose:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"{name} | Epoch {epoch_idx + 1:03d} | "
                    f"lr={lr:.6g} | train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f} stop_acc={stop_acc:.4f}"
                )

            if self.config.use_test_early_stopping:
                if stop_acc > best_score:
                    best_score = stop_acc
                    best_epoch = epoch_idx
                    best_state = copy.deepcopy(model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1

                if stale_epochs >= self.config.early_stopping_patience:
                    if self.config.verbose:
                        print(
                            f"{name} | Early stopping at epoch {epoch_idx + 1}; "
                            f"best epoch was {best_epoch + 1}"
                        )
                    break

        if self.config.use_test_early_stopping and best_state is not None:
            model.load_state_dict(best_state)

        return ClassifierResult(
            model=model,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            history=history,
        )

    @staticmethod
    def _labels_from_keys(keys: List[dict]) -> tuple[List[str], List[str], List[str]]:
        liquids = [key["liquid"] for key in keys]
        concentrations = [key["concentration"] for key in keys]
        joint_labels = [f"{liquid}__{concentration}" for liquid, concentration in zip(liquids, concentrations)]
        return liquids, concentrations, joint_labels

    def fit(self, split: RepeatedMeasurementSplit) -> TwoStageFoldResult:
        self._validate_config()

        train_keys = [split.measurement_keys[int(idx)] for idx in split.train_indices]
        test_keys = [split.measurement_keys[int(idx)] for idx in split.test_indices]
        train_liquids, train_concentrations, train_joint_labels = self._labels_from_keys(train_keys)
        test_liquids, test_concentrations, test_joint_labels = self._labels_from_keys(test_keys)

        for label in split.idx_to_label.values():
            _split_joint_label(label)

        liquid_classifier = self._fit_classifier(
            name=f"fold {split.fold + 1} liquid",
            x_train=split.x_train,
            train_labels=train_liquids,
            x_stop=split.x_test,
            stop_labels=test_liquids,
            sort_mode="default",
            batch_size=self.config.liquid_batch_size,
            lr=self.config.liquid_lr,
            weight_decay=self.config.liquid_weight_decay,
        )

        concentration_classifiers: Dict[str, ClassifierResult] = {}
        for liquid in sorted(set(train_liquids)):
            train_mask = np.asarray([value == liquid for value in train_liquids])
            test_mask = np.asarray([value == liquid for value in test_liquids])
            if not np.any(train_mask) or not np.any(test_mask):
                raise ValueError(f"Missing train or test data for liquid '{liquid}'")

            concentration_classifiers[liquid] = self._fit_classifier(
                name=f"fold {split.fold + 1} {liquid} concentration",
                x_train=split.x_train[train_mask],
                train_labels=[label for label, keep in zip(train_concentrations, train_mask) if keep],
                x_stop=split.x_test[test_mask],
                stop_labels=[label for label, keep in zip(test_concentrations, test_mask) if keep],
                sort_mode="concentration",
                batch_size=self.config.concentration_batch_size,
                lr=self.config.concentration_lr,
                weight_decay=self.config.concentration_weight_decay,
            )

        joint_label_to_idx = split.label_to_idx
        idx_to_joint_label = split.idx_to_label
        y_measurement_true = np.asarray([joint_label_to_idx[label] for label in test_joint_labels], dtype=np.int64)

        liquid_probabilities = self._predict_probabilities(liquid_classifier.model, split.x_test)
        liquid_pred_idx = np.argmax(liquid_probabilities, axis=1)
        liquid_pred_labels = [liquid_classifier.idx_to_label[int(idx)] for idx in liquid_pred_idx]

        concentration_probabilities_by_liquid = {
            liquid: self._predict_probabilities(result.model, split.x_test)
            for liquid, result in concentration_classifiers.items()
        }

        measurement_joint_predictions: List[int] = []
        for row_idx, liquid in enumerate(liquid_pred_labels):
            concentration_result = concentration_classifiers[liquid]
            concentration_probabilities = concentration_probabilities_by_liquid[liquid][row_idx]
            concentration_idx = int(np.argmax(concentration_probabilities))
            concentration = concentration_result.idx_to_label[concentration_idx]
            measurement_joint_predictions.append(joint_label_to_idx[f"{liquid}__{concentration}"])

        y_measurement_pred = np.asarray(measurement_joint_predictions, dtype=np.int64)
        measurement_joint_acc = float((y_measurement_true == y_measurement_pred).mean())
        measurement_liquid_acc = float(np.mean([true == pred for true, pred in zip(test_liquids, liquid_pred_labels)]))

        sample_true: List[int] = []
        sample_pred: List[int] = []
        sample_liquid_true: List[str] = []
        sample_liquid_pred: List[str] = []
        for group_id in sorted(set(int(value) for value in split.test_group_ids)):
            mask = split.test_group_ids == group_id
            group_indices = np.flatnonzero(mask)
            group_true_labels = [test_joint_labels[int(idx)] for idx in group_indices]
            if len(set(group_true_labels)) != 1:
                raise ValueError(f"Test group {group_id} contains multiple joint labels")

            mean_liquid_probability = liquid_probabilities[mask].mean(axis=0)
            liquid = liquid_classifier.idx_to_label[int(np.argmax(mean_liquid_probability))]
            concentration_result = concentration_classifiers[liquid]
            mean_concentration_probability = concentration_probabilities_by_liquid[liquid][mask].mean(axis=0)
            concentration = concentration_result.idx_to_label[int(np.argmax(mean_concentration_probability))]

            true_liquid, _ = _split_joint_label(group_true_labels[0])
            sample_true.append(joint_label_to_idx[group_true_labels[0]])
            sample_pred.append(joint_label_to_idx[f"{liquid}__{concentration}"])
            sample_liquid_true.append(true_liquid)
            sample_liquid_pred.append(liquid)

        y_sample_true = np.asarray(sample_true, dtype=np.int64)
        y_sample_pred = np.asarray(sample_pred, dtype=np.int64)
        sample_joint_acc = float((y_sample_true == y_sample_pred).mean())
        sample_liquid_acc = float(np.mean([true == pred for true, pred in zip(sample_liquid_true, sample_liquid_pred)]))

        num_joint_classes = len(joint_label_to_idx)
        return TwoStageFoldResult(
            fold=split.fold,
            liquid_classifier=liquid_classifier,
            concentration_classifiers=concentration_classifiers,
            measurement_joint_acc=measurement_joint_acc,
            sample_joint_acc=sample_joint_acc,
            measurement_liquid_acc=measurement_liquid_acc,
            sample_liquid_acc=sample_liquid_acc,
            y_measurement_true=y_measurement_true,
            y_measurement_pred=y_measurement_pred,
            y_sample_true=y_sample_true,
            y_sample_pred=y_sample_pred,
            measurement_confusion_matrix=compute_confusion_matrix(
                y_measurement_true,
                y_measurement_pred,
                num_classes=num_joint_classes,
            ),
            sample_confusion_matrix=compute_confusion_matrix(
                y_sample_true,
                y_sample_pred,
                num_classes=num_joint_classes,
            ),
            joint_label_to_idx=joint_label_to_idx,
            idx_to_joint_label=idx_to_joint_label,
        )
