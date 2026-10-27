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

from .model import OrdinalAdamWellcomeCNN1D


class SignalDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        liquid_y: np.ndarray,
        concentration_y: np.ndarray,
        ordinal_targets: np.ndarray,
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.liquid_y = torch.as_tensor(liquid_y, dtype=torch.long)
        self.concentration_y = torch.as_tensor(concentration_y, dtype=torch.long)
        self.ordinal_targets = torch.as_tensor(ordinal_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return (
            self.x[index],
            self.liquid_y[index],
            self.concentration_y[index],
            self.ordinal_targets[index],
        )


@dataclass
class OrdinalConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-1
    liquid_label_smoothing: float = 0.1
    concentration_loss_weight: float = 1.0
    device: Optional[str] = None
    num_workers: int = 0
    verbose: bool = True
    use_lr_scheduler: bool = True
    scheduler_eta_min: float = 1e-5
    use_test_early_stopping: bool = True
    early_stopping_metric: str = "sample_concentration_acc"
    early_stopping_patience: int = 100


@dataclass
class OrdinalFoldResult:
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
    liquid_label_to_idx: Dict[str, int]
    idx_to_liquid_label: Dict[int, str]
    concentration_label_to_idx: Dict[str, int]
    idx_to_concentration_label: Dict[int, str]


def _split_joint_label(label: str) -> tuple[str, str]:
    if "__" not in label:
        raise ValueError(
            "Ordinal training requires joint labels formatted as 'liquid__concentration'."
        )
    liquid, concentration = label.split("__", maxsplit=1)
    return liquid, concentration


def _concentration_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("-"))


def _ordinal_targets(y: np.ndarray, num_classes: int) -> np.ndarray:
    thresholds = np.arange(num_classes - 1, dtype=np.int64)
    return (y[:, None] > thresholds[None, :]).astype(np.float32, copy=False)


class OrdinalTrainer:
    def __init__(self, config: Optional[OrdinalConfig] = None):
        self.config = config or OrdinalConfig()
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
        if not 0.0 <= self.config.liquid_label_smoothing < 1.0:
            raise ValueError("liquid_label_smoothing must be in the range [0, 1)")
        if self.config.concentration_loss_weight <= 0:
            raise ValueError("concentration_loss_weight must be > 0")
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
        liquid_criterion: nn.Module,
        ordinal_criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float, float]:
        model.train()
        total_loss = 0.0
        total_liquid_correct = 0
        total_concentration_correct = 0
        total_items = 0

        for x_batch, liquid_y, concentration_y, ordinal_target in loader:
            x_batch = x_batch.to(self.device)
            liquid_y = liquid_y.to(self.device)
            concentration_y = concentration_y.to(self.device)
            ordinal_target = ordinal_target.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            liquid_logits, concentration_logits = model(x_batch)
            liquid_loss = liquid_criterion(liquid_logits, liquid_y)
            ordinal_loss = ordinal_criterion(concentration_logits, ordinal_target)
            loss = liquid_loss + self.config.concentration_loss_weight * ordinal_loss
            loss.backward()
            optimizer.step()

            liquid_pred = torch.argmax(liquid_logits, dim=1)
            concentration_pred = torch.sum(torch.sigmoid(concentration_logits) > 0.5, dim=1)
            batch_size = int(x_batch.size(0))
            total_loss += float(loss.item()) * batch_size
            total_liquid_correct += int((liquid_pred == liquid_y).sum().item())
            total_concentration_correct += int((concentration_pred == concentration_y).sum().item())
            total_items += batch_size

        return (
            total_loss / max(total_items, 1),
            total_liquid_correct / max(total_items, 1),
            total_concentration_correct / max(total_items, 1),
        )

    @torch.no_grad()
    def _predict_outputs(
        self,
        model: nn.Module,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        loader = DataLoader(
            SignalDataset(
                x,
                np.zeros(len(x), dtype=np.int64),
                np.zeros(len(x), dtype=np.int64),
                np.zeros((len(x), 1), dtype=np.float32),
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        liquid_probabilities = []
        concentration_logits = []
        for x_batch, _, _, _ in loader:
            liquid_logits, ordinal_logits = model(x_batch.to(self.device))
            liquid_probabilities.append(torch.softmax(liquid_logits, dim=1).cpu().numpy())
            concentration_logits.append(ordinal_logits.cpu().numpy())
        return (
            np.concatenate(liquid_probabilities).astype(np.float32, copy=False),
            np.concatenate(concentration_logits).astype(np.float32, copy=False),
        )

    @staticmethod
    def _decode_concentration(concentration_logits: np.ndarray) -> np.ndarray:
        return np.sum(concentration_logits > 0.0, axis=1).astype(np.int64, copy=False)

    @staticmethod
    def _labels_from_keys(keys: List[dict]) -> tuple[List[str], List[str], List[str]]:
        liquids = [key["liquid"] for key in keys]
        concentrations = [key["concentration"] for key in keys]
        joint_labels = [f"{liquid}__{concentration}" for liquid, concentration in zip(liquids, concentrations)]
        return liquids, concentrations, joint_labels

    def fit(self, split: RepeatedMeasurementSplit) -> OrdinalFoldResult:
        self._validate_config()

        train_keys = [split.measurement_keys[int(idx)] for idx in split.train_indices]
        test_keys = [split.measurement_keys[int(idx)] for idx in split.test_indices]
        train_liquids, train_concentrations, train_joint_labels = self._labels_from_keys(train_keys)
        test_liquids, test_concentrations, test_joint_labels = self._labels_from_keys(test_keys)

        for label in split.idx_to_label.values():
            _split_joint_label(label)

        liquid_labels = sorted(set(train_liquids))
        liquid_label_to_idx = {label: idx for idx, label in enumerate(liquid_labels)}
        idx_to_liquid_label = {idx: label for label, idx in liquid_label_to_idx.items()}

        concentration_labels = sorted(set(train_concentrations), key=_concentration_sort_key)
        concentration_label_to_idx = {label: idx for idx, label in enumerate(concentration_labels)}
        idx_to_concentration_label = {idx: label for label, idx in concentration_label_to_idx.items()}

        y_train_liquid = np.asarray([liquid_label_to_idx[label] for label in train_liquids], dtype=np.int64)
        y_test_liquid = np.asarray([liquid_label_to_idx[label] for label in test_liquids], dtype=np.int64)
        y_train_concentration = np.asarray([concentration_label_to_idx[label] for label in train_concentrations], dtype=np.int64)
        y_test_concentration = np.asarray([concentration_label_to_idx[label] for label in test_concentrations], dtype=np.int64)

        ordinal_train_targets = _ordinal_targets(y_train_concentration, num_classes=len(concentration_labels))

        train_loader = DataLoader(
            SignalDataset(
                split.x_train,
                y_train_liquid,
                y_train_concentration,
                ordinal_train_targets,
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        model = OrdinalAdamWellcomeCNN1D(
            in_channels=int(split.x_train.shape[1]),
            num_liquids=len(liquid_labels),
            num_concentrations=len(concentration_labels),
        ).to(self.device)
        liquid_criterion = nn.CrossEntropyLoss(label_smoothing=self.config.liquid_label_smoothing)
        ordinal_criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._create_lr_scheduler(optimizer)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_liquid_acc": [],
            "train_concentration_acc": [],
            "stop_measurement_joint_acc": [],
            "stop_sample_joint_acc": [],
        }
        stop_metric = self.config.early_stopping_metric.lower()
        best_state = None
        best_score = -np.inf
        best_epoch = -1
        stale_epochs = 0

        joint_label_to_idx = split.label_to_idx
        idx_to_joint_label = split.idx_to_label
        y_measurement_true = np.asarray([joint_label_to_idx[label] for label in test_joint_labels], dtype=np.int64)

        for epoch_idx in range(self.config.epochs):
            train_loss, train_liquid_acc, train_concentration_acc = self._run_train_epoch(
                model,
                train_loader,
                liquid_criterion,
                ordinal_criterion,
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
                y_test_liquid=y_test_liquid,
                y_test_concentration=y_test_concentration,
                test_joint_labels=test_joint_labels,
                liquid_label_to_idx=liquid_label_to_idx,
                idx_to_liquid_label=idx_to_liquid_label,
                concentration_label_to_idx=concentration_label_to_idx,
                idx_to_concentration_label=idx_to_concentration_label,
                joint_label_to_idx=joint_label_to_idx,
            )

            history["train_loss"].append(train_loss)
            history["train_liquid_acc"].append(train_liquid_acc)
            history["train_concentration_acc"].append(train_concentration_acc)
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
                    f"train_liquid_acc={train_liquid_acc:.4f} "
                    f"train_conc_acc={train_concentration_acc:.4f} "
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
            y_test_liquid=y_test_liquid,
            y_test_concentration=y_test_concentration,
            test_joint_labels=test_joint_labels,
            liquid_label_to_idx=liquid_label_to_idx,
            idx_to_liquid_label=idx_to_liquid_label,
            concentration_label_to_idx=concentration_label_to_idx,
            idx_to_concentration_label=idx_to_concentration_label,
            joint_label_to_idx=joint_label_to_idx,
        )
        sample_confusion_matrix = compute_confusion_matrix(
            y_sample_true,
            y_sample_pred,
            num_classes=len(joint_label_to_idx),
        )

        return OrdinalFoldResult(
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
            liquid_label_to_idx=liquid_label_to_idx,
            idx_to_liquid_label=idx_to_liquid_label,
            concentration_label_to_idx=concentration_label_to_idx,
            idx_to_concentration_label=idx_to_concentration_label,
        )

    def _evaluate_predictions(
        self,
        *,
        model: nn.Module,
        x_test: np.ndarray,
        test_group_ids: np.ndarray,
        y_test_liquid: np.ndarray,
        y_test_concentration: np.ndarray,
        test_joint_labels: List[str],
        liquid_label_to_idx: Dict[str, int],
        idx_to_liquid_label: Dict[int, str],
        concentration_label_to_idx: Dict[str, int],
        idx_to_concentration_label: Dict[int, str],
        joint_label_to_idx: Dict[str, int],
    ) -> tuple[float, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        liquid_probabilities, concentration_logits = self._predict_outputs(model, x_test)
        liquid_pred_idx = np.argmax(liquid_probabilities, axis=1)
        concentration_pred_idx = self._decode_concentration(concentration_logits)

        liquid_pred_labels = [idx_to_liquid_label[int(idx)] for idx in liquid_pred_idx]
        concentration_pred_labels = [idx_to_concentration_label[int(idx)] for idx in concentration_pred_idx]
        y_measurement_pred = np.asarray(
            [joint_label_to_idx[f"{liquid}__{concentration}"] for liquid, concentration in zip(liquid_pred_labels, concentration_pred_labels)],
            dtype=np.int64,
        )
        y_measurement_true = np.asarray([joint_label_to_idx[label] for label in test_joint_labels], dtype=np.int64)

        measurement_joint_acc = float((y_measurement_true == y_measurement_pred).mean())
        measurement_liquid_acc = float((liquid_pred_idx == y_test_liquid).mean())
        measurement_concentration_acc = float((concentration_pred_idx == y_test_concentration).mean())

        sample_true: List[int] = []
        sample_pred: List[int] = []
        sample_liquid_true: List[int] = []
        sample_liquid_pred: List[int] = []
        sample_concentration_true: List[int] = []
        sample_concentration_pred: List[int] = []

        for group_id in sorted(set(int(value) for value in test_group_ids)):
            mask = test_group_ids == group_id
            group_true_labels = [test_joint_labels[int(idx)] for idx in np.flatnonzero(mask)]
            if len(set(group_true_labels)) != 1:
                raise ValueError(f"Test group {group_id} contains multiple joint labels")

            mean_liquid_probability = liquid_probabilities[mask].mean(axis=0)
            liquid_idx = int(np.argmax(mean_liquid_probability))
            mean_concentration_logit = concentration_logits[mask].mean(axis=0, keepdims=True)
            concentration_idx = int(self._decode_concentration(mean_concentration_logit)[0])

            true_liquid, true_concentration = _split_joint_label(group_true_labels[0])
            sample_true.append(joint_label_to_idx[group_true_labels[0]])
            sample_pred.append(
                joint_label_to_idx[f"{idx_to_liquid_label[liquid_idx]}__{idx_to_concentration_label[concentration_idx]}"]
            )
            sample_liquid_true.append(liquid_label_to_idx[true_liquid])
            sample_liquid_pred.append(liquid_idx)
            sample_concentration_true.append(concentration_label_to_idx[true_concentration])
            sample_concentration_pred.append(concentration_idx)

        y_sample_true = np.asarray(sample_true, dtype=np.int64)
        y_sample_pred = np.asarray(sample_pred, dtype=np.int64)
        sample_joint_acc = float((y_sample_true == y_sample_pred).mean())
        sample_liquid_acc = float(
            (np.asarray(sample_liquid_true, dtype=np.int64) == np.asarray(sample_liquid_pred, dtype=np.int64)).mean()
        )
        sample_concentration_acc = float(
            (
                np.asarray(sample_concentration_true, dtype=np.int64)
                == np.asarray(sample_concentration_pred, dtype=np.int64)
            ).mean()
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
