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

from .model import ConcentrationOrdinalCNN1D
from .preprocessing import build_stage2_residual_data, transform_with_liquid_mean


class SignalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class OrdinalSignalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, ordinal_targets: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.ordinal_targets = torch.as_tensor(ordinal_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.ordinal_targets[index]


@dataclass
class TwoStageResidualOrdinalConfig:
    epochs: int = 100
    liquid_batch_size: int = 32
    concentration_batch_size: int = 32
    liquid_lr: float = 1e-3
    concentration_lr: float = 1e-3
    liquid_weight_decay: float = 1e-2
    concentration_weight_decay: float = 1e-2
    liquid_label_smoothing: float = 0.05
    residual_mode: str = "residual_only"
    device: Optional[str] = None
    num_workers: int = 0
    verbose: bool = True
    use_lr_scheduler: bool = True
    scheduler_eta_min: float = 1e-5
    use_test_early_stopping: bool = True
    early_stopping_patience: int = 100
    use_liquid_pretraining: bool = True
    freeze_pretrained_features: bool = False


@dataclass
class ClassifierResult:
    model: nn.Module
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    history: Dict[str, List[float]]


@dataclass
class ResidualOrdinalClassifierResult:
    model: nn.Module
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    history: Dict[str, List[float]]
    liquid_mean: np.ndarray


@dataclass
class TwoStageResidualOrdinalFoldResult:
    fold: int
    liquid_classifier: ClassifierResult
    concentration_classifiers: Dict[str, ResidualOrdinalClassifierResult]
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
            "Two-stage residual ordinal training requires joint labels formatted as 'liquid__concentration'."
        )
    return label.split("__", maxsplit=1)


def _concentration_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("-"))


def _encode_labels(labels: List[str], *, sort_mode: str) -> tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    if sort_mode == "concentration":
        unique_labels = sorted(set(labels), key=_concentration_sort_key)
    else:
        unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y = np.asarray([label_to_idx[label] for label in labels], dtype=np.int64)
    return y, label_to_idx, idx_to_label


def _ordinal_targets(y: np.ndarray, num_classes: int) -> np.ndarray:
    thresholds = np.arange(num_classes - 1, dtype=np.int64)
    return (y[:, None] > thresholds[None, :]).astype(np.float32, copy=False)


class TwoStageResidualOrdinalTrainer:
    def __init__(self, config: Optional[TwoStageResidualOrdinalConfig] = None):
        self.config = config or TwoStageResidualOrdinalConfig()
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
        if not 0.0 <= self.config.liquid_label_smoothing < 1.0:
            raise ValueError("liquid_label_smoothing must be in the range [0, 1)")
        if self.config.residual_mode.lower() not in {"residual_only", "concat"}:
            raise ValueError("residual_mode must be 'residual_only' or 'concat'")
        if self.config.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.config.use_liquid_pretraining and self.config.residual_mode.lower() == "concat":
            raise ValueError("use_liquid_pretraining=True is only supported with residual_mode='residual_only'")

    def _initialize_from_liquid_model(
        self,
        concentration_model: ConcentrationOrdinalCNN1D,
        liquid_model: nn.Module,
        example_x: np.ndarray,
    ) -> None:
        concentration_model.eval()
        with torch.no_grad():
            concentration_model(torch.as_tensor(example_x[:1], dtype=torch.float32, device=self.device))

        concentration_model.features.load_state_dict(liquid_model.features.state_dict())

        liquid_projection = liquid_model.classifier[1]
        concentration_projection = concentration_model.head[1]
        if liquid_projection.weight.shape != concentration_projection.weight.shape:
            raise ValueError(
                "Cannot copy liquid projection weights into concentration model: "
                f"{tuple(liquid_projection.weight.shape)} != {tuple(concentration_projection.weight.shape)}"
            )
        concentration_projection.load_state_dict(liquid_projection.state_dict())

        if self.config.freeze_pretrained_features:
            for parameter in concentration_model.features.parameters():
                parameter.requires_grad = False
            for parameter in concentration_projection.parameters():
                parameter.requires_grad = False

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

    def _run_ordinal_train_epoch(
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

        for x_batch, y_batch, ordinal_targets in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            ordinal_targets = ordinal_targets.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, ordinal_targets)
            loss.backward()
            optimizer.step()

            preds = torch.sum(torch.sigmoid(logits) > 0.5, dim=1)
            batch_size = int(x_batch.size(0))
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == y_batch).sum().item())
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

    @torch.no_grad()
    def _predict_ordinal_logits(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        model.eval()
        loader = DataLoader(
            SignalDataset(x, np.zeros(len(x), dtype=np.int64)),
            batch_size=self.config.concentration_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        outputs = []
        for x_batch, _ in loader:
            outputs.append(model(x_batch.to(self.device)).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32, copy=False)

    @staticmethod
    def _decode_ordinal(logits: np.ndarray) -> np.ndarray:
        return np.sum(logits > 0.0, axis=1).astype(np.int64, copy=False)

    def _fit_liquid_classifier(
        self,
        *,
        name: str,
        x_train: np.ndarray,
        train_labels: List[str],
        x_stop: np.ndarray,
        stop_labels: List[str],
    ) -> ClassifierResult:
        y_train, label_to_idx, idx_to_label = _encode_labels(train_labels, sort_mode="default")
        y_stop = np.asarray([label_to_idx[label] for label in stop_labels], dtype=np.int64)

        train_loader = DataLoader(
            SignalDataset(x_train, y_train),
            batch_size=self.config.liquid_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        model = AdamWellcomeCNN1D(
            in_channels=int(x_train.shape[1]),
            num_classes=len(label_to_idx),
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.liquid_label_smoothing)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.liquid_lr,
            weight_decay=self.config.liquid_weight_decay,
        )
        scheduler = self._create_lr_scheduler(optimizer)

        history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "stop_acc": []}
        best_state = None
        best_score = -np.inf
        best_epoch = -1
        stale_epochs = 0

        for epoch_idx in range(self.config.epochs):
            train_loss, train_acc = self._run_train_epoch(model, train_loader, criterion, optimizer)
            if scheduler is not None:
                scheduler.step()
            stop_probs = self._predict_probabilities(model, x_stop)
            stop_pred = np.argmax(stop_probs, axis=1)
            stop_acc = float((stop_pred == y_stop).mean()) if y_stop.size else 0.0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["stop_acc"].append(stop_acc)

            if self.config.verbose:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"{name} | Epoch {epoch_idx + 1:03d} | lr={lr:.6g} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} stop_acc={stop_acc:.4f}"
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
                        print(f"{name} | Early stopping at epoch {epoch_idx + 1}; best epoch was {best_epoch + 1}")
                    break

        if self.config.use_test_early_stopping and best_state is not None:
            model.load_state_dict(best_state)

        return ClassifierResult(model=model, label_to_idx=label_to_idx, idx_to_label=idx_to_label, history=history)

    def _fit_ordinal_classifier(
        self,
        *,
        name: str,
        x_train_raw: np.ndarray,
        train_labels: List[str],
        x_stop_raw: np.ndarray,
        stop_labels: List[str],
        pretrained_liquid_model: Optional[nn.Module] = None,
    ) -> ResidualOrdinalClassifierResult:
        y_train, label_to_idx, idx_to_label = _encode_labels(train_labels, sort_mode="concentration")
        y_stop = np.asarray([label_to_idx[label] for label in stop_labels], dtype=np.int64)
        ordinal_train_targets = _ordinal_targets(y_train, num_classes=len(label_to_idx))

        residual_data = build_stage2_residual_data(
            x_train=x_train_raw,
            x_test=x_stop_raw,
            residual_mode=self.config.residual_mode,
        )

        train_loader = DataLoader(
            OrdinalSignalDataset(residual_data.x_train, y_train, ordinal_train_targets),
            batch_size=self.config.concentration_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        model = ConcentrationOrdinalCNN1D(
            in_channels=int(residual_data.x_train.shape[1]),
            num_concentrations=len(label_to_idx),
        ).to(self.device)
        if self.config.use_liquid_pretraining and pretrained_liquid_model is not None:
            self._initialize_from_liquid_model(
                concentration_model=model,
                liquid_model=pretrained_liquid_model,
                example_x=residual_data.x_train,
            )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=self.config.concentration_lr,
            weight_decay=self.config.concentration_weight_decay,
        )
        scheduler = self._create_lr_scheduler(optimizer)

        history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "stop_acc": []}
        best_state = None
        best_score = -np.inf
        best_epoch = -1
        stale_epochs = 0

        for epoch_idx in range(self.config.epochs):
            train_loss, train_acc = self._run_ordinal_train_epoch(model, train_loader, criterion, optimizer)
            if scheduler is not None:
                scheduler.step()
            stop_logits = self._predict_ordinal_logits(model, residual_data.x_test)
            stop_pred = self._decode_ordinal(stop_logits)
            stop_acc = float((stop_pred == y_stop).mean()) if y_stop.size else 0.0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["stop_acc"].append(stop_acc)

            if self.config.verbose:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"{name} | Epoch {epoch_idx + 1:03d} | lr={lr:.6g} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} stop_acc={stop_acc:.4f}"
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
                        print(f"{name} | Early stopping at epoch {epoch_idx + 1}; best epoch was {best_epoch + 1}")
                    break

        if self.config.use_test_early_stopping and best_state is not None:
            model.load_state_dict(best_state)

        return ResidualOrdinalClassifierResult(
            model=model,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            history=history,
            liquid_mean=residual_data.liquid_mean,
        )

    @staticmethod
    def _labels_from_keys(keys: List[dict]) -> tuple[List[str], List[str], List[str]]:
        liquids = [key["liquid"] for key in keys]
        concentrations = [key["concentration"] for key in keys]
        joint_labels = [f"{liquid}__{concentration}" for liquid, concentration in zip(liquids, concentrations)]
        return liquids, concentrations, joint_labels

    def fit(self, split: RepeatedMeasurementSplit) -> TwoStageResidualOrdinalFoldResult:
        self._validate_config()

        train_keys = [split.measurement_keys[int(idx)] for idx in split.train_indices]
        test_keys = [split.measurement_keys[int(idx)] for idx in split.test_indices]
        train_liquids, train_concentrations, _ = self._labels_from_keys(train_keys)
        test_liquids, test_concentrations, test_joint_labels = self._labels_from_keys(test_keys)

        for label in split.idx_to_label.values():
            _split_joint_label(label)

        liquid_classifier = self._fit_liquid_classifier(
            name=f"fold {split.fold + 1} liquid",
            x_train=split.x_train,
            train_labels=train_liquids,
            x_stop=split.x_test,
            stop_labels=test_liquids,
        )

        concentration_classifiers: Dict[str, ResidualOrdinalClassifierResult] = {}
        for liquid in sorted(set(train_liquids)):
            train_mask = np.asarray([value == liquid for value in train_liquids], dtype=bool)
            test_mask = np.asarray([value == liquid for value in test_liquids], dtype=bool)
            if not np.any(train_mask) or not np.any(test_mask):
                raise ValueError(f"Missing train or test data for liquid '{liquid}'")
            concentration_classifiers[liquid] = self._fit_ordinal_classifier(
                name=f"fold {split.fold + 1} {liquid} concentration",
                x_train_raw=split.x_train[train_mask],
                train_labels=[label for label, keep in zip(train_concentrations, train_mask) if keep],
                x_stop_raw=split.x_test[test_mask],
                stop_labels=[label for label, keep in zip(test_concentrations, test_mask) if keep],
                pretrained_liquid_model=liquid_classifier.model,
            )

        joint_label_to_idx = split.label_to_idx
        idx_to_joint_label = split.idx_to_label
        y_measurement_true = np.asarray([joint_label_to_idx[label] for label in test_joint_labels], dtype=np.int64)

        liquid_probabilities = self._predict_probabilities(liquid_classifier.model, split.x_test)
        liquid_pred_idx = np.argmax(liquid_probabilities, axis=1)
        liquid_pred_labels = [liquid_classifier.idx_to_label[int(idx)] for idx in liquid_pred_idx]

        measurement_joint_predictions: List[int] = []
        measurement_concentration_true: List[str] = []
        measurement_concentration_pred: List[str] = []
        for row_idx, liquid in enumerate(liquid_pred_labels):
            concentration_result = concentration_classifiers[liquid]
            x_stage2 = transform_with_liquid_mean(
                x=split.x_test[row_idx:row_idx + 1],
                liquid_mean=concentration_result.liquid_mean,
                residual_mode=self.config.residual_mode,
            )
            concentration_logits = self._predict_ordinal_logits(concentration_result.model, x_stage2)
            concentration_idx = int(self._decode_ordinal(concentration_logits)[0])
            concentration = concentration_result.idx_to_label[concentration_idx]
            measurement_joint_predictions.append(joint_label_to_idx[f"{liquid}__{concentration}"])
            measurement_concentration_pred.append(concentration)

        for label in test_joint_labels:
            _, concentration = _split_joint_label(label)
            measurement_concentration_true.append(concentration)

        y_measurement_pred = np.asarray(measurement_joint_predictions, dtype=np.int64)
        measurement_joint_acc = float((y_measurement_true == y_measurement_pred).mean())
        measurement_liquid_acc = float(np.mean([true == pred for true, pred in zip(test_liquids, liquid_pred_labels)]))
        measurement_route_mask = np.asarray(
            [true == pred for true, pred in zip(test_liquids, liquid_pred_labels)],
            dtype=bool,
        )
        measurement_concentration_correct = np.asarray(
            [true == pred for true, pred in zip(measurement_concentration_true, measurement_concentration_pred)],
            dtype=bool,
        )
        measurement_concentration_acc = (
            float(measurement_concentration_correct[measurement_route_mask].mean())
            if np.any(measurement_route_mask)
            else 0.0
        )

        sample_true: List[int] = []
        sample_pred: List[int] = []
        sample_liquid_true: List[str] = []
        sample_liquid_pred: List[str] = []
        sample_concentration_true: List[str] = []
        sample_concentration_pred: List[str] = []
        for group_id in sorted(set(int(value) for value in split.test_group_ids)):
            mask = split.test_group_ids == group_id
            group_true_labels = [test_joint_labels[int(idx)] for idx in np.flatnonzero(mask)]
            if len(set(group_true_labels)) != 1:
                raise ValueError(f"Test group {group_id} contains multiple joint labels")

            mean_liquid_probability = liquid_probabilities[mask].mean(axis=0)
            liquid = liquid_classifier.idx_to_label[int(np.argmax(mean_liquid_probability))]
            concentration_result = concentration_classifiers[liquid]
            x_stage2 = transform_with_liquid_mean(
                x=split.x_test[mask],
                liquid_mean=concentration_result.liquid_mean,
                residual_mode=self.config.residual_mode,
            )
            mean_concentration_logit = self._predict_ordinal_logits(concentration_result.model, x_stage2).mean(axis=0, keepdims=True)
            concentration_idx = int(self._decode_ordinal(mean_concentration_logit)[0])
            concentration = concentration_result.idx_to_label[concentration_idx]

            true_liquid, true_concentration = _split_joint_label(group_true_labels[0])
            sample_true.append(joint_label_to_idx[group_true_labels[0]])
            sample_pred.append(joint_label_to_idx[f"{liquid}__{concentration}"])
            sample_liquid_true.append(true_liquid)
            sample_liquid_pred.append(liquid)
            sample_concentration_true.append(true_concentration)
            sample_concentration_pred.append(concentration)

        y_sample_true = np.asarray(sample_true, dtype=np.int64)
        y_sample_pred = np.asarray(sample_pred, dtype=np.int64)
        sample_joint_acc = float((y_sample_true == y_sample_pred).mean())
        sample_liquid_acc = float(np.mean([true == pred for true, pred in zip(sample_liquid_true, sample_liquid_pred)]))
        sample_route_mask = np.asarray(
            [true == pred for true, pred in zip(sample_liquid_true, sample_liquid_pred)],
            dtype=bool,
        )
        sample_concentration_correct = np.asarray(
            [true == pred for true, pred in zip(sample_concentration_true, sample_concentration_pred)],
            dtype=bool,
        )
        sample_concentration_acc = (
            float(sample_concentration_correct[sample_route_mask].mean())
            if np.any(sample_route_mask)
            else 0.0
        )

        num_joint_classes = len(joint_label_to_idx)
        return TwoStageResidualOrdinalFoldResult(
            fold=split.fold,
            liquid_classifier=liquid_classifier,
            concentration_classifiers=concentration_classifiers,
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
