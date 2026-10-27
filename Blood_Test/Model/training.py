from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .dataset import BloodConcentrationSplit
from .model import BloodConcentrationCNN1D


class SignalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass
class TrainerConfig:
    epochs: int = 1000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.0
    dropout: float = 0.3
    device: Optional[str] = None
    num_workers: int = 0
    verbose: bool = True
    log_every: int = 25
    use_lr_scheduler: bool = True
    scheduler_eta_min: float = 1e-5
    early_stopping_metric: str = "sample_accuracy"
    early_stopping_patience: int = 150
    random_seed: int = 42


@dataclass
class FoldResult:
    fold: int
    validation_sample: int | None
    validation_set: int | None
    model: nn.Module
    best_epoch: int
    history: Dict[str, List[float]]
    measurement_accuracy: float
    sample_accuracy: float
    y_measurement_true: np.ndarray
    y_measurement_pred: np.ndarray
    y_sample_true: np.ndarray
    y_sample_pred: np.ndarray
    measurement_confusion_matrix: np.ndarray
    sample_confusion_matrix: np.ndarray
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_value, predicted_value in zip(y_true, y_pred):
        matrix[int(true_value), int(predicted_value)] += 1
    return matrix


def _aggregate_sample_predictions(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample_true: List[int] = []
    sample_predicted: List[int] = []

    for group_id in sorted(set(int(value) for value in group_ids)):
        group_mask = group_ids == group_id
        group_labels = np.unique(y_true[group_mask])
        if len(group_labels) != 1:
            raise ValueError(f"Sample group {group_id} contains multiple labels")

        mean_probability = probabilities[group_mask].mean(axis=0)
        sample_true.append(int(group_labels[0]))
        sample_predicted.append(int(np.argmax(mean_probability)))

    return (
        np.asarray(sample_true, dtype=np.int64),
        np.asarray(sample_predicted, dtype=np.int64),
    )


class BloodConcentrationTrainer:
    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self._validate_config()
        self.device = torch.device(
            self.config.device
            if self.config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _validate_config(self) -> None:
        if self.config.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.config.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.config.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not 0.0 <= self.config.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in the range [0, 1)")
        if not 0.0 <= self.config.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.config.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.config.early_stopping_metric not in {
            "sample_accuracy",
            "measurement_accuracy",
        }:
            raise ValueError(
                "early_stopping_metric must be 'sample_accuracy' or "
                "'measurement_accuracy'"
            )
        if self.config.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")

    def _set_random_seed(self, fold: int) -> torch.Generator:
        seed = int(self.config.random_seed + fold)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _create_scheduler(
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

            batch_size = int(x_batch.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((torch.argmax(logits, dim=1) == y_batch).sum().item())
            total_items += batch_size

        return (
            total_loss / max(total_items, 1),
            total_correct / max(total_items, 1),
        )

    @torch.no_grad()
    def _predict_probabilities(
        self,
        model: nn.Module,
        x: np.ndarray,
    ) -> np.ndarray:
        model.eval()
        loader = DataLoader(
            SignalDataset(x, np.zeros(len(x), dtype=np.int64)),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        probabilities = []
        for x_batch, _ in loader:
            logits = model(x_batch.to(self.device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probabilities).astype(np.float32, copy=False)

    def _evaluate_validation(
        self,
        model: nn.Module,
        split: BloodConcentrationSplit,
        criterion: nn.Module,
    ) -> Dict[str, object]:
        probabilities = self._predict_probabilities(model, split.x_validation)
        measurement_predicted = np.argmax(probabilities, axis=1).astype(
            np.int64,
            copy=False,
        )
        measurement_accuracy = float(
            np.mean(split.y_validation == measurement_predicted)
        )
        sample_true, sample_predicted = _aggregate_sample_predictions(
            probabilities,
            split.y_validation,
            split.validation_group_ids,
        )
        sample_accuracy = float(np.mean(sample_true == sample_predicted))

        with torch.no_grad():
            logits = model(
                torch.as_tensor(
                    split.x_validation,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            validation_loss = float(
                criterion(
                    logits,
                    torch.as_tensor(
                        split.y_validation,
                        dtype=torch.long,
                        device=self.device,
                    ),
                ).item()
            )

        return {
            "loss": validation_loss,
            "measurement_accuracy": measurement_accuracy,
            "sample_accuracy": sample_accuracy,
            "measurement_predicted": measurement_predicted,
            "sample_true": sample_true,
            "sample_predicted": sample_predicted,
        }

    def fit(self, split: BloodConcentrationSplit) -> FoldResult:
        generator = self._set_random_seed(split.fold)
        train_loader = DataLoader(
            SignalDataset(split.x_train, split.y_train),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            generator=generator,
        )

        model = BloodConcentrationCNN1D(
            in_channels=int(split.x_train.shape[1]),
            num_classes=len(split.label_to_idx),
            dropout=self.config.dropout,
        ).to(self.device)
        # Materialize the lazy classifier layer before copying model states.
        model.eval()
        with torch.no_grad():
            model(
                torch.zeros(
                    (1, *split.x_train.shape[1:]),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        model.train()

        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._create_scheduler(optimizer)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_measurement_accuracy": [],
            "validation_loss": [],
            "validation_measurement_accuracy": [],
            "validation_sample_accuracy": [],
            "learning_rate": [],
        }
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_score = -np.inf
        best_epoch = -1
        stale_epochs = 0

        if self.config.verbose:
            validation_label = (
                f"set {split.validation_set}"
                if split.validation_set is not None
                else f"S{split.validation_sample}"
            )
            print(
                f"Fold {split.fold + 1}: training on {self.device}; "
                f"validation={validation_label}"
            )

        for epoch_index in range(self.config.epochs):
            train_loss, train_accuracy = self._run_train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
            validation = self._evaluate_validation(model, split, criterion)
            learning_rate = float(optimizer.param_groups[0]["lr"])

            history["train_loss"].append(train_loss)
            history["train_measurement_accuracy"].append(train_accuracy)
            history["validation_loss"].append(float(validation["loss"]))
            history["validation_measurement_accuracy"].append(
                float(validation["measurement_accuracy"])
            )
            history["validation_sample_accuracy"].append(
                float(validation["sample_accuracy"])
            )
            history["learning_rate"].append(learning_rate)

            score = float(validation[self.config.early_stopping_metric])
            if score > best_score + 1e-12:
                best_score = score
                best_epoch = epoch_index
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if scheduler is not None:
                scheduler.step()

            if self.config.verbose and (
                epoch_index == 0
                or (epoch_index + 1) % self.config.log_every == 0
                or epoch_index + 1 == self.config.epochs
            ):
                print(
                    f"  epoch {epoch_index + 1:4d} | "
                    f"train loss={train_loss:.4f} | "
                    f"train acc={train_accuracy:.4f} | "
                    f"val measurement acc={float(validation['measurement_accuracy']):.4f} | "
                    f"val sample acc={float(validation['sample_accuracy']):.4f}"
                )

            if stale_epochs >= self.config.early_stopping_patience:
                if self.config.verbose:
                    print(
                        f"  early stopping at epoch {epoch_index + 1}; "
                        f"best epoch={best_epoch + 1}, best "
                        f"{self.config.early_stopping_metric}={best_score:.4f}"
                    )
                break

        if best_state is None:
            raise RuntimeError("Training completed without producing a model state")
        model.load_state_dict(best_state)
        final_validation = self._evaluate_validation(model, split, criterion)
        num_classes = len(split.label_to_idx)

        measurement_confusion = compute_confusion_matrix(
            split.y_validation,
            final_validation["measurement_predicted"],
            num_classes=num_classes,
        )
        sample_confusion = compute_confusion_matrix(
            final_validation["sample_true"],
            final_validation["sample_predicted"],
            num_classes=num_classes,
        )

        return FoldResult(
            fold=split.fold,
            validation_sample=split.validation_sample,
            validation_set=split.validation_set,
            model=model,
            best_epoch=best_epoch,
            history=history,
            measurement_accuracy=float(final_validation["measurement_accuracy"]),
            sample_accuracy=float(final_validation["sample_accuracy"]),
            y_measurement_true=split.y_validation.copy(),
            y_measurement_pred=final_validation["measurement_predicted"],
            y_sample_true=final_validation["sample_true"],
            y_sample_pred=final_validation["sample_predicted"],
            measurement_confusion_matrix=measurement_confusion,
            sample_confusion_matrix=sample_confusion,
            label_to_idx=split.label_to_idx,
            idx_to_label=split.idx_to_label,
        )
