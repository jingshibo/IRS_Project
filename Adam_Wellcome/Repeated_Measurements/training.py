from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Model.model import AdamWellcomeCNN1D

from .dataset import RepeatedMeasurementSplit
from .evaluation import (
    SampleLevelResult,
    aggregate_repeated_measurements,
    compute_confusion_matrix,
)


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
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    device: Optional[str] = None
    num_workers: int = 0
    verbose: bool = True
    use_lr_scheduler: bool = True
    lr_scheduler_name: str = "cosine"
    scheduler_t_max: Optional[int] = None
    scheduler_eta_min: float = 1e-5
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5
    early_stopping_metric: str = "none"
    early_stopping_patience: int = 100


@dataclass
class RepeatedMeasurementTrainResult:
    model: nn.Module
    history: Dict[str, List[float]]
    measurement_test_acc: float
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    y_test_probabilities: np.ndarray
    measurement_confusion_matrix: np.ndarray
    sample_probability_average: SampleLevelResult
    sample_majority_vote: SampleLevelResult
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]


class RepeatedMeasurementTrainer:
    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
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
        if not 0.0 <= self.config.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in the range [0, 1)")
        if self.config.lr_scheduler_name.lower() not in {"cosine", "step"}:
            raise ValueError("lr_scheduler_name must be 'cosine' or 'step'")
        if self.config.early_stopping_metric.lower() not in {
            "none",
            "measurement_test_acc",
            "sample_probability_test_acc",
            "sample_majority_test_acc",
        }:
            raise ValueError(
                "early_stopping_metric must be 'none', 'measurement_test_acc', "
                "'sample_probability_test_acc', or 'sample_majority_test_acc'"
            )
        if self.config.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be >= 1")

    def _create_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        if not self.config.use_lr_scheduler:
            return None

        scheduler_name = self.config.lr_scheduler_name.lower()
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.scheduler_t_max if self.config.scheduler_t_max is not None else self.config.epochs,
                eta_min=self.config.scheduler_eta_min,
            )
        if scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        raise ValueError("Unsupported lr_scheduler_name")

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
    def _predict_probabilities(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        y_true = []
        y_pred = []
        probabilities = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            logits = model(x_batch)
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(np.argmax(batch_probabilities, axis=1).astype(np.int64, copy=False))
            probabilities.append(batch_probabilities)

        if not y_true:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.empty((0, 0), dtype=np.float32),
            )
        return (
            np.concatenate(y_true),
            np.concatenate(y_pred),
            np.concatenate(probabilities).astype(np.float32, copy=False),
        )

    def fit(self, split: RepeatedMeasurementSplit) -> RepeatedMeasurementTrainResult:
        self._validate_config()
        if len(split.x_train) == 0:
            raise ValueError("training split is empty")
        if len(split.x_test) == 0:
            raise ValueError("test split is empty")

        train_loader = DataLoader(
            SignalDataset(split.x_train, split.y_train),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        test_loader = DataLoader(
            SignalDataset(split.x_test, split.y_test),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        model = AdamWellcomeCNN1D(
            in_channels=int(split.x_train.shape[1]),
            num_classes=len(split.label_to_idx),
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._create_lr_scheduler(optimizer)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
        }
        stop_metric = self.config.early_stopping_metric.lower()
        best_state = None
        best_score = -np.inf
        best_epoch = -1
        stale_epochs = 0

        for epoch_idx in range(self.config.epochs):
            train_loss, train_acc = self._run_train_epoch(model, train_loader, criterion, optimizer)
            if scheduler is not None:
                scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if self.config.verbose:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch_idx + 1:03d} | "
                    f"lr={lr:.6g} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                )

            if stop_metric != "none":
                y_stop_true, y_stop_pred, y_stop_probabilities = self._predict_probabilities(model, test_loader)
                measurement_stop_acc = float((y_stop_true == y_stop_pred).mean()) if y_stop_true.size else 0.0
                probability_stop_result = aggregate_repeated_measurements(
                    y_stop_true,
                    y_stop_pred,
                    y_stop_probabilities,
                    split.test_group_ids,
                    num_classes=len(split.label_to_idx),
                    method="probability_average",
                )
                majority_stop_result = aggregate_repeated_measurements(
                    y_stop_true,
                    y_stop_pred,
                    y_stop_probabilities,
                    split.test_group_ids,
                    num_classes=len(split.label_to_idx),
                    method="majority_vote",
                )

                score_map = {
                    "measurement_test_acc": measurement_stop_acc,
                    "sample_probability_test_acc": probability_stop_result.accuracy,
                    "sample_majority_test_acc": majority_stop_result.accuracy,
                }
                current_score = float(score_map[stop_metric])
                history.setdefault(stop_metric, []).append(current_score)

                if self.config.verbose:
                    print(
                        f"           stop_metric={stop_metric} "
                        f"score={current_score:.4f}"
                    )

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
                            f"Early stopping at epoch {epoch_idx + 1} "
                            f"using {stop_metric}; best epoch was {best_epoch + 1}"
                        )
                    break

        if stop_metric != "none" and best_state is not None:
            model.load_state_dict(best_state)

        y_test_true, y_test_pred, y_test_probabilities = self._predict_probabilities(model, test_loader)
        measurement_test_acc = float((y_test_true == y_test_pred).mean()) if y_test_true.size else 0.0
        measurement_confusion_matrix = compute_confusion_matrix(
            y_test_true,
            y_test_pred,
            num_classes=len(split.label_to_idx),
        )
        sample_probability_average = aggregate_repeated_measurements(
            y_test_true,
            y_test_pred,
            y_test_probabilities,
            split.test_group_ids,
            num_classes=len(split.label_to_idx),
            method="probability_average",
        )
        sample_majority_vote = aggregate_repeated_measurements(
            y_test_true,
            y_test_pred,
            y_test_probabilities,
            split.test_group_ids,
            num_classes=len(split.label_to_idx),
            method="majority_vote",
        )

        return RepeatedMeasurementTrainResult(
            model=model,
            history=history,
            measurement_test_acc=measurement_test_acc,
            y_test_true=y_test_true,
            y_test_pred=y_test_pred,
            y_test_probabilities=y_test_probabilities,
            measurement_confusion_matrix=measurement_confusion_matrix,
            sample_probability_average=sample_probability_average,
            sample_majority_vote=sample_majority_vote,
            label_to_idx=split.label_to_idx,
            idx_to_label=split.idx_to_label,
        )
