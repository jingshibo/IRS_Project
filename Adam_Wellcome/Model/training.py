from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
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
    weight_decay: float = 1e-5
    label_smoothing: float = 0.05
    patience: int = 15
    device: Optional[str] = None
    num_workers: int = 0
    verbose: bool = True
    # Common regular scheduler settings
    use_lr_scheduler: bool = True  # Set False to disable all regular LR scheduling.
    lr_scheduler_name: str = "plateau"  # Choose plateau, cosine, cosine_restarts, or step.
    # plateau only
    scheduler_factor: float = 0.5  # Multiply LR by this factor after a plateau.
    scheduler_patience: int = 50  # Number of bad epochs before reducing LR.
    scheduler_min_lr: float = 1e-6  # Lowest LR allowed after repeated reductions.
    # step only
    scheduler_step_size: int = 50  # Reduce LR every N epochs.
    scheduler_gamma: float = 0.5  # LR multiplier applied at each step.
    # cosine only: The LR decays smoothly from the initial LR down toward a minimum LR over one cycle.
    # if total epochs are larger than T_max, the LR keeps following the cosine curve to increase again.
    scheduler_t_max: Optional[int] = None  # Length of one cosine decay cycle in epochs.
    # cosine and cosine_restarts
    scheduler_eta_min: float = 1e-5  #  the minimum LR reached at the bottom of each cycle
    # cosine_restarts only:  This also uses cosine decay, but instead of decaying once, it periodically restarts back to a higher LR.
    scheduler_restart_t0: int = 300  # the number of epochs in the first cosine cycle before the first restart
    scheduler_restart_t_mult: int = 2  # how much to multiply the cycle length after each restart
    final_model_method: str = "best"
    top_k_checkpoints: int = 3
    best_window_radius: int = 2
    # SWA only
    swa_start_epoch: Optional[int] = None  # Epoch where SWA averaging begins; None uses a late auto start.
    swa_lr: Optional[float] = None  # Target LR during the SWA phase; None reuses the base LR.
    swa_anneal_epochs: int = 10  # Number of epochs used to anneal into the SWA LR.
    swa_anneal_strategy: str = "cos"  # Annealing shape for the SWA transition, cos or linear.


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

    def _validate_config(self) -> None:
        final_model_method = self.config.final_model_method.lower()
        scheduler_name = self.config.lr_scheduler_name.lower()
        if final_model_method not in {"best", "top_k_average", "best_window_average", "swa"}:
            raise ValueError("final_model_method must be 'best', 'top_k_average', 'best_window_average', or 'swa'")
        if self.config.epochs <= 0:
            raise ValueError("epochs must be >= 1")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if self.config.patience <= 0:
            raise ValueError("patience must be >= 1")
        if self.config.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.config.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not 0.0 <= self.config.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in the range [0, 1)")
        if self.config.use_lr_scheduler:
            if scheduler_name not in {"plateau", "cosine", "cosine_restarts", "step"}:
                raise ValueError("lr_scheduler_name must be 'plateau', 'cosine', 'cosine_restarts', or 'step'")
            if self.config.scheduler_min_lr < 0:
                raise ValueError("scheduler_min_lr must be >= 0")
            if scheduler_name == "plateau":
                if not 0.0 < self.config.scheduler_factor < 1.0:
                    raise ValueError("scheduler_factor must be in the range (0, 1) for plateau scheduler")
                if self.config.scheduler_patience < 0:
                    raise ValueError("scheduler_patience must be >= 0 for plateau scheduler")
            if scheduler_name == "step":
                if self.config.scheduler_step_size <= 0:
                    raise ValueError("scheduler_step_size must be >= 1 for step scheduler")
                if not 0.0 < self.config.scheduler_gamma < 1.0:
                    raise ValueError("scheduler_gamma must be in the range (0, 1) for step scheduler")
            if scheduler_name == "cosine":
                scheduler_t_max = self.config.scheduler_t_max if self.config.scheduler_t_max is not None else self.config.epochs
                if scheduler_t_max <= 0:
                    raise ValueError("scheduler_t_max must be >= 1 for cosine scheduler")
                if self.config.scheduler_eta_min < 0:
                    raise ValueError("scheduler_eta_min must be >= 0 for cosine scheduler")
            if scheduler_name == "cosine_restarts":
                if self.config.scheduler_restart_t0 <= 0:
                    raise ValueError("scheduler_restart_t0 must be >= 1 for cosine_restarts scheduler")
                if self.config.scheduler_restart_t_mult < 1:
                    raise ValueError("scheduler_restart_t_mult must be >= 1 for cosine_restarts scheduler")
                if self.config.scheduler_eta_min < 0:
                    raise ValueError("scheduler_eta_min must be >= 0 for cosine_restarts scheduler")
        if final_model_method == "top_k_average" and self.config.top_k_checkpoints <= 0:
            raise ValueError("top_k_checkpoints must be >= 1 when using top_k_average")
        if final_model_method == "best_window_average" and self.config.best_window_radius < 0:
            raise ValueError("best_window_radius must be >= 0 when using best_window_average")
        if final_model_method == "swa":
            if self.config.swa_start_epoch is not None and not 0 <= self.config.swa_start_epoch < self.config.epochs:
                raise ValueError("swa_start_epoch must be >= 0 and < epochs")
            if self.config.swa_lr is not None and self.config.swa_lr <= 0:
                raise ValueError("swa_lr must be > 0")
            if self.config.swa_anneal_epochs <= 0:
                raise ValueError("swa_anneal_epochs must be >= 1")
            if self.config.swa_anneal_strategy not in {"cos", "linear"}:
                raise ValueError("swa_anneal_strategy must be 'cos' or 'linear'")

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
        if not y_true:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(y_true), np.concatenate(y_pred)

    @staticmethod
    def _copy_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: tensor.detach().cpu().clone() for key, tensor in state_dict.items()}

    @staticmethod
    def _average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not state_dicts:
            raise ValueError("state_dicts must contain at least one checkpoint")

        averaged_state = copy.deepcopy(state_dicts[0])
        for key, tensor in averaged_state.items():
            if torch.is_floating_point(tensor):
                stacked = torch.stack([state[key].detach().to(dtype=tensor.dtype) for state in state_dicts], dim=0)
                averaged_state[key] = stacked.mean(dim=0)
            else:
                averaged_state[key] = state_dicts[0][key]
        return averaged_state

    @staticmethod
    def _update_top_checkpoints(
        checkpoints: List[tuple[float, int, Dict[str, torch.Tensor]]],
        score: float,
        epoch_idx: int,
        state_dict: Dict[str, torch.Tensor],
        top_k: int,
    ) -> None:
        checkpoints.append((score, epoch_idx, AdamWellcomeTrainer._copy_state_dict_to_cpu(state_dict)))
        checkpoints.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        del checkpoints[top_k:]

    @staticmethod
    def _copy_recent_window(
        recent_checkpoints: deque[tuple[int, Dict[str, torch.Tensor]]],
        start_epoch: int,
    ) -> List[tuple[int, Dict[str, torch.Tensor]]]:
        return [(epoch_idx, copy.deepcopy(state)) for epoch_idx, state in recent_checkpoints if epoch_idx >= start_epoch]

    def _resolve_swa_start_epoch(self) -> int:
        if self.config.swa_start_epoch is not None:
            if self.config.swa_start_epoch < 0:
                raise ValueError("swa_start_epoch must be >= 0")
            return self.config.swa_start_epoch
        if self.config.epochs <= 1:
            return 0
        return max(int(self.config.epochs * 0.75), 1)

    def _create_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau]:
        if not self.config.use_lr_scheduler:
            return None

        scheduler_name = self.config.lr_scheduler_name.lower()
        if scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr,
            )
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.scheduler_t_max if self.config.scheduler_t_max is not None else self.config.epochs,
                eta_min=self.config.scheduler_eta_min,
            )
        if scheduler_name == "cosine_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.scheduler_restart_t0,
                T_mult=self.config.scheduler_restart_t_mult,
                eta_min=self.config.scheduler_eta_min,
            )
        if scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        raise ValueError("Unsupported lr_scheduler_name")

    @staticmethod
    def _step_lr_scheduler(
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau],
        val_loss: float,
    ) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

    def _select_final_state(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        best_state: Optional[Dict[str, torch.Tensor]],
        top_checkpoints: List[tuple[float, int, Dict[str, torch.Tensor]]],
        best_window_states: List[tuple[int, Dict[str, torch.Tensor]]],
        swa_model: Optional[AveragedModel] = None,
        swa_updates: int = 0,
    ) -> Dict[str, torch.Tensor]:
        method = self.config.final_model_method.lower()
        if best_state is None:
            raise ValueError("No checkpoint was collected during training")
        if method == "best":
            return best_state
        if method == "top_k_average":
            top_k = min(self.config.top_k_checkpoints, len(top_checkpoints))
            if top_k <= 0:
                raise ValueError("top_k_checkpoints must be at least 1 when using top_k_average")
            averaged_state = self._average_state_dicts([state for _, _, state in top_checkpoints[:top_k]])
            model.load_state_dict(averaged_state)
            update_bn(train_loader, model, device=self.device)
            return copy.deepcopy(model.state_dict())
        if method == "best_window_average":
            if not best_window_states:
                raise ValueError("No best-window checkpoints were collected during training")
            averaged_state = self._average_state_dicts([state for _, state in best_window_states])
            model.load_state_dict(averaged_state)
            update_bn(train_loader, model, device=self.device)
            return copy.deepcopy(model.state_dict())
        if method == "swa":
            if swa_model is None or swa_updates <= 0:
                if self.config.verbose:
                    print("SWA selected, but no SWA checkpoints were collected. Falling back to best validation model.")
                return best_state
            update_bn(train_loader, swa_model, device=self.device)
            return copy.deepcopy(swa_model.module.state_dict())
        raise ValueError("final_model_method must be 'best', 'top_k_average', 'best_window_average', or 'swa'")

    def fit(self, splits: SplitBundle) -> TrainResult:
        self._validate_config()
        if len(splits.x_train) == 0:
            raise ValueError("training split is empty")
        if len(splits.x_val) == 0:
            raise ValueError("validation split is empty")

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
        scheduler = self._create_lr_scheduler(optimizer)
        final_model_method = self.config.final_model_method.lower()
        use_swa = final_model_method == "swa"
        use_top_average = final_model_method == "top_k_average"
        use_best_window_average = final_model_method == "best_window_average"
        swa_start_epoch = self._resolve_swa_start_epoch() if use_swa else None
        swa_model = None
        swa_scheduler = None
        swa_updates = 0
        if use_swa:
            swa_scheduler = SWALR(
                optimizer,
                swa_lr=self.config.swa_lr if self.config.swa_lr is not None else self.config.lr,
                anneal_epochs=self.config.swa_anneal_epochs,
                anneal_strategy=self.config.swa_anneal_strategy,
            )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val_acc = -np.inf
        best_epoch = -1
        stale_epochs = 0
        top_checkpoints: List[tuple[float, int, Dict[str, torch.Tensor]]] = []
        recent_checkpoints: deque[tuple[int, Dict[str, torch.Tensor]]] = deque(maxlen=(2 * self.config.best_window_radius) + 1)
        best_window_states: List[tuple[int, Dict[str, torch.Tensor]]] = []
        best_window_pending_future = 0

        for epoch_idx in range(self.config.epochs):
            train_loss, train_acc = self._run_epoch(model, train_loader, criterion, optimizer=optimizer)
            val_loss, val_acc = self._run_epoch(model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if self.config.verbose:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch_idx + 1:03d} | "
                    f"lr={lr:.6g} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

            if use_swa and epoch_idx >= swa_start_epoch:
                if swa_model is None:
                    swa_model = AveragedModel(model).to(self.device)
                swa_model.update_parameters(model)
                swa_updates += 1
                swa_scheduler.step()
            else:
                self._step_lr_scheduler(scheduler, val_loss)

            current_state = self._copy_state_dict_to_cpu(model.state_dict())
            if use_best_window_average:
                recent_checkpoints.append((epoch_idx, current_state))
            if use_top_average:
                self._update_top_checkpoints(
                    top_checkpoints,
                    score=val_acc,
                    epoch_idx=epoch_idx,
                    state_dict=current_state,
                    top_k=max(self.config.top_k_checkpoints, 1),
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch_idx
                best_state = copy.deepcopy(current_state)
                if use_best_window_average:
                    start_epoch = epoch_idx - self.config.best_window_radius
                    best_window_states = self._copy_recent_window(recent_checkpoints, start_epoch)
                    best_window_pending_future = self.config.best_window_radius
                stale_epochs = 0
            else:
                stale_epochs += 1
                if use_best_window_average and best_window_pending_future > 0:
                    best_window_states.append((epoch_idx, copy.deepcopy(current_state)))
                    best_window_pending_future -= 1

            if stale_epochs >= self.config.patience:
                if self.config.verbose:
                    print(f"Early stopping at epoch {epoch_idx + 1}")
                break

        final_state = self._select_final_state(
            model,
            train_loader,
            best_state,
            top_checkpoints,
            best_window_states,
            swa_model=swa_model,
            swa_updates=swa_updates,
        )
        model.load_state_dict(final_state)
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
