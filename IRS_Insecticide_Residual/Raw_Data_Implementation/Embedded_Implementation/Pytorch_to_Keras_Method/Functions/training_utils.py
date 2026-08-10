from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Pytorch_to_Keras_Method.Functions.config import (
    PytorchToKerasConfig,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Models import Model_Training
from IRS_Insecticide_Residual.Raw_Data_Implementation.Models.Model_Structure import OneDCNNClassifier
from IRS_Insecticide_Residual.Utility_Functions import Preprocessing


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_initialized_pytorch_model(
    input_length: int,
    in_channels: int,
    num_classes: int,
    device: torch.device,
) -> OneDCNNClassifier:
    """Create OneDCNNClassifier and initialize LazyLinear layers."""
    model = OneDCNNClassifier(in_channels=in_channels, num_classes=num_classes).to(device)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_channels, input_length, dtype=torch.float32, device=device))
    return model


def choose_final_epochs(
    config: PytorchToKerasConfig,
    x_trainval: np.ndarray,
    y_trainval_labels: np.ndarray,
) -> tuple[int, dict[str, object]]:
    """Choose the final epoch count from PyTorch CV, unless a fixed count is supplied."""
    if config.final_epochs is not None:
        if config.final_epochs < 1:
            raise ValueError(f"final_epochs must be >= 1, got {config.final_epochs}")
        return int(config.final_epochs), {
            "source": "user",
            "final_epochs": int(config.final_epochs),
        }

    if not config.run_cv_for_epoch_selection:
        raise ValueError("Set final_epochs or enable run_cv_for_epoch_selection.")

    cv_folds = Preprocessing.build_normalized_cv_folds(
        x_trainval,
        y_trainval_labels,
        n_splits=config.cv_folds_for_epoch_selection,
        random_seed=config.random_seed,
        clip_max_value=config.clip_max_value,
    )
    train_out = Model_Training.train_1d_cnn_cv(
        cv_folds=cv_folds,
        class_order=config.class_order,
        model_name=config.model_name,
        epochs=config.max_cv_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
        patience=config.patience,
        random_shift_max_points=config.random_shift_max_points,
        random_shift_fill_mode=config.random_shift_fill_mode,
        device=config.device,
        num_workers=config.num_workers,
        tensorboard_log_dir=config.tensorboard_log_dir,
        tensorboard_write_every_n=config.tensorboard_write_every_n,
        verbose=True,
        use_lr_scheduler=config.use_lr_scheduler,
        scheduler_factor=config.scheduler_factor,
        scheduler_patience=config.scheduler_patience,
        scheduler_min_lr=config.scheduler_min_lr,
    )

    best_epochs = [int(fold_result.best_epoch) + 1 for fold_result in train_out["fold_results"]]
    final_epochs = max(int(np.round(np.median(np.asarray(best_epochs, dtype=np.int64)))), 1)
    print(f"PyTorch CV best epochs: {best_epochs}")
    print(f"Using final_epochs={final_epochs} from median PyTorch CV best epoch.")
    return final_epochs, {
        "source": "pytorch_cv_median_best_epoch",
        "best_epochs": best_epochs,
        "final_epochs": final_epochs,
        "mean_best_val_acc": float(train_out["mean_best_val_acc"]),
    }


def train_final_pytorch_model(
    x_train_norm: np.ndarray,
    y_train: np.ndarray,
    final_epochs: int,
    config: PytorchToKerasConfig,
) -> tuple[OneDCNNClassifier, dict[str, list[float]], torch.device]:
    """Train one final PyTorch model on all trainval data without an inner validation split."""
    if config.model_name != "shared_backbone_2ch":
        raise ValueError("This PyTorch-to-Keras path currently supports only shared_backbone_2ch.")

    device = resolve_device(config.device)
    model = build_initialized_pytorch_model(
        input_length=x_train_norm.shape[2],
        in_channels=x_train_norm.shape[1],
        num_classes=len(config.class_order),
        device=device,
    )
    train_ds = Model_Training.MicrowaveSignalDataset(
        x_train_norm,
        y_train,
        random_shift_max_points=config.random_shift_max_points,
        random_shift_fill_mode=config.random_shift_fill_mode,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=len(train_ds) > config.batch_size,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scheduler = None
    if config.final_use_train_loss_scheduler and config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )

    history = {"train_loss": [], "train_acc": [], "lr": []}
    model.train()
    for epoch in range(final_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            running_loss += float(loss.item()) * batch_size
            running_correct += int((torch.argmax(logits, dim=1) == y_batch).sum().item())
            running_total += batch_size

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step(train_loss)

        if (epoch + 1) % config.tensorboard_write_every_n == 0 or epoch == final_epochs - 1:
            print(
                f"final epoch={epoch + 1}/{final_epochs} "
                f"lr={optimizer.param_groups[0]['lr']:.6g} "
                f"training_accuracy={train_acc:.4f} training_loss={train_loss:.4f}"
            )

    model.eval()
    return model, history, device


@torch.no_grad()
def predict_logits_prob(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = Model_Training.MicrowaveSignalDataset(x, np.zeros(len(x), dtype=np.int64))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits_all = []
    prob_all = []
    model.eval()
    for x_batch, _ in loader:
        logits = model(x_batch.to(device))
        prob = torch.softmax(logits, dim=1)
        logits_all.append(logits.detach().cpu().numpy())
        prob_all.append(prob.detach().cpu().numpy())
    return (
        np.concatenate(logits_all, axis=0).astype(np.float32, copy=False),
        np.concatenate(prob_all, axis=0).astype(np.float32, copy=False),
    )


def evaluate_pytorch_model(
    model: torch.nn.Module,
    x_test_norm: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    logits, probs = predict_logits_prob(model, x_test_norm, device=device, batch_size=batch_size)
    pred_idx = np.argmax(probs, axis=1).astype(np.int64)
    accuracy = float(np.mean(pred_idx == y_test))
    return accuracy, pred_idx, probs, logits


def compute_logit_parity(torch_logits: np.ndarray, keras_logits: np.ndarray) -> dict[str, float]:
    abs_diff = np.abs(torch_logits - keras_logits)
    return {
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "p95_abs_diff": float(np.percentile(abs_diff, 95)) if abs_diff.size else 0.0,
    }
