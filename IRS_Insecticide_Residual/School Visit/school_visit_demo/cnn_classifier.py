from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from IRS_Insecticide_Residual.Raw_Data_Implementation.Models import Model_Training
from IRS_Insecticide_Residual.Utility_Functions import Preprocessing

from .classifier import (
    DemoClassificationResult,
    build_pca_feature_maps,
    choose_unknown_position,
)


def train_cnn_demo_classifier(
    x_signal: np.ndarray,
    y_all: np.ndarray,
    class_order: Sequence[str],
    random_seed: int = 42,
    test_size: float = 0.2,
    unknown_test_position: int | None = None,
    model_name: str = "shared_backbone_2ch",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.3,
    patience: int = 10,
    n_splits: int = 5,
    random_shift_max_points: int = 5,
    random_shift_fill_mode: str = "wrap",
    verbose: bool = True,
) -> DemoClassificationResult:
    """Train the existing raw-signal CNN as an alternative school-visit classifier."""
    x_signal = np.asarray(x_signal, dtype=np.float32)
    y_all = np.asarray(y_all)
    indices = np.arange(len(y_all))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=y_all,
        random_state=random_seed,
    )

    x_trainval_signal = x_signal[train_indices]
    x_test_signal = x_signal[test_indices]
    y_trainval = y_all[train_indices]
    y_test = y_all[test_indices]

    cv_folds = Preprocessing.build_normalized_cv_folds(
        x_trainval_signal,
        y_trainval,
        n_splits=n_splits,
        random_seed=random_seed,
        clip_max_value=None,
    )

    train_out = Model_Training.train_1d_cnn_cv(
        cv_folds=cv_folds,
        class_order=class_order,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        patience=patience,
        random_shift_max_points=random_shift_max_points,
        random_shift_fill_mode=random_shift_fill_mode,
        tensorboard_log_dir=None,
        verbose=verbose,
    )

    test_prob_by_fold = []
    for fold_data, fold_result in zip(cv_folds, train_out["fold_results"]):
        x_test_fold = np.asarray(x_test_signal, dtype=np.float32).copy()
        for channel_idx, scaler in enumerate(fold_data["scalers"]):
            x_test_fold[:, channel_idx, :] = scaler.transform(
                x_test_fold[:, channel_idx, :]
            ).astype(np.float32, copy=False)

        clip_max_value = fold_data.get("clip_max_value")
        if clip_max_value is not None:
            x_test_fold = np.clip(
                x_test_fold,
                a_min=None,
                a_max=clip_max_value,
            ).astype(np.float32, copy=False)

        test_prob_by_fold.append(
            Model_Training.predict_prob(
                fold_result.model,
                x_test_fold,
                device=train_out["device"],
            )
        )

    best_fold_idx = int(
        np.argmax([fold_result.best_val_acc for fold_result in train_out["fold_results"]])
    )
    best_fold_data = cv_folds[best_fold_idx]
    best_fold_result = train_out["fold_results"][best_fold_idx]
    x_trainval_for_features = _normalize_signal_with_fold_scalers(
        x_trainval_signal,
        best_fold_data,
    )
    x_test_for_features = _normalize_signal_with_fold_scalers(
        x_test_signal,
        best_fold_data,
    )

    x_trainval_input_flat = x_trainval_for_features.reshape(
        x_trainval_for_features.shape[0],
        -1,
    )
    x_test_input_flat = x_test_for_features.reshape(
        x_test_for_features.shape[0],
        -1,
    )
    (
        pre_cnn_x_train_map,
        pre_cnn_x_test_map,
        _,
        pre_cnn_x_train_map_3d,
        pre_cnn_x_test_map_3d,
        _,
    ) = build_pca_feature_maps(
        x_trainval_input_flat,
        x_test_input_flat,
        random_seed=random_seed,
        values_are_scaled=True,
    )

    x_trainval_cnn_features = extract_cnn_penultimate_features(
        best_fold_result.model,
        x_trainval_for_features,
        device=train_out["device"],
    )
    x_test_cnn_features = extract_cnn_penultimate_features(
        best_fold_result.model,
        x_test_for_features,
        device=train_out["device"],
    )

    feature_scaler = StandardScaler()
    x_trainval_features_scaled = feature_scaler.fit_transform(x_trainval_cnn_features)
    x_test_features_scaled = feature_scaler.transform(x_test_cnn_features)
    x_train_map, x_test_map, reducer, x_train_map_3d, x_test_map_3d, reducer_3d = build_pca_feature_maps(
        x_trainval_features_scaled,
        x_test_features_scaled,
        random_seed=random_seed,
        values_are_scaled=True,
    )

    y_prob = np.mean(np.stack(test_prob_by_fold, axis=0), axis=0)
    pred_idx = np.argmax(y_prob, axis=1)
    y_pred = np.asarray([train_out["idx_to_label"][int(idx)] for idx in pred_idx])
    test_accuracy = float(np.mean(y_pred == y_test))
    confusion_count = confusion_matrix(y_test, y_pred, labels=list(class_order))

    unknown_position = choose_unknown_position(
        y_prob=y_prob,
        y_pred=y_pred,
        y_true=y_test,
        requested_position=unknown_test_position,
    )

    return DemoClassificationResult(
        train_indices=train_indices,
        test_indices=test_indices,
        x_train_map=x_train_map,
        x_test_map=x_test_map,
        x_train_map_3d=x_train_map_3d,
        x_test_map_3d=x_test_map_3d,
        y_train=y_trainval,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        class_order=tuple(class_order),
        scaler=feature_scaler,
        reducer=reducer,
        reducer_3d=reducer_3d,
        classifier=train_out,
        method_name=f"CNN ({model_name})",
        input_description="processed multi-channel signal tensor; feature map uses CNN learned features",
        test_accuracy=test_accuracy,
        confusion_count=confusion_count,
        unknown_global_index=int(test_indices[unknown_position]),
        unknown_map_point=x_test_map[unknown_position],
        unknown_map_point_3d=x_test_map_3d[unknown_position],
        unknown_true_label=str(y_test[unknown_position]),
        unknown_pred_label=str(y_pred[unknown_position]),
        unknown_prob=y_prob[unknown_position],
        pre_cnn_x_train_map=pre_cnn_x_train_map,
        pre_cnn_x_test_map=pre_cnn_x_test_map,
        pre_cnn_unknown_map_point=pre_cnn_x_test_map[unknown_position],
        pre_cnn_x_train_map_3d=pre_cnn_x_train_map_3d,
        pre_cnn_x_test_map_3d=pre_cnn_x_test_map_3d,
        pre_cnn_unknown_map_point_3d=pre_cnn_x_test_map_3d[unknown_position],
    )


def _normalize_signal_with_fold_scalers(
    x_signal: np.ndarray,
    fold_data: dict[str, object],
) -> np.ndarray:
    x_fold = np.asarray(x_signal, dtype=np.float32).copy()
    for channel_idx, scaler in enumerate(fold_data["scalers"]):
        x_fold[:, channel_idx, :] = scaler.transform(
            x_fold[:, channel_idx, :]
        ).astype(np.float32, copy=False)

    clip_max_value = fold_data.get("clip_max_value")
    if clip_max_value is not None:
        x_fold = np.clip(x_fold, a_min=None, a_max=clip_max_value).astype(
            np.float32,
            copy=False,
        )
    return x_fold


@torch.no_grad()
def extract_cnn_penultimate_features(
    model: nn.Module,
    x_signal: np.ndarray,
    device: str | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Return the feature vector passed into the final CNN classification layer."""
    target_layer = _find_last_linear_layer(model)
    captured_batches = []

    def capture_input(_module, inputs, _output):
        captured_batches.append(inputs[0].detach().cpu().numpy())

    handle = target_layer.register_forward_hook(capture_input)
    try:
        model = model.to(device)
        model.eval()
        x_signal = np.asarray(x_signal, dtype=np.float32)
        for start in range(0, len(x_signal), batch_size):
            batch = torch.as_tensor(
                x_signal[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            model(batch)
    finally:
        handle.remove()

    if not captured_batches:
        raise RuntimeError("CNN feature extraction did not capture any batches.")
    return np.concatenate(captured_batches, axis=0).astype(np.float32, copy=False)


def _find_last_linear_layer(model: nn.Module) -> nn.Module:
    linear_layers = [
        module
        for module in model.modules()
        if isinstance(module, (nn.Linear, nn.LazyLinear))
    ]
    if not linear_layers:
        raise ValueError(f"Could not find a linear classification layer in {type(model).__name__}.")
    return linear_layers[-1]
