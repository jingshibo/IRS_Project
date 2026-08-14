from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.config import (
    LiteRTTorchConfig,
)


def save_final_artifacts(
    pytorch_model: torch.nn.Module,
    config: LiteRTTorchConfig,
    final_epochs: int,
    epoch_selection: dict[str, object],
    label_to_idx: dict[str, int],
    idx_to_label: dict[int, str],
    x_train_norm: np.ndarray,
    x_test_norm: np.ndarray,
    y_test: np.ndarray,
    y_test_labels: Sequence[str],
    torch_pred_idx: np.ndarray,
    torch_prob: np.ndarray,
    torch_logits: np.ndarray,
    representative_samples: np.ndarray,
    representative_indices: np.ndarray,
    scalers: Sequence[StandardScaler],
    removed_zero_sample_indices: Sequence[int],
    torch_test_accuracy: float,
    train_history: dict[str, list[float]],
    float_tflite_path: Optional[Path] = None,
    float_edge_sample_logits: Optional[np.ndarray] = None,
    float_edge_sample_parity: Optional[dict[str, float]] = None,
    tflite_validation_results: Optional[dict[str, dict[str, object]]] = None,
) -> dict[str, Path]:
    """Save all important artifacts needed to audit LiteRT Torch conversion and deploy.
    it saves:
    1. trained PyTorch model
    2. representative calibration samples
    3. scaler parameters
    4. prediction/logit arrays
    5. metadata JSON summary
    6. paths to generated TFLite models
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the final PyTorch checkpoint used as the source model for LiteRT export.
    pytorch_checkpoint_path = output_dir / "shared_backbone_final.pth"
    torch.save(
        {
            "state_dict": pytorch_model.state_dict(),
            "model_name": config.model_name,
            "input_length": int(x_train_norm.shape[2]),
            "in_channels": int(x_train_norm.shape[1]),
            "num_classes": len(label_to_idx),
            "class_order": list(config.class_order),
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "final_epochs": int(final_epochs),
            "train_history": train_history,
            "config": asdict(config),
        },
        pytorch_checkpoint_path,
    )

    # Save the exact representative samples used for calibrated full-int8 quantization.
    representative_path = output_dir / "representative_final.npy"
    np.save(representative_path, representative_samples.astype(np.float32, copy=False))

    # Save the train-set indices of those representative samples for audit/reproducibility.
    representative_indices_path = output_dir / "representative_indices_final.npy"
    np.save(representative_indices_path, np.asarray(representative_indices, dtype=np.int64))

    # Save per-channel StandardScaler parameters needed to reproduce deployment preprocessing.
    mean = np.stack([np.asarray(scaler.mean_, dtype=np.float32) for scaler in scalers], axis=0)
    scale = np.stack([np.asarray(scaler.scale_, dtype=np.float32) for scaler in scalers], axis=0)
    scalers_path = output_dir / "scalers_final.npz"
    np.savez(scalers_path, mean=mean, scale=scale)

    # Save holdout labels, PyTorch outputs, normalized test inputs, and per-variant TFLite outputs.
    predictions_path = output_dir / "test_predictions_final.npz"
    prediction_payload = {
        "y_true_idx": y_test,
        "y_true_label": np.asarray(y_test_labels),
        "torch_pred_idx": torch_pred_idx,
        "torch_pred_label": np.asarray([idx_to_label[int(idx)] for idx in torch_pred_idx]),
        "torch_prob": torch_prob.astype(np.float32, copy=False),
        "torch_logits": torch_logits.astype(np.float32, copy=False),
        "x_test_norm_pytorch_layout": x_test_norm.astype(np.float32, copy=False),
    }
    if float_edge_sample_logits is not None:
        prediction_payload["float_edge_sample_logits"] = float_edge_sample_logits.astype(np.float32, copy=False)
    if tflite_validation_results is not None:
        for variant_name, result in tflite_validation_results.items():
            if result.get("logits") is not None:
                prediction_payload[f"{variant_name}_logits"] = np.asarray(result["logits"], dtype=np.float32)
            if result.get("prob") is not None:
                prediction_payload[f"{variant_name}_prob"] = np.asarray(result["prob"], dtype=np.float32)
            if result.get("pred_idx") is not None:
                pred_idx = np.asarray(result["pred_idx"], dtype=np.int64)
                prediction_payload[f"{variant_name}_pred_idx"] = pred_idx
                prediction_payload[f"{variant_name}_pred_label"] = np.asarray(
                    [idx_to_label[int(idx)] for idx in pred_idx]
                )
    np.savez(predictions_path, **prediction_payload)

    # Build a compact summary for each TFLite-variant model. This is one item in the final metadata file.
    tflite_variant_summary = {}
    if tflite_validation_results is not None:
        for variant_name, result in tflite_validation_results.items():
            tflite_variant_summary[variant_name] = {
                "path": str(result.get("path")) if result.get("path") is not None else None,
                "accuracy": result.get("accuracy"),
                "accuracy_diff_vs_pytorch": result.get("accuracy_diff_vs_pytorch"),
                "logit_parity": result.get("parity"),
                "interpreter_metadata": result.get("interpreter_metadata"),
            }
    comparison_summary = {
        "original_pytorch": {
            "accuracy": torch_test_accuracy,
            "accuracy_diff_vs_pytorch": 0.0,
            "path": str(pytorch_checkpoint_path),
        },
        **tflite_variant_summary,
    }

    # Save human-readable training, preprocessing, conversion, and validation metadata.
    # It helps you know exactly:
    # what model was trained, how the data was preprocessed, how it was exported,
    # what files were produced, and how each TFLite variant performed.
    metadata = {
        "model_name": config.model_name,
        "primary_training_framework": "pytorch",
        "deployment_model_format": "litert_torch_tflite",
        "pytorch_checkpoint_path": str(pytorch_checkpoint_path),
        "float_tflite_path": str(float_tflite_path) if float_tflite_path is not None else None,
        "quantize_recipes": list(config.quantize_recipes),
        "class_order": list(config.class_order),
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "selected_value_types": list(config.selected_value_types),
        "signal_segments": [list(segment) for segment in config.signal_segments],
        "spike_radius": config.spike_radius,
        "spike_transform": config.spike_transform,
        "spike_method": config.spike_method,
        "spike_n_sigmas": config.spike_n_sigmas,
        "spike_k": config.spike_k,
        "spike_min_threshold": config.spike_min_threshold,
        "savgol_window_length": config.savgol_window_length,
        "savgol_polyorder": config.savgol_polyorder,
        "savgol_deriv": config.savgol_deriv,
        "savgol_mode": config.savgol_mode,
        "downsample_step": config.downsample_step,
        "downsample_offset": config.downsample_offset,
        "rolling_window_size": config.rolling_window_size,
        "random_seed": config.random_seed,
        "test_size": config.test_size,
        "final_epochs": int(final_epochs),
        "epoch_selection": epoch_selection,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "label_smoothing": config.label_smoothing,
        "use_lr_scheduler": config.use_lr_scheduler,
        "final_use_train_loss_scheduler": config.final_use_train_loss_scheduler,
        "scheduler_factor": config.scheduler_factor,
        "scheduler_patience": config.scheduler_patience,
        "scheduler_min_lr": config.scheduler_min_lr,
        "random_shift_max_points": config.random_shift_max_points,
        "random_shift_fill_mode": config.random_shift_fill_mode,
        "clip_max_value": config.clip_max_value,
        "torch_test_accuracy": torch_test_accuracy,
        "float_edge_sample_parity": float_edge_sample_parity,
        "tflite_variants": tflite_variant_summary,
        "comparison_summary": comparison_summary,
        "train_history": train_history,
        "removed_zero_sample_indices": list(removed_zero_sample_indices),
        "representative_indices_path": str(representative_indices_path),
        "representative_count": int(len(representative_samples)),
        "pytorch_input_shape": list(x_train_norm.shape[1:]),
        "tflite_input_shape": [1, x_train_norm.shape[1], x_train_norm.shape[2]],
        "scaler_mean_shape": list(mean.shape),
        "scaler_scale_shape": list(scale.shape),
        "config": asdict(config),
    }
    metadata_path = output_dir / "deployment_metadata_final.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Return paths to the files generated by this run so the script can print them.
    artifacts = {
        "pytorch_checkpoint": pytorch_checkpoint_path,
        "representative": representative_path,
        "representative_indices": representative_indices_path,
        "scalers": scalers_path,
        "predictions": predictions_path,
        "metadata": metadata_path,
    }
    if float_tflite_path is not None:
        artifacts["float_tflite_model"] = Path(float_tflite_path)
    if tflite_validation_results is not None:
        for variant_name, result in tflite_validation_results.items():
            if result.get("path") is not None:
                artifacts[f"{variant_name}_model"] = Path(result["path"])
    return artifacts
