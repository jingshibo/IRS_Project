from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _scaler_arrays(scalers: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    means = []
    scales = []
    for scaler in scalers:
        means.append(np.asarray(scaler.mean_, dtype=np.float32))
        scales.append(np.asarray(scaler.scale_, dtype=np.float32))
    return np.stack(means, axis=0), np.stack(scales, axis=0)


def save_deployment_artifacts(
    train_out: Mapping[str, Any],
    cv_folds: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    fold_index: int = 0,
    representative_count: int = 128,
    selected_value_types: Sequence[str] = ("original", "first_diff_filtered", "second_diff_filtered"),
    signal_segments: Sequence[tuple[int, int]] = ((0, 1000), (1800, 3500)),
    downsample_step: int = 5,
) -> dict[str, Path]:
    """Save the files needed to transfer and quantize one trained CV fold.

    Call this after `train_out = Model_Training.train_1d_cnn_cv(...)`.
    Representative samples are saved after Python preprocessing and scaling,
    with PyTorch layout [N, C, L].
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_result = train_out["fold_results"][fold_index]
    fold_data = cv_folds[fold_index]
    model = fold_result.model.cpu().eval()

    weights_path = output_dir / f"shared_backbone_fold{fold_index}.pth"
    torch.save(model.state_dict(), weights_path)

    x_train = np.asarray(fold_data["X_train"], dtype=np.float32)
    representative = x_train[: min(representative_count, len(x_train))]
    representative_path = output_dir / f"representative_fold{fold_index}.npy"
    np.save(representative_path, representative)

    mean, scale = _scaler_arrays(fold_data["scalers"])
    scaler_path = output_dir / f"scalers_fold{fold_index}.npz"
    np.savez(scaler_path, mean=mean, scale=scale)

    metadata = {
        "model_name": "shared_backbone_2ch",
        "fold_index": fold_index,
        "class_order": [train_out["idx_to_label"][idx] for idx in sorted(train_out["idx_to_label"])],
        "label_to_idx": train_out["label_to_idx"],
        "selected_value_types": list(selected_value_types),
        "signal_segments": [list(segment) for segment in signal_segments],
        "downsample_step": downsample_step,
        "pytorch_input_shape": list(representative.shape[1:]),
        "keras_input_shape": [representative.shape[2], representative.shape[1]],
        "scaler_mean_shape": list(mean.shape),
        "scaler_scale_shape": list(scale.shape),
        "clip_max_value": fold_data.get("clip_max_value"),
    }
    metadata_path = output_dir / f"deployment_metadata_fold{fold_index}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "weights": weights_path,
        "representative": representative_path,
        "scalers": scaler_path,
        "metadata": metadata_path,
    }

