from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.config import (
    LiteRTTorchConfig,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.litert_export import (
    run_tflite_model,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.training_utils import (
    compute_logit_parity,
)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Convert logits to softmax probabilities using NumPy."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32, copy=False)


def variant_name_for_recipe(recipe_name: str) -> str:
    """Use readable names for saved comparison arrays and metadata."""
    if recipe_name == "dynamic_wi8_afp32":
        return "half_quant_dynamic_wi8_afp32"
    if recipe_name == "static_wi8_ai8":
        return "full_quant_static_wi8_ai8"
    return recipe_name


def build_stratified_representative_indices(
    y: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Select reproducible, class-balanced calibration sample indices."""
    labels = np.asarray(y)
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("Cannot select representative samples from an empty training set.")
    count = min(max(int(count), 1), n_samples)

    indices = np.arange(n_samples, dtype=np.int64)
    if count == n_samples:
        return indices

    _, representative_indices = train_test_split(
        indices,
        test_size=count,
        stratify=labels,
        random_state=seed,
    )
    return np.asarray(representative_indices, dtype=np.int64)


def validate_tflite_variant(
    variant_name: str,
    model_path: Path,
    x_test_norm: np.ndarray,
    y_test: np.ndarray,
    torch_logits: np.ndarray,
    torch_accuracy: float,
    config: LiteRTTorchConfig,
) -> dict[str, object]:
    """Run one saved TFLite variant and compare its outputs with the PyTorch model."""
    output_logits, interpreter_metadata = run_tflite_model(
        model_path,
        x_test_norm,
        limit=None,
    )
    prob = softmax_np(output_logits)
    pred_idx = np.argmax(prob, axis=1).astype(np.int64)
    accuracy = float(np.mean(pred_idx == y_test))
    accuracy_diff = accuracy - torch_accuracy
    parity = compute_logit_parity(
        torch_logits,
        output_logits,
    )

    print(f"{variant_name} holdout test accuracy: {accuracy:.4f} ({accuracy_diff:+.4f} vs PyTorch)")
    print(f"{variant_name} max abs logit diff: {parity['max_abs_diff']:.8g}")
    print(
        f"{variant_name} input_dtype={interpreter_metadata['input_dtype']} "
        f"input_quantization={interpreter_metadata['input_quantization']} "
        f"output_dtype={interpreter_metadata['output_dtype']} "
        f"output_quantization={interpreter_metadata['output_quantization']}"
    )
    if parity["max_abs_diff"] > config.parity_warning_threshold:
        print(
            f"WARNING: {variant_name} logit difference is above "
            f"{config.parity_warning_threshold:.1e}. Check exported ops and quantization."
        )

    return {
        "path": Path(model_path),
        "logits": output_logits,
        "prob": prob,
        "pred_idx": pred_idx,
        "accuracy": accuracy,
        "accuracy_diff_vs_pytorch": accuracy_diff,
        "parity": parity,
        "interpreter_metadata": interpreter_metadata,
    }
