from __future__ import annotations

from pathlib import Path

import numpy as np

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.config import (
    LiteRTTorchConfig,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.litert_export import (
    run_tflite_model,
    variant_name_for_recipe,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.metrics import (
    compute_logit_parity,
    softmax_np,
)


def build_litert_validation_placeholders(
    tflite_variant_paths: dict[str, Path],
) -> dict[str, dict[str, object]]:
    """Build metadata placeholders when saved LiteRT/TFLite validation is skipped."""
    return {
        variant_name: {
            "path": variant_path,
            "logits": None,
            "prob": None,
            "pred_idx": None,
            "accuracy": None,
            "accuracy_diff_vs_pytorch": None,
            "parity": None,
            "interpreter_metadata": None,
        }
        for variant_name, variant_path in tflite_variant_paths.items()
    }


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
