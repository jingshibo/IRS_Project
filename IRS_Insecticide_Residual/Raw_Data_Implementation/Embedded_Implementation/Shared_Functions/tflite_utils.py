from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import tensorflow as tf

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.keras_model import (
    torch_to_keras_input,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.metrics import (
    compute_logit_parity,
    softmax_np,
)

InputLayout = Literal["keras", "pytorch"]


def _quantize_input(input_value: np.ndarray, input_detail: dict) -> np.ndarray:
    input_dtype = input_detail["dtype"]
    if input_dtype == np.float32:
        return input_value.astype(np.float32, copy=False)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if scale == 0:
        raise ValueError(f"Cannot quantize interpreter input with scale={scale}.")
    quantized = np.round(input_value / scale + zero_point)
    info = np.iinfo(input_dtype)
    return np.clip(quantized, info.min, info.max).astype(input_dtype)


def _dequantize_output(output_value: np.ndarray, output_detail: dict) -> np.ndarray:
    output_dtype = output_detail["dtype"]
    if output_dtype == np.float32:
        return output_value.astype(np.float32, copy=False)

    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if scale == 0:
        return output_value.astype(np.float32)
    return (output_value.astype(np.float32) - float(zero_point)) * float(scale)


def run_tflite_model(
    tflite_path: Path,
    x_pytorch_layout: np.ndarray,
    input_layout: InputLayout = "keras",
) -> tuple[np.ndarray, dict[str, object]]:
    """Run a saved `.tflite` model on normalized [N, C, L] inputs.

    Use `input_layout="keras"` for Keras-exported models that expect [N, L, C].
    Use `input_layout="pytorch"` for models that already expect [N, C, L].
    """
    samples = np.asarray(x_pytorch_layout, dtype=np.float32)
    if input_layout == "keras":
        samples = torch_to_keras_input(samples)
    elif input_layout != "pytorch":
        raise ValueError(f"input_layout must be 'keras' or 'pytorch', got {input_layout!r}.")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_shape = tuple(int(v) for v in input_detail["shape"][1:])

    if tuple(samples.shape[1:]) != input_shape:
        raise ValueError(
            f"TFLite model expects per-sample input shape {input_shape}, "
            f"but got {tuple(samples.shape[1:])}."
        )

    interpreter.allocate_tensors()
    outputs = []
    for sample in samples:
        input_value = sample[np.newaxis, ...]
        interpreter.set_tensor(input_detail["index"], _quantize_input(input_value, input_detail))
        interpreter.invoke()
        output_value = interpreter.get_tensor(output_detail["index"])
        outputs.append(_dequantize_output(output_value, output_detail)[0])

    metadata = {
        "input_shape": [int(v) for v in input_detail["shape"]],
        "input_dtype": str(input_detail["dtype"]),
        "input_quantization": tuple(float(v) for v in input_detail.get("quantization", (0.0, 0))),
        "output_shape": [int(v) for v in output_detail["shape"]],
        "output_dtype": str(output_detail["dtype"]),
        "output_quantization": tuple(float(v) for v in output_detail.get("quantization", (0.0, 0))),
    }
    return np.stack(outputs, axis=0).astype(np.float32, copy=False), metadata


def validate_keras_tflite_variant(
    variant_name: str,
    model_path: Path,
    x_test_norm: np.ndarray,
    y_test: np.ndarray,
    keras_logits: np.ndarray,
    keras_accuracy: float,
    config: Any,
    torch_logits: Optional[np.ndarray] = None,
    torch_accuracy: Optional[float] = None,
) -> dict[str, object]:
    """Run one Keras-exported TFLite variant and compare it with available baselines."""
    output_logits, interpreter_metadata = run_tflite_model(model_path, x_test_norm, input_layout="keras")
    prob = softmax_np(output_logits)
    pred_idx = np.argmax(prob, axis=1).astype(np.int64)
    accuracy = float(np.mean(pred_idx == y_test))
    accuracy_diff_vs_keras = accuracy - keras_accuracy
    parity_vs_keras = compute_logit_parity(keras_logits, output_logits)

    result = {
        "path": Path(model_path),
        "logits": output_logits,
        "prob": prob,
        "pred_idx": pred_idx,
        "accuracy": accuracy,
        "accuracy_diff_vs_keras": accuracy_diff_vs_keras,
        "parity_vs_keras": parity_vs_keras,
        "interpreter_metadata": interpreter_metadata,
    }

    if torch_logits is not None and torch_accuracy is not None:
        accuracy_diff_vs_pytorch = accuracy - torch_accuracy
        parity_vs_pytorch = compute_logit_parity(torch_logits, output_logits)
        result["accuracy_diff_vs_pytorch"] = accuracy_diff_vs_pytorch
        result["parity"] = parity_vs_pytorch
        result["parity_vs_pytorch"] = parity_vs_pytorch
        print(f"{variant_name} holdout test accuracy: {accuracy:.4f} ({accuracy_diff_vs_pytorch:+.4f} vs PyTorch)")
        print(f"{variant_name} max abs logit diff vs PyTorch: {parity_vs_pytorch['max_abs_diff']:.8g}")
        print(f"{variant_name} max abs logit diff vs Keras: {parity_vs_keras['max_abs_diff']:.8g}")
        warning_parity = parity_vs_pytorch
        warning_text = "Check Keras transfer, TFLite conversion, and quantization."
    else:
        result["parity"] = parity_vs_keras
        print(f"{variant_name} holdout test accuracy: {accuracy:.4f} ({accuracy_diff_vs_keras:+.4f} vs Keras)")
        print(f"{variant_name} max abs logit diff vs Keras: {parity_vs_keras['max_abs_diff']:.8g}")
        warning_parity = parity_vs_keras
        warning_text = "Check TFLite conversion and quantization."

    print(
        f"{variant_name} input_dtype={interpreter_metadata['input_dtype']} "
        f"input_quantization={interpreter_metadata['input_quantization']} "
        f"output_dtype={interpreter_metadata['output_dtype']} "
        f"output_quantization={interpreter_metadata['output_quantization']}"
    )
    if warning_parity["max_abs_diff"] > config.parity_warning_threshold:
        print(
            f"WARNING: {variant_name} logit difference is above "
            f"{config.parity_warning_threshold:.1e}. {warning_text}"
        )

    return result
