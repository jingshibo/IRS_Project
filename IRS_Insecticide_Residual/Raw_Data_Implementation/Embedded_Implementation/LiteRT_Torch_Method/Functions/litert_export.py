from __future__ import annotations

import copy
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
import torch


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"Expected one model output, got {len(value)} outputs.")
        return _to_numpy(value[0])
    return np.asarray(value)


def _import_litert_torch() -> ModuleType:
    """Import the current LiteRT Torch package, with legacy fallback."""
    try:
        import litert_torch

        return litert_torch
    except ImportError:
        try:
            import ai_edge_torch

            return ai_edge_torch
        except ImportError as exc:
            raise ImportError(
                "LiteRT Torch is not installed. Install the current package with "
                "`pip install litert-torch` or the nightly package with "
                "`pip install --pre litert-torch-nightly`."
            ) from exc


def export_pytorch_model_to_litert(
    model: torch.nn.Module,
    output_path: Path,
    sample_input: torch.Tensor,
) -> tuple[Path, np.ndarray]:
    """Convert a PyTorch model to `.tflite` with LiteRT Torch and return sample logits."""
    litert_torch = _import_litert_torch()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = copy.deepcopy(model).cpu().eval()
    sample_cpu = sample_input.detach().cpu().to(torch.float32)
    with torch.no_grad():
        edge_model = litert_torch.convert(model_cpu, (sample_cpu,))
        edge_output = edge_model(sample_cpu)
    edge_model.export(str(output_path))
    return output_path, _to_numpy(edge_output).astype(np.float32, copy=False)


def quantize_litert_model(
    source_path: Path,
    output_path: Path,
    recipe_name: str,
) -> Path:
    """Apply a no-calibration AI Edge Quantizer recipe to a LiteRT model.

    This helper intentionally supports only recipes that do not need calibration
    data, such as `dynamic_wi8_afp32` or `weight_only_wi8_afp32`. Full W8A8
    static quantization should be handled separately after confirming the exact
    installed ai-edge-quantizer calibration API.
    """
    if recipe_name in {"static_wi8_ai8", "static_int8", "w8a8"}:
        raise ValueError(
            "Static W8A8 quantization needs calibration data and is not handled by this helper. "
            "Export the float LiteRT model first, then use a calibrated AI Edge Quantizer flow."
        )

    try:
        from ai_edge_quantizer import quantizer, recipe
    except ImportError as exc:
        raise ImportError("ai-edge-quantizer is required for post-export quantization.") from exc

    if not hasattr(recipe, recipe_name):
        raise ValueError(f"ai_edge_quantizer.recipe has no recipe named {recipe_name!r}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qt = quantizer.Quantizer(str(source_path))
    qt.load_quantization_recipe(getattr(recipe, recipe_name)())
    result = qt.quantize()
    if hasattr(result, "export_model"):
        result.export_model(str(output_path))
    elif hasattr(result, "save"):
        result.save(str(output_path.parent), model_name=output_path.stem)
    else:
        raise RuntimeError("Unsupported ai-edge-quantizer result object; cannot save quantized model.")
    return output_path


def _load_interpreter(tflite_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(str(tflite_path))
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter

            return Interpreter(model_path=str(tflite_path))
        except ImportError:
            try:
                import tensorflow as tf

                return tf.lite.Interpreter(model_path=str(tflite_path))
            except ImportError as exc:
                raise ImportError(
                    "A LiteRT/TFLite interpreter is required for validation. Install one of "
                    "`ai-edge-litert`, `tflite-runtime`, or `tensorflow`."
                ) from exc


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
    limit: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run a `.tflite` model sample-by-sample on [N, C, L] input."""
    samples = np.asarray(x_pytorch_layout, dtype=np.float32)
    if limit is not None:
        samples = samples[:limit]
    interpreter = _load_interpreter(Path(tflite_path))
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_shape = np.asarray(input_detail["shape"], dtype=np.int64)
    expected_sample_shape = tuple(int(v) for v in input_shape[1:])

    if tuple(samples.shape[1:]) != expected_sample_shape:
        raise ValueError(
            f"TFLite model expects per-sample input shape {expected_sample_shape}, "
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
