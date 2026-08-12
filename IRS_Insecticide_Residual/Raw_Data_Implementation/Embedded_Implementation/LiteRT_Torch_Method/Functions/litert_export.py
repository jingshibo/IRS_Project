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
    sample_input: torch.Tensor, # LiteRT needs sample_input to build a fixed computation graph.
) -> tuple[Path, np.ndarray]:
    """Convert a PyTorch model to `.tflite` with LiteRT Torch and return sample logits."""
    litert_torch = _import_litert_torch()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = copy.deepcopy(model).cpu().eval()
    sample_cpu = sample_input.detach().cpu().to(torch.float32)
    with torch.no_grad():
        #  sample_input has two purposes:
        # 1. Gives LiteRT Torch the input shape/dtype needed to convert the model.
        edge_model = litert_torch.convert(model_cpu, (sample_cpu,)) # uses sample_cpu to trace/convert the model.
        # 2. Provides one sample for a quick conversion parity check.
        edge_output = edge_model(sample_cpu) # checks the converted LiteRT Torch model output to compare with the original PyTorch model output.
    edge_model.export(str(output_path)) # The exported model can be different from the edge_model due to serialization
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
    # ESP32-S3 deployment will likely need calibrated full-int8 eventually.
    # This helper does not perform that calibration step; use representative_final.npy
    # later with the installed ai-edge-quantizer calibration API.
    if recipe_name in {"static_wi8_ai8", "static_int8", "w8a8"}:
        raise ValueError(
            "Static W8A8 quantization needs calibration data and is not handled by this helper. "
            "Export the float LiteRT model first, then use a calibrated AI Edge Quantizer flow."
        )

    try:
        from ai_edge_quantizer import quantizer, recipe
    except ImportError as exc:
        raise ImportError("ai-edge-quantizer is required for post-export quantization.") from exc

    if not hasattr(recipe, recipe_name): # Checks whether the requested recipe exists
        raise ValueError(f"ai_edge_quantizer.recipe has no recipe named {recipe_name!r}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # quantize the model
    qt = quantizer.Quantizer(str(source_path))
    # No-calibration recipes can be applied directly to the exported float LiteRT model.
    qt.load_quantization_recipe(getattr(recipe, recipe_name)())
    result = qt.quantize()
    _save_quantized_model(result, output_path)
    return output_path


def _save_quantized_model(result, output_path: Path) -> None:
    """Save a quantized model and allow rerunning the script over existing outputs."""
    output_path = Path(output_path)
    if hasattr(result, "export_model"):
        try:
            result.export_model(str(output_path), overwrite=True)
        except TypeError:
            if output_path.exists():
                output_path.unlink()
            result.export_model(str(output_path))
    elif hasattr(result, "save"):
        try:
            result.save(str(output_path.parent), model_name=output_path.stem, overwrite=True)
        except TypeError:
            if output_path.exists():
                output_path.unlink()
            result.save(str(output_path.parent), model_name=output_path.stem)
    else:
        raise RuntimeError("Unsupported ai-edge-quantizer result object; cannot save quantized model.")


def _need_calibration(qt) -> bool:
    """Return Quantizer.need_calibration for package versions exposing it as a property or method."""
    need_calibration = getattr(qt, "need_calibration", None)
    # different ai-edge-quantizer versions expose need_calibration information differently.
    if need_calibration is None:
        raise ValueError("The installed ai-edge-quantizer Quantizer has no need_calibration attribute.")
    if callable(need_calibration): # Some versions use a method: qt.need_calibration(
        return bool(need_calibration())
    return bool(need_calibration) # The installed version uses a boolean property


def _get_signature_input_name(tflite_path: Path) -> tuple[str, str]:
    """Read the LiteRT signature and input name needed by ai-edge-quantizer calibration."""
    interpreter = _load_interpreter(Path(tflite_path)) # Loads the exported float .tflite model on the computer.
    if not hasattr(interpreter, "get_signature_list"): # Checks whether this interpreter can read model signatures.
        raise ValueError("The installed LiteRT interpreter cannot read model signatures.")

    signatures = interpreter.get_signature_list() # signature here means the named callable interface inside the .tflite model.
    if not signatures: # Without the signature, we do not know how to format calibration data for ai-edge-quantizer
        raise ValueError("The LiteRT model has no signature information for calibration.")

    signature_key = next(iter(signatures)) # Gets the first signature name from the model signatures.
    input_names = signatures[signature_key].get("inputs", []) # Gets the input tensor names for that signature.
    if len(input_names) != 1: # Our model requires exactly one input, otherwise fails
        raise ValueError(f"Expected one model input for calibration, got {len(input_names)} inputs.")
    return signature_key, input_names[0]


def quantize_litert_model_with_calibration(
    source_path: Path,
    output_path: Path,
    recipe_name: str,
    representative_samples: np.ndarray,
    calibration_threads: int = 16,
) -> Path:
    """Apply calibrated AI Edge Quantizer static quantization to a LiteRT model.

    This is the full-int8 path for deployment. The source model should be the
    exported float `.tflite` file. The representative samples should be real,
    already-normalized training inputs in PyTorch/LiteRT layout [N, C, L].
    Calibration runs on the computer and measures activation ranges before
    writing the quantized `.tflite` file.
    """
    try:
        from ai_edge_quantizer import quantizer, recipe
    except ImportError as exc:
        raise ImportError("ai-edge-quantizer is required for calibrated quantization.") from exc

    if not hasattr(recipe, recipe_name):
        raise ValueError(f"ai_edge_quantizer.recipe has no recipe named {recipe_name!r}.")

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(representative_samples, dtype=np.float32)
    if samples.ndim < 2:
        raise ValueError(f"Expected representative samples with a batch dimension, got shape {samples.shape}.")

    signature_key, input_name = _get_signature_input_name(source_path)
    calibration_data = {
        signature_key: [
            {input_name: sample[np.newaxis, ...].astype(np.float32, copy=False)}
            for sample in samples
        ]
    }

    qt = quantizer.Quantizer(str(source_path))
    qt.load_quantization_recipe(getattr(recipe, recipe_name)())
    if not _need_calibration(qt):
        raise ValueError(f"Recipe {recipe_name!r} does not require calibration; use quantize_litert_model instead.")

    calibration_result = qt.calibrate(calibration_data, num_threads=calibration_threads)
    result = qt.quantize(calibration_result)
    _save_quantized_model(result, output_path)
    return output_path


def _load_interpreter(tflite_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(str(tflite_path))
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter

            return Interpreter(model_path=str(tflite_path))
        except ImportError as exc:
            raise ImportError(
                "A LiteRT/TFLite interpreter is required for validation. Install "
                "`ai-edge-litert` or `tflite-runtime`, or set SKIP_TFLITE_VALIDATION = True."
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
    limit: Optional[int] = None, # The number of test samples to run
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the exported `.tflite` model on the computer sample-by-sample on [N, C, L] input."""
    samples = np.asarray(x_pytorch_layout, dtype=np.float32)
    if limit is not None:
        samples = samples[:limit]
    interpreter = _load_interpreter(Path(tflite_path)) # Loads the .tflite model using a desktop interpreter.
    # Gets model input/output information
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    # checks the expected input shape
    input_shape = np.asarray(input_detail["shape"], dtype=np.int64)
    expected_sample_shape = tuple(int(v) for v in input_shape[1:]) # A single sample has shape: [3, 540]

    if tuple(samples.shape[1:]) != expected_sample_shape:
        raise ValueError(
            f"TFLite model expects per-sample input shape {expected_sample_shape}, "
            f"but got {tuple(samples.shape[1:])}."
        )

    interpreter.allocate_tensors() # Prepares memory for running inference.
    outputs = [] # tflite_logits.shape == [N, 3]
    for sample in samples:
        input_value = sample[np.newaxis, ...] # A single sample has shape: [3, 540]
        interpreter.set_tensor(input_detail["index"], _quantize_input(input_value, input_detail)) # puts the sample into the model.
        interpreter.invoke() # runs the model on the input sample.
        output_value = interpreter.get_tensor(output_detail["index"]) # gets model output (tflite_logits).
        outputs.append(_dequantize_output(output_value, output_detail)[0]) # converts output back to float32

    metadata = {
        "input_shape": [int(v) for v in input_detail["shape"]],
        "input_dtype": str(input_detail["dtype"]),
        "input_quantization": tuple(float(v) for v in input_detail.get("quantization", (0.0, 0))),
        "output_shape": [int(v) for v in output_detail["shape"]],
        "output_dtype": str(output_detail["dtype"]),
        "output_quantization": tuple(float(v) for v in output_detail.get("quantization", (0.0, 0))),
    }
    return np.stack(outputs, axis=0).astype(np.float32, copy=False), metadata
