from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


def representative_dataset_from_npy(path: Path) -> Iterable[list[np.ndarray]]:
    """Defines a generator function. Yield representative samples in the format TFLite expects: [N, C, L] or [N, L, C]."""
    samples = np.load(path).astype(np.float32) # Load representative samples from a .npy file.
    if samples.ndim != 3:
        raise ValueError(f"Representative data must be 3D, got shape {samples.shape}")
    if samples.shape[1] in {1, 2, 3, 4}:
        samples = samples.transpose(0, 2, 1) # [N, C, L] - PyTorch layout, transpose to [N, L, C] for TFLite.
    for sample in samples:
        # Yield a single sample with batch dimension added, as TFLite expects a list of input arrays.
        yield [sample[np.newaxis, ...].astype(np.float32)] # np.newaxis changes a single sample [540, 3] to [1, 540, 3]


def convert_keras_model_to_tflite(
    keras_model: tf.keras.Model,
    output_path: Path,
    representative_npy: Path | None = None,  # Calibration samples for full-int8 quantization.
    int8: bool = False,  # Whether to export a fully quantized int8 model.
    dynamic_range: bool = False,  # Whether to export dynamic-range quantized weights with float I/O.
) -> None:
    # Full-int8 and dynamic-range quantization are different export modes, so they cannot both be enabled.
    if int8 and dynamic_range:
        raise ValueError("Use either int8 full quantization or dynamic-range quantization, not both.")

    # Create a TensorFlow Lite converter from the in-memory Keras model.
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if int8: # Configure full-int8 quantization when requested.
        # Full-int8 quantization needs representative data to calibrate activation ranges.
        if representative_npy is None:
            # Without representative data, TensorFlow cannot calibrate the int8 activation scales.
            raise ValueError("--representative-npy is required for int8 conversion")

        converter.optimizations = [tf.lite.Optimize.DEFAULT] # Enable TFLite's default optimization pipeline, including quantization.
        converter.representative_dataset = lambda: representative_dataset_from_npy(representative_npy) # # Provide calibration samples
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # Restrict to built-in int8-compatible TFLite operations.
        converter.inference_input_type = tf.int8 # Make the model input tensor int8 for embedded deployment.
        converter.inference_output_type = tf.int8 # Make the model output tensor int8 for embedded deployment.

    elif dynamic_range: # Configure dynamic-range quantization when requested.
        converter.optimizations = [tf.lite.Optimize.DEFAULT] # Dynamic-range quantization not requiring calibration samples.

    model_bytes = converter.convert() # Run the TFLite conversion and receive the serialized model bytes.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model_bytes) # Write the serialized TFLite model to disk.
    print(f"Saved TFLite model: {output_path}")
    print(f"Size: {len(model_bytes) / 1024:.1f} KiB")


def export_keras_tflite_variants(
    keras_model: tf.keras.Model,
    output_dir: Path,
    variant_names: Iterable[str],
    representative_samples: np.ndarray,
) -> dict[str, Path]:
    """Export standard Keras TFLite variants and return their paths."""
    output_dir = Path(output_dir)
    tflite_variant_paths = {}

    for variant_name in variant_names:
        if variant_name == "float":
            variant_tflite_path = output_dir / "shared_backbone_float.tflite"
            convert_keras_model_to_tflite(
                keras_model=keras_model,
                output_path=variant_tflite_path,
            )
        elif variant_name == "dynamic_wi8_afp32":
            variant_tflite_path = output_dir / "shared_backbone_dynamic_wi8_afp32.tflite"
            convert_keras_model_to_tflite(
                keras_model=keras_model,
                output_path=variant_tflite_path,
                dynamic_range=True,
            )
        elif variant_name == "full_int8":
            variant_tflite_path = output_dir / "shared_backbone_full_int8.tflite"
            representative_path_for_export = output_dir / "representative_final.npy"
            representative_path_for_export.parent.mkdir(parents=True, exist_ok=True)
            np.save(representative_path_for_export, representative_samples.astype(np.float32, copy=False))
            convert_keras_model_to_tflite(
                keras_model=keras_model,
                output_path=variant_tflite_path,
                int8=True,
                representative_npy=representative_path_for_export,
            )
        else:
            raise ValueError(f"Unsupported TFLite variant: {variant_name}")
        tflite_variant_paths[variant_name] = variant_tflite_path

    return tflite_variant_paths


def build_tflite_validation_placeholders(
    tflite_variant_paths: dict[str, Path],
    include_pytorch_baseline: bool = False,
) -> dict[str, dict[str, object]]:
    """Build metadata placeholders when saved TFLite validation is skipped."""
    placeholders = {}
    for variant_name, variant_path in tflite_variant_paths.items():
        placeholder = {
            "path": variant_path,
            "logits": None,
            "prob": None,
            "pred_idx": None,
            "accuracy": None,
            "accuracy_diff_vs_keras": None,
            "parity": None,
            "parity_vs_keras": None,
            "interpreter_metadata": None,
        }
        if include_pytorch_baseline:
            placeholder["accuracy_diff_vs_pytorch"] = None
            placeholder["parity_vs_pytorch"] = None
        placeholders[variant_name] = placeholder
    return placeholders


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Keras shared-backbone model to TFLite.")
    parser.add_argument("--keras-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--representative-npy", type=Path, default=None)
    parser.add_argument("--float", action="store_true", help="Export float32 TFLite instead of full int8.")
    parser.add_argument(
        "--dynamic-range",
        action="store_true",
        help="Export dynamic-range quantized TFLite with quantized weights and float input/output.",
    )
    args = parser.parse_args()
    if args.float and args.dynamic_range:
        parser.error("--float and --dynamic-range are mutually exclusive")

    convert_keras_model_to_tflite(
        keras_model=tf.keras.models.load_model(args.keras_model),
        output_path=args.output,
        representative_npy=args.representative_npy,
        int8=not args.float and not args.dynamic_range,
        dynamic_range=args.dynamic_range,
    )


if __name__ == "__main__":
    main()
