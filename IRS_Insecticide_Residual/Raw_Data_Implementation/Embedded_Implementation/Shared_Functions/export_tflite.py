from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


def representative_dataset_from_npy(path: Path) -> Iterable[list[np.ndarray]]:
    """Yield representative samples saved as [N, C, L] or [N, L, C]."""
    samples = np.load(path).astype(np.float32)
    if samples.ndim != 3:
        raise ValueError(f"Representative data must be 3D, got shape {samples.shape}")
    if samples.shape[1] in {1, 2, 3, 4}:
        samples = samples.transpose(0, 2, 1)
    for sample in samples:
        yield [sample[np.newaxis, ...].astype(np.float32)]


def convert_to_tflite(
    keras_model_path: Path,
    output_path: Path,
    representative_npy: Path | None = None,
    int8: bool = True,
    dynamic_range: bool = False,
) -> None:
    convert_keras_model_to_tflite(
        keras_model=tf.keras.models.load_model(keras_model_path),
        output_path=output_path,
        representative_npy=representative_npy,
        int8=int8,
        dynamic_range=dynamic_range,
    )


def convert_keras_model_to_tflite(
    keras_model: tf.keras.Model,
    output_path: Path,
    representative_npy: Path | None = None,
    int8: bool = True,
    dynamic_range: bool = False,
) -> None:
    if int8 and dynamic_range:
        raise ValueError("Use either int8 full quantization or dynamic-range quantization, not both.")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if int8:
        if representative_npy is None:
            raise ValueError("--representative-npy is required for int8 conversion")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset_from_npy(representative_npy)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif dynamic_range:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    model_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model_bytes)
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
                representative_npy=None,
                int8=False,
            )
        elif variant_name == "dynamic_wi8_afp32":
            variant_tflite_path = output_dir / "shared_backbone_dynamic_wi8_afp32.tflite"
            convert_keras_model_to_tflite(
                keras_model=keras_model,
                output_path=variant_tflite_path,
                representative_npy=None,
                int8=False,
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
                representative_npy=representative_path_for_export,
                int8=True,
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

    convert_to_tflite(
        keras_model_path=args.keras_model,
        output_path=args.output,
        representative_npy=args.representative_npy,
        int8=not args.float and not args.dynamic_range,
        dynamic_range=args.dynamic_range,
    )


if __name__ == "__main__":
    main()
