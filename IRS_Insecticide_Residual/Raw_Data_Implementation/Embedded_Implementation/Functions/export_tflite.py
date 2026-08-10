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
) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(keras_model_path))

    if int8:
        if representative_npy is None:
            raise ValueError("--representative-npy is required for int8 conversion")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset_from_npy(representative_npy)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    model_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model_bytes)
    print(f"Saved TFLite model: {output_path}")
    print(f"Size: {len(model_bytes) / 1024:.1f} KiB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Keras shared-backbone model to TFLite.")
    parser.add_argument("--keras-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--representative-npy", type=Path, default=None)
    parser.add_argument("--float", action="store_true", help="Export float32 TFLite instead of full int8.")
    args = parser.parse_args()

    convert_to_tflite(
        keras_model_path=args.keras_model,
        output_path=args.output,
        representative_npy=args.representative_npy,
        int8=not args.float,
    )


if __name__ == "__main__":
    main()

