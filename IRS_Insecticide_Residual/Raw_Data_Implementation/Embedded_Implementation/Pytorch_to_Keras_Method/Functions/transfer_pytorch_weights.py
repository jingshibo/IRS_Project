from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf
import torch

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.keras_model import (
    SharedBackboneConfig,
    build_shared_backbone_keras_model,
    torch_to_keras_input,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Models.Model_Structure import OneDCNNClassifier


def _load_state_dict(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
        if all(isinstance(k, str) for k in payload.keys()):
            return payload
    raise ValueError(f"Could not find a PyTorch state_dict in {path}")


def _set_conv1d_weights(keras_model: tf.keras.Model, state: Mapping[str, Any], block_idx: int) -> None:
    conv = keras_model.get_layer(f"features_{block_idx}_conv")
    prefix = f"features.features.{block_idx}.0"
    weight = state[f"{prefix}.weight"].detach().cpu().numpy().transpose(2, 1, 0)
    bias = state[f"{prefix}.bias"].detach().cpu().numpy()
    conv.set_weights([weight, bias])


def _set_batch_norm_weights(
    keras_model: tf.keras.Model,
    state: Mapping[str, Any],
    keras_layer_name: str,
    torch_prefix: str,
) -> None:
    bn = keras_model.get_layer(keras_layer_name)
    gamma = state[f"{torch_prefix}.weight"].detach().cpu().numpy()
    beta = state[f"{torch_prefix}.bias"].detach().cpu().numpy()
    moving_mean = state[f"{torch_prefix}.running_mean"].detach().cpu().numpy()
    moving_var = state[f"{torch_prefix}.running_var"].detach().cpu().numpy()
    bn.set_weights([gamma, beta, moving_mean, moving_var])


def _set_dense_weights(
    keras_model: tf.keras.Model,
    state: Mapping[str, Any],
    keras_layer_name: str,
    torch_prefix: str,
) -> None:
    dense = keras_model.get_layer(keras_layer_name)
    weight = state[f"{torch_prefix}.weight"].detach().cpu().numpy().T
    bias = state[f"{torch_prefix}.bias"].detach().cpu().numpy()
    dense.set_weights([weight, bias])


def transfer_shared_backbone_weights(
    keras_model: tf.keras.Model,
    pytorch_state_dict: Mapping[str, Any],
) -> tf.keras.Model:
    """Copy OneDCNNClassifier weights into the equivalent Keras model."""
    for block_idx in range(4):
        _set_conv1d_weights(keras_model, pytorch_state_dict, block_idx)
        _set_batch_norm_weights(
            keras_model,
            pytorch_state_dict,
            keras_layer_name=f"features_{block_idx}_bn",
            torch_prefix=f"features.features.{block_idx}.1",
        )

    _set_dense_weights(keras_model, pytorch_state_dict, "classifier_dense_0", "classifier.1")
    _set_batch_norm_weights(keras_model, pytorch_state_dict, "classifier_bn_0", "classifier.2")
    _set_dense_weights(keras_model, pytorch_state_dict, "classifier_dense_1", "classifier.5")
    _set_batch_norm_weights(keras_model, pytorch_state_dict, "classifier_bn_1", "classifier.6")
    _set_dense_weights(keras_model, pytorch_state_dict, "classifier_logits", "classifier.9")
    return keras_model


def build_pytorch_model_from_state(
    state: Mapping[str, Any],
    input_length: int,
    in_channels: int,
    num_classes: int,
) -> OneDCNNClassifier:
    model = OneDCNNClassifier(in_channels=in_channels, num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_channels, input_length, dtype=torch.float32))
    model.load_state_dict(state)
    model.eval()
    return model


def compare_pytorch_and_keras(
    pytorch_model: OneDCNNClassifier,
    keras_model: tf.keras.Model,
    input_length: int,
    in_channels: int,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x_torch_np = rng.normal(size=(4, in_channels, input_length)).astype(np.float32)
    x_torch = torch.from_numpy(x_torch_np)
    with torch.no_grad():
        torch_logits = pytorch_model(x_torch).detach().cpu().numpy()
    keras_logits = keras_model(torch_to_keras_input(x_torch_np), training=False).numpy()
    abs_diff = np.abs(torch_logits - keras_logits)
    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
    }


def save_metadata(path: Path, config: SharedBackboneConfig, equivalence: dict[str, float]) -> None:
    payload = {
        "model_name": "shared_backbone_2ch",
        "keras_input_layout": "[batch, length, channels]",
        "pytorch_input_layout": "[batch, channels, length]",
        "input_length": config.input_length,
        "in_channels": config.in_channels,
        "num_classes": config.num_classes,
        "class_order": ["LOW", "TARGET", "HIGH"],
        "equivalence": equivalence,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer shared_backbone_2ch PyTorch weights to Keras.")
    parser.add_argument("--pytorch-weights", required=True, type=Path, help="Path to a .pt/.pth state_dict or checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Output .keras file.")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Optional metadata JSON path.")
    parser.add_argument("--input-length", type=int, default=540)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=3)
    args = parser.parse_args()

    config = SharedBackboneConfig(
        input_length=args.input_length,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    )
    state = _load_state_dict(args.pytorch_weights)
    keras_model = build_shared_backbone_keras_model(config, match_pytorch_flatten=True)
    pytorch_model = build_pytorch_model_from_state(
        state,
        input_length=config.input_length,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
    )
    transfer_shared_backbone_weights(keras_model, state)
    equivalence = compare_pytorch_and_keras(
        pytorch_model,
        keras_model,
        input_length=config.input_length,
        in_channels=config.in_channels,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    keras_model.save(args.output)
    print(f"Saved Keras model: {args.output}")
    print(f"Max abs diff: {equivalence['max_abs_diff']:.8g}")
    print(f"Mean abs diff: {equivalence['mean_abs_diff']:.8g}")

    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        save_metadata(args.metadata_output, config, equivalence)
        print(f"Saved metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()
