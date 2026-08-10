from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Functions.data_pipeline import (
    build_raw_multichannel_dataset,
    encode_labels,
    fit_transform_channel_scalers,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.config import (
    DEFAULT_OUTPUT_DIR,
    LiteRTTorchConfig,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.final_artifacts import (
    save_final_artifacts,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.litert_export import (
    export_pytorch_model_to_litert,
    quantize_litert_model,
    run_tflite_model,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.training_utils import (
    choose_final_epochs,
    compute_logit_parity,
    evaluate_pytorch_model,
    set_random_seed,
    train_final_pytorch_model,
)
from IRS_Insecticide_Residual.Utility_Functions import Preprocessing


# =========================
# Editable Run Settings
# =========================
# Edit these values directly, the same way Classify_Raw_Data.py is currently
# used. Then run this file from PyCharm or with `python -m ...final_model_training`.
EXCEL_PATH = "/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# If FINAL_EPOCHS is None and RUN_CV_FOR_EPOCH_SELECTION is True, the script
# reruns PyTorch CV and uses the median best epoch for final training.
FINAL_EPOCHS = None
RUN_CV_FOR_EPOCH_SELECTION = True
MAX_CV_EPOCHS = 100

BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.3
PATIENCE = 25
USE_LR_SCHEDULER = True
SCHEDULER_FACTOR = 0.7
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6

RANDOM_SEED = 42
RANDOM_SHIFT_MAX_POINTS = 5
RANDOM_SHIFT_FILL_MODE = "wrap"
DEVICE = None  # Use None for auto, or set "cpu", "cuda", "cuda:0".
NUM_WORKERS = 0

# Keep False by default because final training has no validation split.
FINAL_USE_TRAIN_LOSS_SCHEDULER = False

REPRESENTATIVE_COUNT = 128
PARITY_SAMPLE_COUNT = 128
PARITY_WARNING_THRESHOLD = 1e-4

# Optional no-calibration quantizer recipe, for example "dynamic_wi8_afp32".
# Leave as None for float LiteRT export only.
QUANTIZE_RECIPE = None

SKIP_LITERT_EXPORT = False
SKIP_TFLITE_VALIDATION = False

FINAL_CONFIG = LiteRTTorchConfig(
    excel_path=EXCEL_PATH,
    output_dir=OUTPUT_DIR,
    final_epochs=FINAL_EPOCHS,
    run_cv_for_epoch_selection=RUN_CV_FOR_EPOCH_SELECTION,
    max_cv_epochs=MAX_CV_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    label_smoothing=LABEL_SMOOTHING,
    patience=PATIENCE,
    use_lr_scheduler=USE_LR_SCHEDULER,
    scheduler_factor=SCHEDULER_FACTOR,
    scheduler_patience=SCHEDULER_PATIENCE,
    scheduler_min_lr=SCHEDULER_MIN_LR,
    random_seed=RANDOM_SEED,
    random_shift_max_points=RANDOM_SHIFT_MAX_POINTS,
    random_shift_fill_mode=RANDOM_SHIFT_FILL_MODE,
    representative_count=REPRESENTATIVE_COUNT,
    final_use_train_loss_scheduler=FINAL_USE_TRAIN_LOSS_SCHEDULER,
    device=DEVICE,
    num_workers=NUM_WORKERS,
    parity_sample_count=PARITY_SAMPLE_COUNT,
    parity_warning_threshold=PARITY_WARNING_THRESHOLD,
    quantize_recipe=QUANTIZE_RECIPE,
)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32, copy=False)


# =========================
# Run Pipeline
# =========================
config = FINAL_CONFIG

if config.model_name != "shared_backbone_2ch":
    raise ValueError("This LiteRT Torch script currently supports only shared_backbone_2ch.")

set_random_seed(config.random_seed)
x_all, y_all, removed_zero_sample_indices = build_raw_multichannel_dataset(config)
x_trainval, x_test, y_trainval_labels, y_test_labels = Preprocessing.split_holdout(
    x_all,
    y_all,
    test_size=config.test_size,
    random_seed=config.random_seed,
)

final_epochs, epoch_selection = choose_final_epochs(config, x_trainval, y_trainval_labels)
x_train_norm, x_test_norm, scalers = fit_transform_channel_scalers(
    x_trainval,
    x_test,
    clip_max_value=config.clip_max_value,
)
y_train, label_to_idx, idx_to_label = encode_labels(y_trainval_labels, config.class_order)
y_test = np.asarray([label_to_idx[label] for label in y_test_labels], dtype=np.int64)

pytorch_model, train_history, device = train_final_pytorch_model(
    x_train_norm=x_train_norm,
    y_train=y_train,
    final_epochs=final_epochs,
    config=config,
)
torch_test_accuracy, torch_pred_idx, torch_prob, torch_logits = evaluate_pytorch_model(
    pytorch_model,
    x_test_norm,
    y_test,
    device=device,
    batch_size=config.batch_size,
)
print(f"Final PyTorch holdout test accuracy: {torch_test_accuracy:.4f}")

output_dir = Path(config.output_dir)
tflite_path = None
quantized_tflite_path = None
litert_edge_sample_logits = None
litert_edge_sample_parity = None
tflite_logits = None
tflite_prob = None
tflite_pred_idx = None
tflite_test_accuracy = None
tflite_parity = None
tflite_interpreter_metadata = None

if not SKIP_LITERT_EXPORT:
    tflite_path = output_dir / "shared_backbone_litert_float.tflite"
    sample_input = torch.as_tensor(x_test_norm[:1], dtype=torch.float32, device=device)
    sample_torch_logits = torch_logits[:1]
    tflite_path, litert_edge_sample_logits = export_pytorch_model_to_litert(
        pytorch_model,
        output_path=tflite_path,
        sample_input=sample_input,
    )
    litert_edge_sample_parity = compute_logit_parity(sample_torch_logits, litert_edge_sample_logits)
    print(f"LiteRT Torch sample max abs logit diff: {litert_edge_sample_parity['max_abs_diff']:.8g}")

    if config.quantize_recipe is not None:
        quantized_tflite_path = output_dir / f"shared_backbone_litert_{config.quantize_recipe}.tflite"
        quantize_litert_model(
            source_path=tflite_path,
            output_path=quantized_tflite_path,
            recipe_name=config.quantize_recipe,
        )
        print(f"Saved quantized LiteRT model: {quantized_tflite_path}")

    validation_path = quantized_tflite_path or tflite_path
    if not SKIP_TFLITE_VALIDATION:
        parity_count = min(config.parity_sample_count, len(x_test_norm))
        tflite_logits, tflite_interpreter_metadata = run_tflite_model(
            validation_path,
            x_test_norm,
            limit=None,
        )
        tflite_prob = _softmax_np(tflite_logits)
        tflite_pred_idx = np.argmax(tflite_prob, axis=1).astype(np.int64)
        tflite_test_accuracy = float(np.mean(tflite_pred_idx == y_test))
        tflite_parity = compute_logit_parity(
            torch_logits[:parity_count],
            tflite_logits[:parity_count],
        )
        print(f"LiteRT/TFLite holdout test accuracy: {tflite_test_accuracy:.4f}")
        print(f"LiteRT/TFLite max abs logit diff: {tflite_parity['max_abs_diff']:.8g}")
        if tflite_parity["max_abs_diff"] > config.parity_warning_threshold:
            print(
                "WARNING: LiteRT/TFLite logit difference is above "
                f"{config.parity_warning_threshold:.1e}. Check exported ops and quantization."
            )

artifacts = save_final_artifacts(
    pytorch_model=pytorch_model,
    config=config,
    final_epochs=final_epochs,
    epoch_selection=epoch_selection,
    label_to_idx=label_to_idx,
    idx_to_label=idx_to_label,
    x_train_norm=x_train_norm,
    x_test_norm=x_test_norm,
    y_test=y_test,
    y_test_labels=y_test_labels,
    torch_pred_idx=torch_pred_idx,
    torch_prob=torch_prob,
    torch_logits=torch_logits,
    scalers=scalers,
    removed_zero_sample_indices=removed_zero_sample_indices,
    torch_test_accuracy=torch_test_accuracy,
    train_history=train_history,
    tflite_path=tflite_path,
    quantized_tflite_path=quantized_tflite_path,
    litert_edge_sample_logits=litert_edge_sample_logits,
    litert_edge_sample_parity=litert_edge_sample_parity,
    tflite_logits=tflite_logits,
    tflite_pred_idx=tflite_pred_idx,
    tflite_prob=tflite_prob,
    tflite_test_accuracy=tflite_test_accuracy,
    tflite_parity=tflite_parity,
    tflite_interpreter_metadata=tflite_interpreter_metadata,
)
print("Saved final LiteRT Torch deployment artifacts:")
for name, path in artifacts.items():
    print(f"  {name}: {path}")
