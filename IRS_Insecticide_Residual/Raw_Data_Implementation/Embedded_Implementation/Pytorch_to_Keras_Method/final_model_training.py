from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Functions.data_pipeline import (
    build_raw_multichannel_dataset,
    encode_labels,
    fit_transform_channel_scalers,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Functions.export_tflite import (
    convert_to_tflite,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Functions.keras_model import (
    SharedBackboneConfig,
    build_shared_backbone_keras_model,
    torch_to_keras_input,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Pytorch_to_Keras_Method.Functions.config import (
    DEFAULT_OUTPUT_DIR,
    PytorchToKerasConfig,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Pytorch_to_Keras_Method.Functions.final_artifacts import (
    save_final_artifacts,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Pytorch_to_Keras_Method.Functions.transfer_pytorch_weights import (
    transfer_shared_backbone_weights,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Pytorch_to_Keras_Method.Functions.training_utils import (
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

EXPORT_TFLITE = False
FLOAT_TFLITE = False

FINAL_CONFIG = PytorchToKerasConfig(
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
)


def _keras_logits_and_prob(
    model: tf.keras.Model,
    x_pytorch_layout: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    keras_logits = model.predict(
        torch_to_keras_input(x_pytorch_layout),
        batch_size=batch_size,
        verbose=0,
    ).astype(np.float32, copy=False)
    keras_prob = tf.nn.softmax(keras_logits, axis=1).numpy().astype(np.float32, copy=False)
    return keras_logits, keras_prob


# =========================
# Run Pipeline
# =========================
config = FINAL_CONFIG

if config.model_name != "shared_backbone_2ch":
    raise ValueError("This PyTorch-to-Keras script currently supports only shared_backbone_2ch.")
if not config.match_pytorch_flatten:
    raise ValueError("PyTorch-to-Keras exact weight transfer requires match_pytorch_flatten=True.")

set_random_seed(config.random_seed)
print(f"TensorFlow version: {tf.__version__}")

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

keras_config = SharedBackboneConfig(
    input_length=x_train_norm.shape[2],
    in_channels=x_train_norm.shape[1],
    num_classes=len(label_to_idx),
)
keras_model = build_shared_backbone_keras_model(
    keras_config,
    include_softmax=False,
    match_pytorch_flatten=True,
)
transfer_shared_backbone_weights(keras_model, pytorch_model.state_dict())

keras_logits, keras_prob = _keras_logits_and_prob(keras_model, x_test_norm, batch_size=config.batch_size)
keras_pred_idx = np.argmax(keras_prob, axis=1).astype(np.int64)
keras_test_accuracy = float(np.mean(keras_pred_idx == y_test))
parity_count = min(config.parity_sample_count, len(x_test_norm))
parity = compute_logit_parity(
    torch_logits[:parity_count],
    keras_logits[:parity_count],
)
print(f"Transferred Keras holdout test accuracy: {keras_test_accuracy:.4f}")
print(f"PyTorch-to-Keras max abs logit diff: {parity['max_abs_diff']:.8g}")
if parity["max_abs_diff"] > config.parity_warning_threshold:
    print(
        "WARNING: PyTorch-to-Keras logit difference is above "
        f"{config.parity_warning_threshold:.1e}. Check architecture and weight-transfer parity."
    )

artifacts = save_final_artifacts(
    pytorch_model=pytorch_model,
    keras_model=keras_model,
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
    keras_pred_idx=keras_pred_idx,
    keras_prob=keras_prob,
    keras_logits=keras_logits,
    scalers=scalers,
    removed_zero_sample_indices=removed_zero_sample_indices,
    torch_test_accuracy=torch_test_accuracy,
    keras_test_accuracy=keras_test_accuracy,
    parity=parity,
    train_history=train_history,
)

if EXPORT_TFLITE:
    tflite_name = "shared_backbone_float.tflite" if FLOAT_TFLITE else "shared_backbone_int8.tflite"
    tflite_path = Path(config.output_dir) / tflite_name
    convert_to_tflite(
        keras_model_path=artifacts["keras_model"],
        output_path=tflite_path,
        representative_npy=None if FLOAT_TFLITE else artifacts["representative"],
        int8=not FLOAT_TFLITE,
    )
    artifacts["tflite_model"] = tflite_path

print("Saved final PyTorch-to-Keras deployment artifacts:")
for name, path in artifacts.items():
    print(f"  {name}: {path}")
