from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Retrain_Keras_Method.Functions.config import (
    DEFAULT_OUTPUT_DIR,
    FinalTrainingConfig,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Functions.data_pipeline import (
    build_raw_multichannel_dataset,
    encode_labels,
    fit_transform_channel_scalers,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Retrain_Keras_Method.Functions.final_artifacts import (
    save_final_artifacts,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Functions.keras_model import (
    torch_to_keras_input,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Retrain_Keras_Method.Functions.training_utils import (
    build_compiled_model,
    choose_final_epochs,
    evaluate_model,
    fit_keras_model,
    set_random_seed,
    to_one_hot,
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
# reruns Keras CV and uses the median best epoch for final training.
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
REPRESENTATIVE_COUNT = 128

# Keep True for structural parity with the original PyTorch model.
MATCH_PYTORCH_FLATTEN = True

FINAL_CONFIG = FinalTrainingConfig(
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
    random_shift_max_points=RANDOM_SHIFT_MAX_POINTS,
    random_shift_fill_mode=RANDOM_SHIFT_FILL_MODE,
    match_pytorch_flatten=MATCH_PYTORCH_FLATTEN,
    random_seed=RANDOM_SEED,
    representative_count=REPRESENTATIVE_COUNT,
)


# =========================
# Run Pipeline
# =========================
config = FINAL_CONFIG

if config.model_name != "shared_backbone_2ch":
    raise ValueError("This Keras final-training script currently supports only shared_backbone_2ch.")

set_random_seed(config.random_seed)
print(f"TensorFlow version: {tf.__version__}")

x_all, y_all, removed_zero_sample_indices = build_raw_multichannel_dataset(config)
x_trainval, x_test, y_trainval_labels, y_test_labels = Preprocessing.split_holdout(
    x_all,
    y_all,
    test_size=config.test_size,
    random_seed=config.random_seed,
)
final_epochs = choose_final_epochs(config, x_trainval, y_trainval_labels)

x_train_norm, x_test_norm, scalers = fit_transform_channel_scalers(
    x_trainval,
    x_test,
    clip_max_value=config.clip_max_value,
)
y_train, label_to_idx, idx_to_label = encode_labels(y_trainval_labels, config.class_order)
y_test = np.asarray([label_to_idx[label] for label in y_test_labels], dtype=np.int64)

model = build_compiled_model(
    input_length=x_train_norm.shape[2],
    in_channels=x_train_norm.shape[1],
    num_classes=len(label_to_idx),
    config=config,
)
model.summary()
history = fit_keras_model(
    model,
    torch_to_keras_input(x_train_norm),
    to_one_hot(y_train, len(label_to_idx)),
    config,
    epochs=final_epochs,
)
test_accuracy, test_pred_idx, test_prob = evaluate_model(
    model,
    torch_to_keras_input(x_test_norm),
    y_test,
    batch_size=config.batch_size,
)
print(f"Final holdout test accuracy: {test_accuracy:.4f}")

artifacts = save_final_artifacts(
    model=model,
    config=config,
    final_epochs=final_epochs,
    label_to_idx=label_to_idx,
    idx_to_label=idx_to_label,
    x_train_norm=x_train_norm,
    x_test_norm=x_test_norm,
    y_test=y_test,
    y_test_labels=y_test_labels,
    test_pred_idx=test_pred_idx,
    test_prob=test_prob,
    scalers=scalers,
    removed_zero_sample_indices=removed_zero_sample_indices,
    test_accuracy=test_accuracy,
    history=history,
)
print("Saved final deployment artifacts:")
for name, path in artifacts.items():
    print(f"  {name}: {path}")
