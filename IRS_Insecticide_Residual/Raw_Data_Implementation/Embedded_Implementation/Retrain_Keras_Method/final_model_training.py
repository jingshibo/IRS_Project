from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Retrain_Keras_Method.Functions.config import (
    DEFAULT_OUTPUT_DIR,
    FinalTrainingConfig,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.data_pipeline import (
    build_raw_multichannel_dataset,
    encode_labels,
    fit_transform_channel_scalers,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Retrain_Keras_Method.Functions.final_artifacts import (
    save_final_artifacts,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.calibration import (
    build_stratified_representative_indices,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.export_tflite import (
    build_tflite_validation_placeholders,
    export_keras_tflite_variants,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.keras_model import (
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
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.tflite_utils import (
    validate_keras_tflite_variant,
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
# reruns Keras CV and uses the second-largest best epoch for final training.
FINAL_EPOCHS = 60
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
REPRESENTATIVE_COUNT = 128  # The number of normalized training samples saved for optional TFLite int8 calibration.
PARITY_WARNING_THRESHOLD = 1e-4  # Warning cutoff for Keras-vs-TFLite output mismatch.

# Keep True for architecture parity with the original PyTorch model.
MATCH_PYTORCH_FLATTEN = True  # Keep the same flatten ordering as the PyTorch-style model for consistency.

# Control the Keras/TFLite export configuration.
TFLITE_VARIANTS = (
    "float",
    "dynamic_wi8_afp32",
    "full_int8",
)  # Save float, dynamic-range, and calibrated full-int8 models for comparison.
SKIP_TFLITE_EXPORT = False  # Whether to skip exporting .tflite files after final Keras training.
SKIP_TFLITE_VALIDATION = False  # Whether to skip desktop TFLite validation.

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
    parity_warning_threshold=PARITY_WARNING_THRESHOLD,
    tflite_variants=TFLITE_VARIANTS,
)


# =========================
# Validate configuration
# =========================
config = FINAL_CONFIG
if config.model_name != "shared_backbone_2ch":
    raise ValueError("This Keras final-training script currently supports only shared_backbone_2ch.")


# =========================
# Build holdout data split
# =========================
set_random_seed(config.random_seed)
print(f"TensorFlow version: {tf.__version__}")

# Build the raw multichannel dataset before any train/test split or normalization.
x_all, y_all, removed_zero_sample_indices = build_raw_multichannel_dataset(config)
# Keep holdout test data separate. It is used only for final Keras and TFLite evaluation.
x_trainval, x_test, y_trainval_labels, y_test_labels = Preprocessing.split_holdout(
    x_all,
    y_all,
    test_size=config.test_size,
    random_seed=config.random_seed,
)


# =========================
# Select final epochs and preprocess data
# =========================
# Use the fixed FINAL_EPOCHS value, or rerun Keras CV to choose the final epoch count.
final_epochs, epoch_selection = choose_final_epochs(config, x_trainval, y_trainval_labels)

# Fit per-channel StandardScaler objects on trainval only, then reuse them for test and deployment.
x_train_norm, x_test_norm, scalers = fit_transform_channel_scalers(
    x_trainval,
    x_test,
    clip_max_value=config.clip_max_value,
)
y_train, label_to_idx, idx_to_label = encode_labels(y_trainval_labels, config.class_order)
y_test = np.asarray([label_to_idx[label] for label in y_test_labels], dtype=np.int64)
# Pick a small stratified subset of normalized training samples for full-int8 TFLite calibration.
representative_indices = build_stratified_representative_indices(
    y_train,
    count=config.representative_count,
    seed=config.random_seed,
)
representative_samples = x_train_norm[representative_indices]


# =========================
# Train final Keras model
# =========================
# Build and compile the Keras model with the correct input shape: [batch, length, channels].
model = build_compiled_model(
    input_length=x_train_norm.shape[2],
    in_channels=x_train_norm.shape[1],
    num_classes=len(label_to_idx),
    config=config,
)
model.summary()
# Keras trains on [N, L, C], so convert the normalized [N, C, L] data before fitting.
history = fit_keras_model(
    model,
    torch_to_keras_input(x_train_norm),
    to_one_hot(y_train, len(label_to_idx)),
    config,
    epochs=final_epochs,
)
# Evaluate the saved-framework source model before exporting any TFLite variants.
test_accuracy, test_pred_idx, test_prob, test_logits = evaluate_model(
    model,
    torch_to_keras_input(x_test_norm),
    y_test,
    batch_size=config.batch_size,
)
print(f"Final holdout test accuracy: {test_accuracy:.4f}")


# =========================
# Export and validate TFLite variants from retrained Keras model
# =========================
tflite_variant_paths = {}
tflite_validation_results = {}
if not SKIP_TFLITE_EXPORT:
    # Export float, dynamic-range, and full-int8 variants from the trained Keras model.
    tflite_variant_paths = export_keras_tflite_variants(
        keras_model=model,
        output_dir=Path(config.output_dir),
        variant_names=config.tflite_variants,
        representative_samples=representative_samples,
    )

    if not SKIP_TFLITE_VALIDATION:
        # Reload each saved .tflite file with the desktop interpreter and compare it with Keras logits.
        for variant_name, variant_path in tflite_variant_paths.items():
            tflite_validation_results[variant_name] = validate_keras_tflite_variant(
                variant_name=variant_name,
                model_path=variant_path,
                x_test_norm=x_test_norm,
                y_test=y_test,
                keras_logits=test_logits,
                keras_accuracy=test_accuracy,
                config=config,
            )

        # Print an accuracy and logit summary for side-by-side TFLite variant comparison.
        print("TFLite variant comparison summary:")
        print(f"  Retrained Keras accuracy: {test_accuracy:.4f}")
        for variant_name, result in tflite_validation_results.items():
            metadata = result["interpreter_metadata"]
            print(
                f"  {variant_name}: accuracy={result['accuracy']:.4f} "
                f"accuracy_diff={result['accuracy_diff_vs_keras']:+.4f} "
                f"max_abs_logit_diff={result['parity']['max_abs_diff']:.8g} "
                f"input_dtype={metadata['input_dtype']} output_dtype={metadata['output_dtype']}"
            )
        print(
            "  sample[0] retrained Keras logits: "
            f"{np.array2string(test_logits[0], precision=6, suppress_small=False)}"
        )
        for variant_name, result in tflite_validation_results.items():
            print(
                f"  sample[0] {variant_name} logits: "
                f"{np.array2string(result['logits'][0], precision=6, suppress_small=False)}"
            )
    else:
        # Keep metadata structure consistent when export is enabled but saved TFLite validation is skipped.
        tflite_validation_results = build_tflite_validation_placeholders(tflite_variant_paths)


# =========================
# Save final Keras and TFLite artifacts
# =========================
artifacts = save_final_artifacts(
    model=model,
    config=config,
    final_epochs=final_epochs,
    epoch_selection=epoch_selection,
    label_to_idx=label_to_idx,
    idx_to_label=idx_to_label,
    x_train_norm=x_train_norm,
    x_test_norm=x_test_norm,
    y_test=y_test,
    y_test_labels=y_test_labels,
    test_pred_idx=test_pred_idx,
    test_prob=test_prob,
    test_logits=test_logits,
    representative_samples=representative_samples,
    representative_indices=representative_indices,
    scalers=scalers,
    removed_zero_sample_indices=removed_zero_sample_indices,
    test_accuracy=test_accuracy,
    history=history,
    tflite_variant_paths=tflite_variant_paths,
    tflite_validation_results=tflite_validation_results,
)
print("Saved final deployment artifacts.")
# for name, path in artifacts.items():
#     print(f"  {name}: {path}")
