from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.data_pipeline import (
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
    export_litert_tflite_variants,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.calibration import (
    build_stratified_representative_indices,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.metrics import (
    compute_logit_parity,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.training_utils import (
    choose_final_epochs,
    evaluate_pytorch_model,
    set_random_seed,
    train_final_pytorch_model,
)
from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.LiteRT_Torch_Method.Functions.validation_utils import (
    build_litert_validation_placeholders,
    validate_tflite_variant,
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
# reruns PyTorch CV and uses the second-largest best epoch for final training.
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
DEVICE = None  # Use None for auto, or set "cpu", "cuda", "cuda:0".
NUM_WORKERS = 0

REPRESENTATIVE_COUNT = 128  # The number of normalized training sample saved for TFLite int8 quantization calibration.
PARITY_WARNING_THRESHOLD = 1e-4  # the warning cutoff for model-output mismatch.
CALIBRATION_THREADS = 16  # Number of CPU threads used by ai-edge-quantizer calibration.

# Control the LiteRT/TFLite export configuration.
QUANTIZE_RECIPES = ("dynamic_wi8_afp32", "static_wi8_ai8")  # Save both half-quant and full-quant models for comparison.
SKIP_LITERT_EXPORT = False  # if the trained/evaluated/saved PyTorch model will finally be converted into .tflite.
SKIP_TFLITE_VALIDATION = False  # if the converted TFLite model will be compared with the original PyTorch model.

FINAL_CONFIG = LiteRTTorchConfig(  # including both properties from config.EmbeddedPipelineConfig and LiteRTTorchConfig
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
    device=DEVICE,
    num_workers=NUM_WORKERS,
    parity_warning_threshold=PARITY_WARNING_THRESHOLD,
    quantize_recipes=QUANTIZE_RECIPES,
    calibration_threads=CALIBRATION_THREADS,
)


# =========================
# Validate configuration
# =========================
config = FINAL_CONFIG

if config.model_name != "shared_backbone_2ch":
    raise ValueError("This LiteRT Torch script currently supports only shared_backbone_2ch.")


# =========================
# Build holdout data split
# =========================
set_random_seed(config.random_seed)
x_all, y_all, removed_zero_sample_indices = build_raw_multichannel_dataset(config)
x_trainval, x_test, y_trainval_labels, y_test_labels = Preprocessing.split_holdout(
    x_all,
    y_all,
    test_size=config.test_size,
    random_seed=config.random_seed,
)


# =========================
# Select final epochs and preprocess data
# =========================
# Use the fixed FINAL_EPOCHS value, or rerun PyTorch CV to choose the final epoch count.
final_epochs, epoch_selection = choose_final_epochs(config, x_trainval, y_trainval_labels)
# Fit per-channel StandardScaler objects on trainval only, then reuse them for test and deployment.
x_train_norm, x_test_norm, scalers = fit_transform_channel_scalers(
    x_trainval,
    x_test,
    clip_max_value=config.clip_max_value,
)
y_train, label_to_idx, idx_to_label = encode_labels(y_trainval_labels, config.class_order)
y_test = np.asarray([label_to_idx[label] for label in y_test_labels], dtype=np.int64)
representative_indices = build_stratified_representative_indices(
    y_train,
    count=config.representative_count,
    seed=config.random_seed,
)
representative_samples = x_train_norm[representative_indices]


# =========================
# Train final PyTorch model
# =========================
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


# =========================
# Export and validate LiteRT/TFLite variants from final PyTorch model
# =========================
output_dir = Path(config.output_dir)  # Folder where final LiteRT artifacts are saved.
float_edge_sample_logits = None  # Float LiteRT Torch converted-model logits for one sample before saving TFLite.
float_edge_sample_parity = None  # Difference summary between PyTorch logits and float LiteRT Torch converted-model logits.
tflite_variant_paths = {}  # Paths for float, half-quant, and full-quant TFLite models.
tflite_validation_results = {}  # The accuracy, output logits, and parity results for each TFLite model variant.

if not SKIP_LITERT_EXPORT:
    # sample_input is needed because LiteRT Torch conversion needs to know the model's expected input shape and data type.
    sample_input = torch.as_tensor(x_test_norm[:1], dtype=torch.float32, device=device)
    float_edge_sample_logits, tflite_variant_paths = export_litert_tflite_variants(
        model=pytorch_model,
        output_dir=output_dir,
        sample_input=sample_input,
        representative_samples=representative_samples,
        quantize_recipes=config.quantize_recipes,
        calibration_threads=config.calibration_threads,
    )
    # Compare PyTorch logits with the LiteRT Torch converted-model output for the same sample.
    float_edge_sample_parity = compute_logit_parity(torch_logits[:1], float_edge_sample_logits)
    print(f"Float LiteRT Torch sample max abs logit diff: {float_edge_sample_parity['max_abs_diff']:.8g}")

    # Reload the exported tflite models (float, half-quantized, full-quantized) by desktop TFLite interpreter.
    # And check whether they produce the same output as the original PyTorch model.
    # Note: the tflite model before saving can be slightly different from the one after reloading, so we validate the saved .tflite files.
    if not SKIP_TFLITE_VALIDATION:
        for variant_name, variant_path in tflite_variant_paths.items():
            tflite_validation_results[variant_name] = validate_tflite_variant(
                variant_name=variant_name,
                model_path=variant_path,
                x_test_norm=x_test_norm,
                y_test=y_test,
                torch_logits=torch_logits,
                torch_accuracy=torch_test_accuracy,
                config=config,
            )

        # Print an accuracy and digit summary of the TFLite variant comparison results.
        print("TFLite variant comparison summary:")
        print(f"  PyTorch/original accuracy: {torch_test_accuracy:.4f}")
        for variant_name, result in tflite_validation_results.items():
            metadata = result["interpreter_metadata"]
            print(
                f"  {variant_name}: accuracy={result['accuracy']:.4f} "
                f"accuracy_diff={result['accuracy_diff_vs_pytorch']:+.4f} "
                f"max_abs_logit_diff={result['parity']['max_abs_diff']:.8g} "
                f"input_dtype={metadata['input_dtype']} output_dtype={metadata['output_dtype']}"
            )
        # Print the logits for the first sample from PyTorch and each TFLite variant for visual comparison.
        print(
            "  sample[0] PyTorch logits: "
            f"{np.array2string(torch_logits[0], precision=6, suppress_small=False)}"
        )
        for variant_name, result in tflite_validation_results.items():
            print(
                f"  sample[0] {variant_name} logits: "
                f"{np.array2string(result['logits'][0], precision=6, suppress_small=False)}"
            )

    elif tflite_variant_paths: # if validation was skipped but we exported some TFLite files...
        tflite_validation_results = build_litert_validation_placeholders(tflite_variant_paths)

# =========================
# Save final tflite outputs
# =========================
artifacts = save_final_artifacts(  # Save all important output files, and return their file paths
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
    representative_samples=representative_samples,
    representative_indices=representative_indices,
    scalers=scalers,
    removed_zero_sample_indices=removed_zero_sample_indices,
    torch_test_accuracy=torch_test_accuracy,
    train_history=train_history,
    tflite_variant_paths=tflite_variant_paths,
    float_edge_sample_logits=float_edge_sample_logits,
    float_edge_sample_parity=float_edge_sample_parity,
    tflite_validation_results=tflite_validation_results,
)
print("Saved final LiteRT Torch deployment artifacts.")
# for name, path in artifacts.items():
#     print(f"  {name}: {path}")
