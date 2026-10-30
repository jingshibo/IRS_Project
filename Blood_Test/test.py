##
import importlib
from pathlib import Path

from Blood_Test.Functions import (
    append_pbs_complex_difference,
    append_pbs_plain_difference,
    apply_savgol_filter_nested,
    apply_savgol_to_re_im_and_recompute,
    downsample_nested_data,
    exclude_data_blocks,
    load_blood_folder,
    plot_difference_channel_samples,
    plot_difference_channel_summary,
    plot_filtering_comparison,
    plot_raw_signal_representatives,
    print_blood_data_summary,
)
from Blood_Test.Model import dataset, training
from Blood_Test.Model.evaluation import (
    plot_recall_matrix_with_numbers,
    print_result_summary,
    summarize_results,
)


## Load the raw data in PyCharm/Python console mode
DATA_DIR = Path(r"/home/shibojing/data/blood")

# Optional exclusions use: (sample type, set number, concentration).
# Use an empty tuple () to include every measurement.
EXCLUDED_DATA_BLOCKS = (
)

# Choose "leave_one_sample_out" for eight folds (S1-S8), or
# "leave_one_set_out" for three folds (set 1-set 3).
EVALUATION_METHOD = "leave_one_set_out"

raw_data = load_blood_folder(DATA_DIR)
raw_data = exclude_data_blocks(
    raw_data,
    excluded_blocks=EXCLUDED_DATA_BLOCKS,
)
print_blood_data_summary(raw_data)
# plot_raw_signal_representatives(raw_data)


## Filter real/imaginary signals and recompute amplitude/phase
filtered_data = apply_savgol_to_re_im_and_recompute(
    raw_data,
    window_length=101,
    polyorder=3,
)
print("Savitzky-Golay filtering completed: window_length=101, polyorder=3")
# plot_filtering_comparison(raw_data, filtered_data)


## Calculate set-specific PBS-referenced features
# For each set, average its five PBS repetitions and append the pointwise
# [real, imaginary, amplitude, phase] differences from that set's mean PBS trace.
plain_difference_data = append_pbs_plain_difference(filtered_data)

# Append two complex-valued comparison features using the same set's mean PBS:
# the phase of (SAU / PBS) and the magnitude of (SAU - PBS).
complex_difference_data = append_pbs_complex_difference(plain_difference_data)

# Smooth all ten existing and PBS-derived channels along the frequency axis.
filtered_difference_data = apply_savgol_filter_nested(
    complex_difference_data,
    window_length=31,
    polyorder=3,
)
print("PBS difference calculation and Savitzky-Golay filtering completed")

plot_difference_channel_samples(
    complex_difference_data,
    channel_indices=(8, 11, 12),
    samples_per_group=4,
    random_seed=42,
)
plot_difference_channel_summary(
    complex_difference_data,
    channel_indices=(8, 11, 12),
)

plot_difference_channel_samples(
    filtered_difference_data,
    channel_indices=(8, 11, 12),
    samples_per_group=4,
    random_seed=42,
)
plot_difference_channel_summary(
    filtered_difference_data,
    channel_indices=(8, 11, 12),
)


## Downsample the processed signals
# Keep every 100th frequency point, reducing 20,000 points to 200 while
# preserving every set, sample, repetition, and all 10 signal channels.
downsampled_difference_data = downsample_nested_data(
    filtered_difference_data,
    ratio=100,
)
print(
    "Downsampling completed: "
    f"SAU 10-4 shape={downsampled_difference_data['SAU']['10-4'].shape}"
)


## Build the three-class concentration dataset and grouped CV splits
# A group is one independent (set, concentration, sample), containing 5 repetitions.
grouped_data = dataset.build_concentration_dataset(
    downsampled_difference_data,
    channel_indices=(11, ),  # complex_ratio_phase
)

# Select either the eight-fold sample evaluation or the stricter three-fold
# set evaluation. Normalization always uses only the training partition.
if EVALUATION_METHOD == "leave_one_sample_out":
    cv_splits = dataset.build_leave_one_sample_out_splits(grouped_data)
elif EVALUATION_METHOD == "leave_one_set_out":
    cv_splits = dataset.build_leave_one_set_out_splits(grouped_data)
else:
    raise ValueError(
        "EVALUATION_METHOD must be 'leave_one_sample_out' or "
        f"'leave_one_set_out', got {EVALUATION_METHOD!r}"
    )

print(
    f"Evaluation method: {EVALUATION_METHOD}\n"
    f"Built {len(cv_splits)} folds from {len(grouped_data.sample_groups)} independent "
    f"samples and {len(grouped_data.x)} repeated measurements; x shape={grouped_data.x.shape}."
)
for split in cv_splits:
    validation_label = (
        f"set {split.validation_set}"
        if split.validation_set is not None
        else f"S{split.validation_sample}"
    )
    print(
        f"Fold {split.fold + 1}: validation={validation_label}, "
        f"train={len(split.x_train)}, "
        f"validation={len(split.x_validation)} measurements"
    )


## Train one three-class concentration model per fold
importlib.reload(training)
trainer = training.BloodConcentrationTrainer(
    training.TrainerConfig(
        epochs=1000,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-2,
        early_stopping_metric="sample_accuracy",
        early_stopping_patience=150,
        verbose=True,
    )
)

fold_results = []
for split in cv_splits:
    print(f"Training fold {split.fold + 1}/{len(cv_splits)}")
    fold_results.append(trainer.fit(split))


## Inspect every fold separately
for fold_result in fold_results:
    validation_label = (
        f"set {fold_result.validation_set}"
        if fold_result.validation_set is not None
        else f"S{fold_result.validation_sample}"
    )
    fold_summary = summarize_results([fold_result])
    class_order = [
        fold_summary["idx_to_label"][index]
        for index in range(len(fold_summary["idx_to_label"]))
    ]

    print(
        f"\n{'=' * 72}\n"
        f"Fold {fold_result.fold + 1}: validation={validation_label}\n"
        f"Class order for all matrices: {class_order}"
    )
    print_result_summary(fold_summary)
    print("Measurement confusion matrix (rows=true, columns=predicted):")
    print(fold_summary["measurement_confusion_matrix"])
    print("Measurement recall matrix (row-normalized):")
    print(fold_summary["measurement_recall_matrix"])
    print("Sample confusion matrix (rows=true, columns=predicted):")
    print(fold_summary["sample_confusion_matrix"])
    print("Sample recall matrix (row-normalized):")
    print(fold_summary["sample_recall_matrix"])

    plot_recall_matrix_with_numbers(
        fold_summary,
        level="sample",
        title=f"Fold {fold_result.fold + 1} Sample Recall: {validation_label}",
    )


## Summarize all folds with pooled confusion and recall matrices
result_summary = summarize_results(fold_results)
class_order = [
    result_summary["idx_to_label"][index]
    for index in range(len(result_summary["idx_to_label"]))
]
print(
    f"\n{'=' * 72}\n"
    f"Pooled result across all {len(fold_results)} folds\n"
    f"Class order for all matrices: {class_order}"
)
print_result_summary(result_summary)
print("Pooled measurement confusion matrix (rows=true, columns=predicted):")
print(result_summary["measurement_confusion_matrix"])
print("Pooled measurement recall matrix (row-normalized):")
print(result_summary["measurement_recall_matrix"])
print("Pooled sample confusion matrix (rows=true, columns=predicted):")
print(result_summary["sample_confusion_matrix"])
print("Pooled sample recall matrix (row-normalized):")
print(result_summary["sample_recall_matrix"])

plot_recall_matrix_with_numbers(
    result_summary,
    level="sample",
    title=f"Pooled Sample Recall: {EVALUATION_METHOD}",
)
