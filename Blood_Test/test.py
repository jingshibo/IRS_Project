import importlib
from pathlib import Path

from Functions import (
    append_pbs_complex_difference,
    append_pbs_reference_difference,
    apply_savgol_filter_nested,
    apply_savgol_to_re_im_and_recompute,
    downsample_nested_data,
    exclude_data_blocks,
    load_blood_folder,
    print_blood_data_summary,
)
from Model import dataset, training
from Model.evaluation import (
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

raw_data = load_blood_folder(DATA_DIR)
raw_data = exclude_data_blocks(
    raw_data,
    excluded_blocks=EXCLUDED_DATA_BLOCKS,
)
print_blood_data_summary(raw_data)


## Filter real/imaginary signals and recompute amplitude/phase
filtered_data = apply_savgol_to_re_im_and_recompute(
    raw_data,
    window_length=31,
    polyorder=3,
)
print("Savitzky-Golay filtering completed: window_length=31, polyorder=3")


## Calculate set-specific PBS-referenced features
# For each set, average its five PBS repetitions and append the pointwise
# [real, imaginary, amplitude, phase] differences from that set's mean PBS trace.
difference_data = append_pbs_reference_difference(filtered_data)

# Append two complex-valued comparison features using the same set's mean PBS:
# the phase of (SAU / PBS) and the magnitude of (SAU - PBS).
difference_data = append_pbs_complex_difference(difference_data)

# Smooth all ten existing and PBS-derived channels along the frequency axis.
filtered_difference_data = apply_savgol_filter_nested(
    difference_data,
    window_length=31,
    polyorder=3,
)
print("PBS difference calculation and Savitzky-Golay filtering completed")


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
# Use complex subtraction magnitude (channel 7) as the initial single-channel input.
# A group is one independent (set, concentration, sample), containing 5 repetitions.
data = dataset.build_concentration_dataset(
    downsampled_difference_data,
    channel_indices=(8,), ## 8 is the best one
)

# In each of 8 folds, one sample index is held out for validation and the other
# seven are used for training. Normalization uses only the training partition.
cv_splits = dataset.build_leave_one_sample_out_splits(data)

print(
    f"Built {len(cv_splits)} folds from {len(data.sample_groups)} independent "
    f"samples and {len(data.x)} repeated measurements; x shape={data.x.shape}."
)
for split in cv_splits:
    print(
        f"Fold {split.fold + 1}: validation=S{split.validation_sample}, "
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


## Summarize measurement-level and five-repetition-averaged sample results
result_summary = summarize_results(fold_results)
print_result_summary(result_summary)
plot_recall_matrix_with_numbers(result_summary, level="sample")
