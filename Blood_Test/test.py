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
    exclude_frequency_bands,
    load_blood_folder,
    plot_bacterial_set_differences,
    plot_difference_channel_samples,
    plot_difference_channel_summary,
    plot_filtering_comparison,
    plot_original_phase_set_differences,
    plot_pbs_set_differences,
    plot_pbs_representative_set_differences,
    plot_raw_signal_representatives,
    plot_representative_channel_samples,
    plot_representative_sets_by_concentration,
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

# Single channel selection for representative plots, processed-signal plots,
# and model input. Channel 8 = complex ratio phase; 10 = conjugate-product
# phase difference. The set-phase comparison plots require channel 3, 8, or 10.
CHANNEL_INDEX = 8

# Optional exclusions use: (sample type, set number, concentration).
# Use an empty tuple () to include every measurement.
EXCLUDED_DATA_BLOCKS = (
)

# Frequency bands use zero-based indices in the ORIGINAL 20,000-point signal,
# including both endpoints. Apply the same exclusions to every measurement.
# Start with the narrow peak near index 8890; edit/add bands as needed.
# Set EXCLUDED_FREQUENCY_BANDS = () to keep the complete signal.
EXCLUDED_FREQUENCY_BANDS = (
    (8800, 9000),
    # (10500, 10700),  # Example of a second band.
)
DOWNSAMPLE_RATIO = 100

# Choose "leave_one_sample_out" for eight folds (S1-S8), or
# "leave_one_set_out" for three folds (set 1-set 3).
EVALUATION_METHOD = "leave_one_set_out"

# Choose "own_set" for the original correction, or "other_sets_mean" so each
# set is corrected using the average PBS reference from the other two sets.
PBS_REFERENCE_MODE = "own_set"

raw_data = load_blood_folder(DATA_DIR)
raw_data = exclude_data_blocks(raw_data, excluded_blocks=EXCLUDED_DATA_BLOCKS)
print_blood_data_summary(raw_data)
# plot_raw_signal_representatives(raw_data)


## Filter real/imaginary signals and recompute amplitude/phase
filtered_data = apply_savgol_to_re_im_and_recompute(
    raw_data,
    window_length=31,
    polyorder=3,
)
print("Savitzky-Golay filtering completed: window_length=31, polyorder=3")
# plot_filtering_comparison(raw_data, filtered_data)
# plot_pbs_set_differences(filtered_data)
# plot_bacterial_set_differences(filtered_data)


## Compare bacterial set means after corresponding own-set PBS correction
# This diagnostic branch is deliberately independent of PBS_REFERENCE_MODE:
# Set 1 uses PBS 1, Set 2 uses PBS 2, and Set 3 uses PBS 3.
own_set_plain_difference_data = append_pbs_plain_difference(
    filtered_data,
    reference_mode="own_set",
)
own_set_complex_difference_data = append_pbs_complex_difference(
    own_set_plain_difference_data,
    reference_mode="own_set",
)
# plot_bacterial_set_differences(
#     own_set_complex_difference_data,
#     channel_indices=(4, 5, 9, 10),
#     comparison_label="Channels 4, 5, 9, 10 using corresponding own-set PBS",
# )


## Calculate set-specific PBS-referenced features
# Append pointwise [real, imaginary, amplitude, phase] differences using the
# selected own-set or cross-set PBS reference strategy.
plain_difference_data = append_pbs_plain_difference(
    filtered_data,
    reference_mode=PBS_REFERENCE_MODE,
)

# Append complex-valued comparison features using the selected PBS reference.
complex_difference_data = append_pbs_complex_difference(
    plain_difference_data,
    reference_mode=PBS_REFERENCE_MODE,
)

# Smooth all thirteen existing and PBS-derived channels along the frequency axis.
filtered_difference_data = apply_savgol_filter_nested(
    complex_difference_data,
    window_length=31,
    polyorder=3,
)
print(
    "PBS difference calculation and Savitzky-Golay filtering completed: "
    f"reference_mode={PBS_REFERENCE_MODE}"
)

# plot_difference_channel_samples(
#     complex_difference_data,
#     channel_indices=(CHANNEL_INDEX,),
#     samples_per_group=4,
#     random_seed=42,
# )
# plot_difference_channel_summary(
#     complex_difference_data,
#     channel_indices=(CHANNEL_INDEX,),
# )
#
# plot_difference_channel_samples(
#     filtered_difference_data,
#     channel_indices=(CHANNEL_INDEX,),
#     samples_per_group=4,
#     random_seed=42,
# )
# plot_difference_channel_summary(
#     filtered_difference_data,
#     channel_indices=(CHANNEL_INDEX,),
# )


## One closest-to-mean sample per concentration, with one figure per set
# Use CHANNEL_INDEX from filtered_difference_data, before downsampling. The
# function reports the selected biological sample and its distance to the mean.
representative_figures, representative_samples = plot_representative_channel_samples(
    filtered_difference_data,
    channel_index=CHANNEL_INDEX,
)


## Compare PBS differences and directly overlay the existing representatives
# PBS pairs share one plot. For representatives, plot Sets 1, 2, 3 together
# within each concentration, using the same stored curves without subtraction.
set_phase_comparison_figures, set_phase_differences = plot_pbs_representative_set_differences(
    filtered_data,
    representative_samples,
    include_representative_differences=False,
)
representative_set_comparison_figures = plot_representative_sets_by_concentration(
    representative_samples,
)


## Original filtered phase differences, reusing the same selected sample IDs
# Read channel 3 from filtered_data before PBS correction. One figure per
# concentration overlays Set 1-2, Set 1-3, Set 2-3. The selected channel above
# determines sample IDs only; this diagnostic always compares original phase.
original_phase_difference_figures, original_phase_differences = plot_original_phase_set_differences(
    filtered_data,
    representative_samples,
)


## Downsample the processed signals
# Keep every DOWNSAMPLE_RATIO-th frequency point, preserving every set, sample,
# repetition, and all thirteen signal channels.
downsampled_difference_data = downsample_nested_data(
    filtered_difference_data,
    ratio=DOWNSAMPLE_RATIO,
)
print(
    "Downsampling completed: "
    f"SAU 10-4 shape={downsampled_difference_data['SAU']['10-4'].shape}"
)


## Exclude selected frequency bands from the model input
# Keep the full processed arrays above for live plots and baseline comparison.
# Select AFTER smoothing/downsampling so retained frequency positions do not
# shift and the filters never operate across a newly created gap.
# retained_frequency_indices maps the selected input back to original indices.
selected_difference_data, retained_frequency_indices = exclude_frequency_bands(
    downsampled_difference_data,
    excluded_bands=EXCLUDED_FREQUENCY_BANDS,
    original_frequency_points=filtered_difference_data["SAU"]["10-4"].shape[-2],
    downsample_ratio=DOWNSAMPLE_RATIO,
)
print(
    f"Frequency selection: excluded bands={EXCLUDED_FREQUENCY_BANDS}; "
    f"kept {len(retained_frequency_indices)}/"
    f"{downsampled_difference_data['SAU']['10-4'].shape[-2]} downsampled points; "
    f"SAU 10-4 shape={selected_difference_data['SAU']['10-4'].shape}"
)


## Inspect the downsampled signals AFTER frequency-band exclusion
# One live figure per set, with PBS and concentrations in separate rows.
# Show all eight independent SAU samples, averaging repetitions per sample.
# Markers use consecutive selected-point indices (0-196 when three of 200
# points are removed); automatic y limits display the full peak depths.
downsampled_signal_figures = plot_difference_channel_samples(
    selected_difference_data,
    channel_indices=(CHANNEL_INDEX,),
    samples_per_group=8,
    frequency_step=1,
    marker="o",
    plot_label="After downsampling and frequency-band exclusion",
)


## Build the three-class concentration dataset and grouped CV splits
# A group is one independent (set, concentration, sample), containing 5 repetitions.
grouped_data = dataset.build_concentration_dataset(
    selected_difference_data,
    channel_indices=(CHANNEL_INDEX,),
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
