import gc
import importlib
import time
from pathlib import Path

import torch

from Blood_Test.Functions import (
    append_pbs_complex_difference,
    append_pbs_plain_difference,
    apply_savgol_filter_nested,
    apply_savgol_to_re_im_and_recompute,
    downsample_nested_data,
    exclude_data_blocks,
    load_blood_folder,
    print_blood_data_summary,
)
from Blood_Test.Model import dataset, training
from Blood_Test.Model.channel_comparison import (
    CHANNEL_NAMES,
    build_channel_comparison_result,
    build_difference_pair_combinations,
    build_difference_pair_result,
    build_original_difference_combinations,
    build_single_channel_result,
    plot_channel_metric_heatmap,
    plot_difference_pair_metric_heatmap,
    plot_single_channel_metric,
    print_ranked_difference_pair_results,
    print_ranked_channel_results,
    print_ranked_single_channel_results,
    sort_channel_comparison_results,
    sort_difference_pair_results,
    sort_single_channel_results,
)
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
EVALUATION_METHOD = "leave_one_sample_out"

# Choose "own_set" for the original correction, or "other_sets_mean" so each
# set is corrected using the average PBS reference from the other two sets.
PBS_REFERENCE_MODE = "other_sets_mean"

# Enable each experiment independently. The new difference-pair mode evaluates
# all C(7, 2) = 21 unique pairs formed from channels 4 through 10.
RUN_TWO_CHANNEL_COMPARISON = False
RUN_DIFFERENCE_PAIR_COMPARISON = True
RUN_SINGLE_CHANNEL_COMPARISON = False
SHOW_RESULT_PLOTS = True
# Display figures live; result lists and matrices remain in console memory.

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
# Append pointwise [real, imaginary, amplitude, phase] differences using the
# selected own-set or cross-set PBS reference strategy.
plain_difference_data = append_pbs_plain_difference(
    filtered_data,
    reference_mode=PBS_REFERENCE_MODE,
)

# Append two complex-valued comparison features using the same set's mean PBS:
# the phase of (SAU / PBS) and the magnitude of (SAU - PBS).
complex_difference_data = append_pbs_complex_difference(
    plain_difference_data,
    reference_mode=PBS_REFERENCE_MODE,
)

# Smooth all ten existing and PBS-derived channels along the frequency axis.
filtered_difference_data = apply_savgol_filter_nested(
    complex_difference_data,
    window_length=31,
    polyorder=3,
)
print(
    "PBS difference calculation and Savitzky-Golay filtering completed: "
    f"reference_mode={PBS_REFERENCE_MODE}"
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


## Configure the comparison experiments
if EVALUATION_METHOD not in {"leave_one_sample_out", "leave_one_set_out"}:
    raise ValueError(
        "EVALUATION_METHOD must be 'leave_one_sample_out' or "
        f"'leave_one_set_out', got {EVALUATION_METHOD!r}"
    )

def build_selected_channel_splits(channel_indices):
    """Build one dataset and the evaluation folds selected above."""
    selected_data = dataset.build_concentration_dataset(
        downsampled_difference_data,
        channel_indices=channel_indices,
    )
    if EVALUATION_METHOD == "leave_one_sample_out":
        splits = dataset.build_leave_one_sample_out_splits(selected_data)
    else:
        splits = dataset.build_leave_one_set_out_splits(selected_data)
    return selected_data, splits


def validation_name(split):
    """Return a readable label for the held-out sample index or set."""
    if split.validation_set is not None:
        return f"set {split.validation_set}"
    return f"S{split.validation_sample}"


## Use the same model settings for every channel selection
importlib.reload(training)
trainer = training.BloodConcentrationTrainer(
    training.TrainerConfig(
        epochs=1000,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-2,
        early_stopping_metric="sample_accuracy",
        early_stopping_patience=150,
        # The outer loops report progress, avoiding epoch output from 102/272 models.
        verbose=False,
    )
)


## Train and evaluate the 24 original-channel + difference-channel pairs
pair_results = []
channel_combinations = build_original_difference_combinations()
if RUN_TWO_CHANNEL_COMPARISON:
    for combination_number, (original_channel, difference_channel) in enumerate(
        channel_combinations,
        start=1,
    ):
        selected_channels = (original_channel, difference_channel)
        original_name = CHANNEL_NAMES[original_channel]
        difference_name = CHANNEL_NAMES[difference_channel]
        started_at = time.perf_counter()
        print(
            f"\nCombination {combination_number}/{len(channel_combinations)}: "
            f"{selected_channels} = {original_name} + {difference_name}"
        )

        grouped_data, cv_splits = build_selected_channel_splits(selected_channels)
        fold_results = []
        for split in cv_splits:
            print(
                f"  fold {split.fold + 1}/{len(cv_splits)} "
                f"(validation={validation_name(split)})"
            )
            fold_results.append(trainer.fit(split))

        result_summary = summarize_results(fold_results)
        elapsed_seconds = time.perf_counter() - started_at
        pair_results.append(
            build_channel_comparison_result(
                original_channel=original_channel,
                difference_channel=difference_channel,
                summary=result_summary,
                elapsed_seconds=elapsed_seconds,
            )
        )
        print_result_summary(result_summary)
        print(f"Elapsed time: {elapsed_seconds / 60.0:.1f} minutes")

        plot_recall_matrix_with_numbers(
            result_summary,
            level="sample",
            title=f"Sample Recall: {original_name} + {difference_name}",
            show=SHOW_RESULT_PLOTS,
        )

        del fold_results, result_summary, cv_splits, grouped_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if pair_results:
    ranked_pair_results = sort_channel_comparison_results(pair_results)
    print_ranked_channel_results(ranked_pair_results)
    plot_channel_metric_heatmap(
        ranked_pair_results,
        metric="pooled_sample_accuracy",
        show=SHOW_RESULT_PLOTS,
    )
    plot_channel_metric_heatmap(
        ranked_pair_results,
        metric="sample_macro_f1",
        show=SHOW_RESULT_PLOTS,
    )


## Train and evaluate all 21 unique pairs of difference channels 4 through 10
difference_pair_results = []
difference_pair_combinations = build_difference_pair_combinations(
    difference_channels=tuple(range(4, 11)),
)
if RUN_DIFFERENCE_PAIR_COMPARISON:
    for combination_number, (first_channel, second_channel) in enumerate(
        difference_pair_combinations,
        start=1,
    ):
        selected_channels = (first_channel, second_channel)
        first_name = CHANNEL_NAMES[first_channel]
        second_name = CHANNEL_NAMES[second_channel]
        started_at = time.perf_counter()
        print(
            f"\nDifference pair {combination_number}/"
            f"{len(difference_pair_combinations)}: "
            f"{selected_channels} = {first_name} + {second_name}"
        )

        grouped_data, cv_splits = build_selected_channel_splits(selected_channels)
        fold_results = []
        for split in cv_splits:
            print(
                f"  fold {split.fold + 1}/{len(cv_splits)} "
                f"(validation={validation_name(split)})"
            )
            fold_results.append(trainer.fit(split))

        result_summary = summarize_results(fold_results)
        elapsed_seconds = time.perf_counter() - started_at
        difference_pair_results.append(
            build_difference_pair_result(
                first_channel=first_channel,
                second_channel=second_channel,
                summary=result_summary,
                elapsed_seconds=elapsed_seconds,
            )
        )
        print_result_summary(result_summary)
        print(f"Elapsed time: {elapsed_seconds / 60.0:.1f} minutes")

        plot_recall_matrix_with_numbers(
            result_summary,
            level="sample",
            title=f"Sample Recall: {first_name} + {second_name}",
            show=SHOW_RESULT_PLOTS,
        )

        del fold_results, result_summary, cv_splits, grouped_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if difference_pair_results:
    ranked_difference_pair_results = sort_difference_pair_results(
        difference_pair_results
    )
    print_ranked_difference_pair_results(ranked_difference_pair_results)
    plot_difference_pair_metric_heatmap(
        ranked_difference_pair_results,
        metric="pooled_sample_accuracy",
        show=SHOW_RESULT_PLOTS,
    )
    plot_difference_pair_metric_heatmap(
        ranked_difference_pair_results,
        metric="sample_macro_f1",
        show=SHOW_RESULT_PLOTS,
    )


## Train and evaluate all 10 channels separately
single_channel_results = []
single_channels = tuple(range(10))
if RUN_SINGLE_CHANNEL_COMPARISON:
    for channel_number, channel in enumerate(single_channels, start=1):
        channel_name = CHANNEL_NAMES[channel]
        started_at = time.perf_counter()
        print(
            f"\nSingle channel {channel_number}/{len(single_channels)}: "
            f"{channel} = {channel_name}"
        )

        grouped_data, cv_splits = build_selected_channel_splits((channel,))
        fold_results = []
        for split in cv_splits:
            print(
                f"  fold {split.fold + 1}/{len(cv_splits)} "
                f"(validation={validation_name(split)})"
            )
            fold_results.append(trainer.fit(split))

        result_summary = summarize_results(fold_results)
        elapsed_seconds = time.perf_counter() - started_at
        single_channel_results.append(
            build_single_channel_result(
                channel=channel,
                summary=result_summary,
                elapsed_seconds=elapsed_seconds,
            )
        )
        print_result_summary(result_summary)
        print(f"Elapsed time: {elapsed_seconds / 60.0:.1f} minutes")

        plot_recall_matrix_with_numbers(
            result_summary,
            level="sample",
            title=f"Sample Recall: {channel_name}",
            show=SHOW_RESULT_PLOTS,
        )

        del fold_results, result_summary, cv_splits, grouped_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if single_channel_results:
    ranked_single_results = sort_single_channel_results(single_channel_results)
    print_ranked_single_channel_results(ranked_single_results)
    plot_single_channel_metric(
        ranked_single_results,
        metric="pooled_sample_accuracy",
        show=SHOW_RESULT_PLOTS,
    )
    plot_single_channel_metric(
        ranked_single_results,
        metric="sample_macro_f1",
        show=SHOW_RESULT_PLOTS,
    )
