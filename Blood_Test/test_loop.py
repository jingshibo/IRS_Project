import gc
import importlib
import time
from pathlib import Path

import torch

from Functions import (
    append_pbs_complex_difference,
    append_pbs_plain_difference,
    apply_savgol_filter_nested,
    apply_savgol_to_re_im_and_recompute,
    downsample_nested_data,
    exclude_data_blocks,
    load_blood_folder,
    print_blood_data_summary,
)
from Model import dataset, training
from Model.channel_comparison import (
    CHANNEL_NAMES,
    build_channel_comparison_result,
    build_original_difference_combinations,
    build_single_channel_result,
    plot_channel_metric_heatmap,
    plot_single_channel_metric,
    print_ranked_channel_results,
    print_ranked_single_channel_results,
    save_channel_comparison_csv,
    save_single_channel_comparison_csv,
    sort_channel_comparison_results,
    sort_single_channel_results,
)
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

# Choose "leave_one_sample_out" for eight folds (S1-S8), or
# "leave_one_set_out" for three folds (set 1-set 3).
EVALUATION_METHOD = "leave_one_set_out"

# Enable either experiment independently. With leave-one-set-out, running both
# trains (24 + 10) x 3 = 102 models. Leave-one-sample-out trains 272 models.
RUN_TWO_CHANNEL_COMPARISON = True
RUN_SINGLE_CHANNEL_COMPARISON = True
SHOW_RESULT_PLOTS = True

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

RESULTS_ROOT = (
    Path(__file__).resolve().parent
    / "loop_results"
    / EVALUATION_METHOD
)
PAIR_RESULTS_DIR = RESULTS_ROOT / "two_channel_combinations"
PAIR_RECALL_DIR = PAIR_RESULTS_DIR / "recall_matrices"
PAIR_RESULTS_CSV = PAIR_RESULTS_DIR / "channel_comparison_results.csv"
SINGLE_RESULTS_DIR = RESULTS_ROOT / "individual_channels"
SINGLE_RECALL_DIR = SINGLE_RESULTS_DIR / "recall_matrices"
SINGLE_RESULTS_CSV = SINGLE_RESULTS_DIR / "single_channel_results.csv"


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
            output_path=(
                PAIR_RECALL_DIR
                / f"{original_channel}_{original_name}__"
                f"{difference_channel}_{difference_name}.png"
            ),
            show=False,
        )
        # Save after every pair so an interrupted run retains partial progress.
        save_channel_comparison_csv(pair_results, PAIR_RESULTS_CSV)

        del fold_results, result_summary, cv_splits, grouped_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if pair_results:
    ranked_pair_results = sort_channel_comparison_results(pair_results)
    print_ranked_channel_results(ranked_pair_results)
    save_channel_comparison_csv(ranked_pair_results, PAIR_RESULTS_CSV)
    plot_channel_metric_heatmap(
        ranked_pair_results,
        metric="pooled_sample_accuracy",
        output_path=PAIR_RESULTS_DIR / "sample_accuracy_heatmap.png",
        show=SHOW_RESULT_PLOTS,
    )
    plot_channel_metric_heatmap(
        ranked_pair_results,
        metric="sample_macro_f1",
        output_path=PAIR_RESULTS_DIR / "sample_macro_f1_heatmap.png",
        show=SHOW_RESULT_PLOTS,
    )


## Train and evaluate all 10 channels separately
single_channel_results = []
single_channels = tuple(sorted(CHANNEL_NAMES))
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
            output_path=SINGLE_RECALL_DIR / f"{channel}_{channel_name}.png",
            show=False,
        )
        # Save after every channel so an interrupted run retains partial progress.
        save_single_channel_comparison_csv(
            single_channel_results,
            SINGLE_RESULTS_CSV,
        )

        del fold_results, result_summary, cv_splits, grouped_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if single_channel_results:
    ranked_single_results = sort_single_channel_results(single_channel_results)
    print_ranked_single_channel_results(ranked_single_results)
    save_single_channel_comparison_csv(ranked_single_results, SINGLE_RESULTS_CSV)
    plot_single_channel_metric(
        ranked_single_results,
        metric="pooled_sample_accuracy",
        output_path=SINGLE_RESULTS_DIR / "sample_accuracy.png",
        show=SHOW_RESULT_PLOTS,
    )
    plot_single_channel_metric(
        ranked_single_results,
        metric="sample_macro_f1",
        output_path=SINGLE_RESULTS_DIR / "sample_macro_f1.png",
        show=SHOW_RESULT_PLOTS,
    )
