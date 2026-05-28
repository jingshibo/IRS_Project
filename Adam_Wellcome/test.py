##
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import importlib
from Functions import (
    append_pbs_complex_difference,
    append_pbs_reference_difference,
    apply_savgol_filter_nested,
    apply_savgol_to_re_im_and_recompute,
    downsample_nested_data,
    load_liquid_folder,
    plot_all_liquids,
    plot_selected_liquids_concentration_overlay,
    plot_selected_concentrations_by_liquid,
    plot_selected_concentrations_overlay,
    plot_selected_concentrations_stats_overlay,
)
from Model import dataset, training, model
from Model.Result_Processing import (
    plot_grouped_recall_matrix,
    plot_grouped_recall_matrix_with_numbers,
    plot_selected_confusion_matrix,
    print_result_summary,
    summarize_fold_results,
)

## load data
DATA_DIR = Path(r"/home/shibojing/data/adam wellcome")
# read raw data and calculate amp and phase based on raw data
raw_data = load_liquid_folder(DATA_DIR)
# filter re and im data for more stable amp/phase calculation (phase is very sensitive to sign variations in re/im)
filtered_data = apply_savgol_to_re_im_and_recompute(
    raw_data,
    window_length=301,
    polyorder=3,
)
# plot_all_liquids(filtered_data, channel_idx=2)

## preprocessing
# calculate difference between each liquid and the PBS reference.
difference_data = append_pbs_reference_difference(filtered_data)
difference_data = append_pbs_complex_difference(difference_data)
# filtering difference data
filtered_difference_data = apply_savgol_filter_nested(
    difference_data,
    window_length=301,
    polyorder=3,
)
# downsample raw data
downsample_ratio = 100
downsampled_difference_data = downsample_nested_data(
    filtered_difference_data,
    ratio=downsample_ratio,
)

##
# plot_all_liquids(filtered_difference_data, channel_idx=9)
#
# selected_concentrations = ["10-10", "10-5", "10-1"]
# # plot_selected_concentrations_by_liquid(
# #     filtered_difference_data,
# #     selected_concentrations,
# #     channel_idx=9,
# # )
# plot_selected_concentrations_overlay(
#     filtered_difference_data,
#     selected_concentrations,
#     channel_idx=9,
# )
#
# # plot_selected_concentrations_stats_overlay(
# #     filtered_difference_data,
# #     selected_concentrations,
# #     channel_idx=9,
# #     error_mode="sd",
# # )
#
# selected_liquids = ["Abau", "ECOLI", "FS1061", "FS1430", "FS1431"]
#
# plot_selected_liquids_concentration_overlay(
#     filtered_difference_data,
#     selected_liquids,
#     channel_idx=9,
# )

##
RANDOM_SEED = 42
data = dataset.build_classification_dataset(
    downsampled_difference_data,
    channel_indices=(9, ), ## how about only amplitude? actually worse than phase and both amp and phase.
    label_mode="joint",
    include_reference=False,
)
cv_splits = dataset.build_stratified_cv_splits(
    data,
    test_ratio=0.15,
    n_splits=5,
    random_seed=RANDOM_SEED,
)

##
importlib.reload(model)
importlib.reload(training)
# initialize the trainer object with desired configuration
trainer = training.AdamWellcomeTrainer(
    training.TrainerConfig(
        epochs=1000,
        batch_size=128,
        patience=300,
        verbose=True,
        lr_scheduler_name="cosine", # options: CosineAnnealingLR, ReduceLROnPlateau, ExponentialLR
        final_model_method="best", # options: best, top_k_average, swa, best_window_average
    )
)
# train models
fold_results = []
for split in cv_splits:
    print(f"Fold {split.fold + 1}/{len(cv_splits)}")
    fold_results.append(trainer.fit(split))

# print results
result_summary = summarize_fold_results(fold_results)
# print_result_summary(result_summary)

## Example plot calls:
plot_grouped_recall_matrix(result_summary, split="test")
# plot_grouped_recall_matrix_with_numbers(result_summary, split="test")
# plot_selected_confusion_matrix(result_summary, "test_recall", liquid="Abau") # Abau, ECOLI, FS1061, FS1430, FS1431
