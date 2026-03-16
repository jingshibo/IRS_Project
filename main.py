##
import pandas as pd
from datetime import datetime
from Utility_Functions import Grid_Search, Model_Training, Plotting_Functions, Preprocessing, Model_Structure
import importlib


## Load data
excel_path = "/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"
df = pd.read_excel(excel_path, sheet_name=0)  # or sheet_name="Sheet1"
# assume the first column is the label column
label_col = df.columns[0]
class_order = ("LOW", "TARGET", "HIGH")
categorized_dict = {key: group.drop(columns=[label_col]).reset_index(drop=True) for key, group in df.groupby(label_col)}
# slice segment
signal_segments = ((000, 1000), (1800, 3500)) # Slice raw signals before preprocessing signal_segments = ((000, 4000),).
# signal_segments = ((0, 4000),)
sliced_dict = Preprocessing.slice_dict_signal_segments(categorized_dict, segments=signal_segments)


## Preprocessing
# Downsample after slicing, before any derivative/filter preprocessing.
original_dict = Preprocessing.downsample_dict_signals(sliced_dict, step=1, offset=0)
# Smooth Original signals with Savitzky-Golay filter (not necessary but better).
original_filtered_dict = Preprocessing.apply_savgol_filter_dict(original_dict, window_length=51, polyorder=3, deriv=0)
# central difference calculate of each sample
central_diff_dict = Preprocessing.compute_central_diff_dict(original_filtered_dict)
# Smooth central-difference signals with Savitzky-Golay filter (row-wise).
central_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(central_diff_dict, window_length=201, polyorder=3, deriv=0)
# second central difference calculate of each sample
second_diff_dict = Preprocessing.compute_second_central_diff_dict(original_dict)
# Smooth central-difference signals with Savitzky-Golay filter (row-wise).
second_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(second_diff_dict, window_length=51, polyorder=3, deriv=0)

# Row-wise average for each sample in each class group.
original_stats = Preprocessing.compute_mean_std_stats(original_dict)
# Mean/std stats for original, same structure as overall_stats.
original_filtered_stats = Preprocessing.compute_mean_std_stats(original_filtered_dict)
# Mean/std stats for central differences, same structure as overall_stats.
central_diff_stats = Preprocessing.compute_mean_std_stats(central_diff_dict)
# Mean/std stats for central differences, same structure as overall_stats.
central_diff_filtered_stats = Preprocessing.compute_mean_std_stats(central_diff_filtered_dict)
# Mean/std stats for second central differences.
second_diff_stats = Preprocessing.compute_mean_std_stats(second_diff_dict)
# Mean/std stats for filtered second central differences.
second_diff_filtered_stats = Preprocessing.compute_mean_std_stats(second_diff_filtered_dict)


## Split
RANDOM_SEED = 42
# Select which signal representations to use as model input channels.
# Options below are examples; any ordered combination is allowed.
value_type_dicts = {
    "original": original_filtered_dict,
    "first_diff_filtered": central_diff_filtered_dict,
    "second_diff_filtered": second_diff_filtered_dict,
}
selected_value_types = ["original", "first_diff_filtered"]  # e.g. ("original", "first_diff_filtered", "second_diff_filtered")

x_all, y_all = Preprocessing.build_multi_channel_dataset(
    data_dict_map=value_type_dicts,
    selected_types=selected_value_types,
)
# Step 1: random holdout 15% test set.
X_trainval, X_test, y_trainval, y_test = Preprocessing.split_holdout(
    x_all,
    y_all,
    test_size=0.15,
    random_seed=RANDOM_SEED,
)
# Step 2: stratified 5-fold on remaining 85% + fold-wise normalization.
cv_folds = Preprocessing.build_normalized_cv_folds(
    X_trainval,
    y_trainval,
    n_splits=5,
    random_seed=RANDOM_SEED,
)


## Model Training
importlib.reload(Model_Structure)
importlib.reload(Model_Training)
model_name = "shared_backbone_2ch"  # canonical options: "shared_backbone_2ch", "multi_scale_1d_cnn", "two_tower_late_fusion", "two_tower_mid_fusion_cnn", "tcn_classifier"
# train_out: Classification_Models.TrainOutput means when train_out is assigned, it should match the type TrainOutput (only for IDE to check).
train_out: Model_Training.TrainOutput = Model_Training.train_1d_cnn_cv(
    cv_folds=cv_folds,
    class_order=class_order,
    model_name=model_name,
    epochs=100,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    label_smoothing=0.3,
    patience=25,
    random_shift_max_points=5,
    random_shift_fill_mode="wrap",
    tensorboard_log_dir="runs/1d_cnn_cv",
)
print("Mean best val acc:", train_out["mean_best_val_acc"])
print("Class index mapping:", train_out["label_to_idx"])
print("Model name:", model_name)



## Plot class mean-std curves.
# # Plot original data mean-std curves
# Plotting_Functions.plot_mean_std_curves(original_stats, class_order=class_order, title="Class Mean-Std Curves", ylabel="Mean value")
#
# # Plot filtered original data mean-std curves
# Plotting_Functions.plot_mean_std_curves(original_filtered_stats, class_order=class_order, title="Filtered Class Mean-Std Curves", ylabel="Mean value")
#
# # Plot raw central-difference mean-std curves.
# Plotting_Functions.plot_mean_std_curves(central_diff_stats, class_order=class_order, title="Central Difference Mean-Std Curves",
#                                         ylabel="Central difference value", ylim=(-15, 15))
#
# # Plot filtered central-difference mean-std curves.
# Plotting_Functions.plot_mean_std_curves(central_diff_filtered_stats, class_order=class_order,
#                                         title="Filtered Central Difference Mean-Std Curves", ylabel="Central difference value", ylim=(-5, 5))
#
# # Plot raw second-difference mean-std curves.
# Plotting_Functions.plot_mean_std_curves(second_diff_stats, class_order=class_order,
#                                         title="Second Difference Mean-Std Curves",
#                                         ylabel="Second difference value", ylim=(-10, 10))
#
# # Plot filtered second-difference mean-std curves.
# Plotting_Functions.plot_mean_std_curves(second_diff_filtered_stats, class_order=class_order,
#                                         title="Filtered Second Difference Mean-Std Curves",
#                                         ylabel="Second difference value", ylim=(-1, 1))


## Plot example samples
# # Plot five original samples per class in three vertical subplots.
# Plotting_Functions.plot_class_samples_vertical(original_dict, class_order=class_order, n_samples=5,
#                                                title="Original Samples by Class", ylabel="Original value")
#
# # Plot five filtered original samples per class in three vertical subplots.
# Plotting_Functions.plot_class_samples_vertical(original_filtered_dict, class_order=class_order, n_samples=5,
#                                                title="Original Samples by Class", ylabel="Original value")
#
# # Plot five central-difference samples per class in three vertical subplots.
# Plotting_Functions.plot_class_samples_vertical(central_diff_dict, class_order=class_order, n_samples=5,
#                                                title="Central-Difference Samples by Class", ylabel="Central difference value")
#
# # Plot five filtered central-difference samples per class in three vertical subplots.
# Plotting_Functions.plot_class_samples_vertical(central_diff_filtered_dict, class_order=class_order, n_samples=5,
#                                                title="Filtered Central-Difference Samples by Class", ylabel="Central difference value")
#
# # Plot five second-difference samples per class in three vertical subplots.
# Plotting_Functions.plot_class_samples_vertical(second_diff_dict, class_order=class_order, n_samples=5,
#                                                title="Second-Difference Samples by Class", ylabel="Second difference value")
#
# # Plot five filtered second-difference samples per class in three vertical subplots.
# Plotting_Functions.plot_class_samples_vertical(second_diff_filtered_dict, class_order=class_order, n_samples=5,
#                                                title="Filtered Second-Difference Samples by Class", ylabel="Second difference value")
#
# Example: plot the first row from the original dataframe (excluding label column)
# Plotting_Functions.plot_single_sample(df, row_index=0, label_column=label_col)





##  Inspect normalized data
# #  from the first CV fold.
# first_fold = cv_folds[0]
# X_train_norm = first_fold["X_train"]
# y_train_norm = first_fold["y_train"]
#
# for ch_idx, value_type_name in enumerate(selected_value_types):
#     normalized_channel_dict = {}
#     for cls in class_order:
#         class_mask = y_train_norm == cls
#         if class_mask.any():
#             normalized_channel_dict[cls] = pd.DataFrame(X_train_norm[class_mask, ch_idx, :])
#
#     normalized_channel_stats = Preprocessing.compute_mean_std_stats(normalized_channel_dict)
#     pretty_name = value_type_name.replace("_", " ").title()
#
#     Plotting_Functions.plot_mean_std_curves(
#         normalized_channel_stats,
#         class_order=class_order,
#         title=f"Normalized {pretty_name} Mean-Std Curves (Fold 0 Train)",
#         ylabel=f"Normalized {value_type_name} value",
#     )
#
#     Plotting_Functions.plot_class_samples_vertical(
#         normalized_channel_dict,
#         class_order=class_order,
#         n_samples=5,
#         title=f"Normalized {pretty_name} Samples by Class (Fold 0 Train)",
#         ylabel=f"Normalized {value_type_name} value",
#     )



# Which value in each combination occurring the most
## Grid Search (optional)
RUN_GRID_SEARCH = False
if RUN_GRID_SEARCH:
    importlib.reload(Grid_Search)
    grid_search_space = {
        "base_channels": [32],  # no difference between 32 and 64, better than 16
        "kernel_size": [5],  # no difference between 5 and 7, better than 3, 9. 50 is better than others, seems larger is better? no much differences
        "l1": [512],  # the combination of (512, 256) is close to (1024, 128), better than others
        "l2": [256],
        "weight_decay": [1e-4],  # no difference from 1e-3 to 1e-5
        "dropout": [0.1],  # 0 is better than 0.3 and much better than 0.5
        "label_smoothing": [0.3],  # higher than 0.1 is better, no difference between 0.2 and 0.3
        "lr": [1e-4],  # 1e-4 is better than other higher or lower values
        "batch_size": [32],  # 32 is better than 16 and larger values
        "conv_stride": [2],
        "conv_dilation": [1],  # 1 is better than 2
        "avgpool_kernel_size": [3], # 3 is much better than 2
        "avgpool_stride": [2],
        "pool_type": ["max"],  # max is better than avg
        "leaky_relu_alpha": [0.05],  # no obvious differece between the three
        "scheduler_factor": [0.7],  # 0.7 is much better than 0.5 and 0.3
    }
    # Two-tower models require exactly 2 channels. Current selected_value_types should match that.
    grid_out = Grid_Search.run_grid_search_three_models(
        cv_folds=cv_folds,
        search_space=grid_search_space,
        model_names=("shared_backbone_2ch",),  # canonical options: "shared_backbone_2ch", "two_tower_late_fusion", "two_tower_mid_fusion_cnn", "tcn_classifier"
        class_order=class_order,
        epochs=100,
        patience=25,
        tensorboard_log_dir_root=None,  # disabled
        verbose_train=False,
        print_progress=True,
    )
    Grid_Search.print_top_grid_results(grid_out, top_k=100)
    grid_result_dir = f"runs/grid_search_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saved_paths = Grid_Search.save_grid_search_results(
        grid_out,
        output_dir=grid_result_dir,
        prefix="three_models",
    )
    print("Saved grid-search results:", saved_paths)

    # print output value summaries

    n_above_095 = sum(trial.mean_best_val_acc > 0.950 for trial in grid_out["valid_trials"])
    print(f"Number of valid grid-search results with mean_best_val_acc > 0.95: {n_above_095}")
    grid_top100_param_freq = Grid_Search.summarize_top_param_frequencies(
        grid_out,
        top_k=100,
        include_model_name=True,
    )
    print("Top-100 parameter frequency summary dict:", grid_top100_param_freq)
    Grid_Search.print_top_param_frequencies(grid_out, top_k=100, include_model_name=True)
    print("Best grid-search trial:", grid_out["best_trial"])
