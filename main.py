##
import pandas as pd
from datetime import datetime
from Utility_Functions import Grid_Search, Model_Training, Preprocessing, Model_Structure
import importlib
import viewing


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
# Hampel filtering, removing outliers, followed by a sqrt transformation to supress large values
sliced_filtered_dict = Preprocessing.fast_spike_filter_dict(sliced_dict, radius=3, transform="sqrt", method="fast",
                                                            n_sigmas=3.0, k=4.0, min_threshold=1000.0)
# Smooth Original signals with Savitzky-Golay filter (before downsampling, which can cause aliasing). window_length default:31
original_filtered_dict = Preprocessing.apply_savgol_filter_dict(sliced_filtered_dict, window_length=31, polyorder=3, deriv=0, mode="mirror")
# Downsample after filtering, before SG derivative (performing on clean signal to avoid amplified noise due to derivative).
original_filtered_dict = Preprocessing.downsample_dict_signals(original_filtered_dict, step=1, offset=0)
# central difference calculate of each sample
central_diff_dict = Preprocessing.compute_central_diff_dict(original_filtered_dict)
# Smooth central-difference signals with Savitzky-Golay filter (row-wise). window_length default:201
central_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(central_diff_dict, window_length=31, polyorder=3, deriv=0, mode="mirror")
# second second difference calculate of each sample
second_diff_dict = Preprocessing.compute_second_central_diff_dict(original_filtered_dict)
# Smooth second-difference signals with Savitzky-Golay filter (row-wise). window_length default:201
second_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(second_diff_dict, window_length=31, polyorder=3, deriv=0, mode="mirror")
# rolling variance of the original filtered signal
rolling_variance_dict = Preprocessing.compute_rolling_variance_dict(original_filtered_dict, window_size=31)
# rolling energy of the derivative signal
derivative_energy_dict = Preprocessing.compute_derivative_energy_dict(central_diff_filtered_dict, window_size=31)
# envelope Original signals with Savitzky-Golay filter (for residual calculation).
original_envelope_dict = Preprocessing.apply_savgol_filter_dict(sliced_filtered_dict, window_length=201, polyorder=3, deriv=0)
# calculate the residual of the original signal (separating coarse and fine information)
original_residual_dict = Preprocessing.calculate_residual_dict(original_filtered_dict, original_envelope_dict)


## Split
RANDOM_SEED = 42
# Select which signal representations to use as model input channels.
# Options below are examples; any ordered combination is allowed.
value_type_dicts = {
    "original": original_filtered_dict,
    "first_diff_filtered": central_diff_filtered_dict,
    "second_diff_filtered": second_diff_filtered_dict,
    "rolling_variance": rolling_variance_dict,
    "derivative_energy": derivative_energy_dict,
    "residual": original_residual_dict
}
# e.g. ("original", "first_diff_filtered", "second_diff_filtered", "rolling_variance", "derivative_energy", "residual")
selected_value_types = ["original", "first_diff_filtered", "second_diff_filtered"]

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
    clip_max_value=None,
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


## Optional plotting
RANDOM_SEED = 42
PLOT_OPTIONS = {
    "threshold_hits": False,
    "classification_examples": False,
    "reference_samples": True,
    "mean_std_overview": False,
    "random_sample_overview": True,
    "normalized_data_inspection": False,
}

PLOT_CONFIG = {
    "threshold_hits": {
        "channel_idx": 0,
        "threshold": 25.0,
        "split": None,
        "nrows": 4,
        "ncols": 4,
        "title_prefix": "Normalized threshold-hit signals",
    },
    "classification_examples": {
        "classes": class_order,
        "top_k": 10,
        "channel_idx": 0,
    },
    "reference_samples": {
        "fold_id": 4,
        "sample_idx": 136,
        "channel_idx": 0,
    },
    "random_sample_overview": {
        "classes": ("LOW",), # "LOW", "TARGET", "HIGH"
        "n_samples": 30,
        "ncols": 6,
    },
    "normalized_data_inspection": {
        "fold_id": 0,
        "n_samples": 5,
    },
}

if PLOT_OPTIONS["threshold_hits"]:
    viewing.plot_threshold_hits(
        cv_folds=cv_folds,
        **PLOT_CONFIG["threshold_hits"],
    )

if PLOT_OPTIONS["classification_examples"]:
    viewing.plot_classification_examples(
        train_out=train_out,
        cv_folds=cv_folds,
        class_order=class_order,
        **PLOT_CONFIG["classification_examples"],
    )

if PLOT_OPTIONS["reference_samples"]:
    viewing.plot_certain_samples(
        sliced_dict=sliced_dict,
        sliced_filtered_dict=sliced_filtered_dict,
        x_trainval=X_trainval,
        y_trainval=y_trainval,
        cv_folds=cv_folds,
        train_out=train_out,
        **PLOT_CONFIG["reference_samples"],
    )

if PLOT_OPTIONS["mean_std_overview"]:
    viewing.plot_mean_std_overview(
        class_order=class_order,
        original_dict=sliced_dict,
        original_filtered_dict=original_filtered_dict,
        central_diff_dict=central_diff_dict,
        central_diff_filtered_dict=central_diff_filtered_dict,
        second_diff_dict=second_diff_dict,
        second_diff_filtered_dict=second_diff_filtered_dict,
        rolling_variance_dict=rolling_variance_dict,
        derivative_energy_dict=derivative_energy_dict,
    )

if PLOT_OPTIONS["random_sample_overview"]:
    viewing.plot_random_sample_overview(
        class_order=class_order,
        random_seed=RANDOM_SEED,
        sliced_dict=sliced_dict,
        sliced_filtered_dict=sliced_filtered_dict,
        original_filtered_dict=original_filtered_dict,
        original_envelope_dict=original_envelope_dict,
        original_residual_dict=original_residual_dict,
        central_diff_dict=central_diff_dict,
        central_diff_filtered_dict=central_diff_filtered_dict,
        second_diff_dict=second_diff_dict,
        second_diff_filtered_dict=second_diff_filtered_dict,
        rolling_variance_dict=rolling_variance_dict,
        derivative_energy_dict=derivative_energy_dict,
        **PLOT_CONFIG["random_sample_overview"],
    )

if PLOT_OPTIONS["normalized_data_inspection"]:
    viewing.inspect_normalized_data(
        cv_folds=cv_folds,
        class_order=class_order,
        selected_value_types=selected_value_types,
        **PLOT_CONFIG["normalized_data_inspection"],
    )


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
        model_names=("shared_backbone_2ch",), # canonical options: "shared_backbone_2ch", "two_tower_late_fusion", "two_tower_mid_fusion_cnn", "tcn_classifier"
        class_order=class_order,
        epochs=100,
        patience=25,
        tensorboard_log_dir_root=None, # disabled
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

