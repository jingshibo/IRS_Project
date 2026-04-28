import importlib
from datetime import datetime
import pandas as pd
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Raw_Data_Implementation import Grid_Search, Model_Training, Model_Structure
from Utility_Functions import Preprocessing, Viewing


## load data
excel_path = "/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"
df = pd.read_excel(excel_path, sheet_name=0)
label_col = df.columns[0]
class_order = ("LOW", "TARGET", "HIGH")
categorized_dict = {key: group.drop(columns=[label_col]).reset_index(drop=True) for key, group in df.groupby(label_col)}


## preprocessing
signal_segments = ((0, 1000), (1800, 3500))
sliced_dict = Preprocessing.slice_dict_signal_segments(categorized_dict, segments=signal_segments)
sliced_filtered_dict = Preprocessing.fast_spike_filter_dict(
    sliced_dict,
    radius=3,
    transform="sqrt",
    method="fast",
    n_sigmas=3.0,
    k=4.0,
    min_threshold=1000.0,
)
original_filtered_dict = Preprocessing.apply_savgol_filter_dict(
    sliced_filtered_dict,
    window_length=31,
    polyorder=3,
    deriv=0,
    mode="mirror",
)
original_filtered_dict = Preprocessing.downsample_dict_signals(original_filtered_dict, step=5, offset=0)
central_diff_dict = Preprocessing.compute_central_diff_dict(original_filtered_dict)
central_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(
    central_diff_dict,
    window_length=31,
    polyorder=3,
    deriv=0,
    mode="mirror",
)
second_diff_dict = Preprocessing.compute_second_central_diff_dict(original_filtered_dict)
second_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(
    second_diff_dict,
    window_length=31,
    polyorder=3,
    deriv=0,
    mode="mirror",
)

rolling_variance_dict = Preprocessing.compute_rolling_variance_dict(original_filtered_dict, window_size=31)
derivative_energy_dict = Preprocessing.compute_derivative_energy_dict(central_diff_filtered_dict, window_size=31)
original_envelope_dict = Preprocessing.apply_savgol_filter_dict(original_filtered_dict, window_length=31, polyorder=3, deriv=0)
original_residual_dict = Preprocessing.calculate_residual_dict(original_filtered_dict, original_envelope_dict)


## create dataset
random_seed = 42
value_type_dicts = {
    "original": original_filtered_dict,
    "first_diff_filtered": central_diff_filtered_dict,
    "second_diff_filtered": second_diff_filtered_dict,
    "rolling_variance": rolling_variance_dict,
    "derivative_energy": derivative_energy_dict,
    "residual": original_residual_dict,
}
selected_value_types = ["original", "first_diff_filtered", "second_diff_filtered"]

x_all, y_all = Preprocessing.build_multi_channel_dataset(
    data_dict_map=value_type_dicts,
    selected_types=selected_value_types,
)

x_trainval, x_test, y_trainval, y_test = Preprocessing.split_holdout(
    x_all,
    y_all,
    test_size=0.15,
    random_seed=random_seed,
)

cv_folds = Preprocessing.build_normalized_cv_folds(
    x_trainval,
    y_trainval,
    n_splits=5,
    random_seed=random_seed,
    clip_max_value=None,
)


## model training
importlib.reload(Model_Structure)
importlib.reload(Model_Training)
model_name = "shared_backbone_2ch"
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


## plotting
plot_options = {
    "threshold_hits": False,
    "classification_examples": False,
    "certain_samples": True,
    "mean_std_overview": False,
    "random_sample_overview": False,
    "normalized_data_inspection": False,
}

plot_config = {
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
    "certain_samples": {
        "fold_id": 4,
        "sample_idx": 136,
        "channel_idx": 0,
    },
    "random_sample_overview": {
        "classes": ("LOW",),
        "n_samples": 30,
        "ncols": 6,
    },
    "normalized_data_inspection": {
        "fold_id": 0,
        "n_samples": 5,
    },
}

if plot_options["threshold_hits"]:
    Viewing.plot_threshold_hits(
        cv_folds=cv_folds,
        **plot_config["threshold_hits"],
    )

if plot_options["classification_examples"]:
    Viewing.plot_classification_examples(
        train_out=train_out,
        cv_folds=cv_folds,
        class_order=class_order,
        **plot_config["classification_examples"],
    )

if plot_options["certain_samples"]:
    Viewing.plot_certain_samples(
        categorized_dict=categorized_dict,
        sliced_dict=sliced_dict,
        sliced_filtered_dict=sliced_filtered_dict,
        x_trainval=x_trainval,
        y_trainval=y_trainval,
        cv_folds=cv_folds,
        train_out=train_out,
        **plot_config["certain_samples"],
    )

if plot_options["mean_std_overview"]:
    Viewing.plot_mean_std_overview(
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

if plot_options["random_sample_overview"]:
    Viewing.plot_random_sample_overview(
        class_order=class_order,
        random_seed=random_seed,
        categorized_dict=categorized_dict,
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
        **plot_config["random_sample_overview"],
    )

if plot_options["normalized_data_inspection"]:
    Viewing.inspect_normalized_data(
        cv_folds=cv_folds,
        class_order=class_order,
        selected_value_types=selected_value_types,
        **plot_config["normalized_data_inspection"],
    )


## grid search
run_grid_search = False
if run_grid_search:
    importlib.reload(Grid_Search)
    grid_search_space = {
        "base_channels": [32],
        "kernel_size": [5],
        "l1": [512],
        "l2": [256],
        "weight_decay": [1e-4],
        "dropout": [0.1],
        "label_smoothing": [0.3],
        "lr": [1e-4],
        "batch_size": [32],
        "conv_stride": [2],
        "conv_dilation": [1],
        "avgpool_kernel_size": [3],
        "avgpool_stride": [2],
        "pool_type": ["max"],
        "leaky_relu_alpha": [0.05],
        "scheduler_factor": [0.7],
    }

    grid_out = Grid_Search.run_grid_search_three_models(
        cv_folds=cv_folds,
        search_space=grid_search_space,
        model_names=("shared_backbone_2ch",),
        class_order=class_order,
        epochs=100,
        patience=25,
        tensorboard_log_dir_root=None,
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



