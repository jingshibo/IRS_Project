import os
import sys

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from Feature_Implementation import Feature_Extraction, Feature_Preprocessing, Feature_Training
from Utility_Functions import Preprocessing, Viewing


excel_path = "/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"
df = pd.read_excel(excel_path, sheet_name=0)
label_col = df.columns[0]
class_order = ("LOW", "TARGET", "HIGH")
categorized_dict = {key: group.drop(columns=[label_col]).reset_index(drop=True) for key, group in df.groupby(label_col)}


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


RANDOM_SEED = 42
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

X_trainval_signal, X_test_signal, y_trainval, y_test = Preprocessing.split_holdout(
    x_all,
    y_all,
    test_size=0.15,
    random_seed=RANDOM_SEED,
)

X_trainval_features, feature_names = Feature_Extraction.extract_feature_matrix(
    X_trainval_signal,
    channel_names=selected_value_types,
)
X_test_features, _ = Feature_Extraction.extract_feature_matrix(
    X_test_signal,
    channel_names=selected_value_types,
)
print("Signal train/val shape:", X_trainval_signal.shape)
print("Feature train/val shape:", X_trainval_features.shape)
print("Signal holdout shape:", X_test_signal.shape)
print("Feature holdout shape:", X_test_features.shape)
print("Extracted feature count:", len(feature_names))

cv_indices = Preprocessing.build_stratified_cv_indices(
    y_trainval,
    n_splits=5,
    random_seed=RANDOM_SEED,
)
signal_cv_folds = Preprocessing.build_normalized_cv_folds(
    X_trainval_signal,
    y_trainval,
    n_splits=5,
    random_seed=RANDOM_SEED,
    clip_max_value=None,
    cv_indices=cv_indices,
)
feature_cv_folds = Feature_Preprocessing.build_normalized_feature_cv_folds(
    X_trainval_features,
    y_trainval,
    n_splits=5,
    random_seed=RANDOM_SEED,
    cv_indices=cv_indices,
)

model_name = "feature_mlp_classifier"
train_out = Feature_Training.train_feature_mlp_cv(
    cv_folds=feature_cv_folds,
    class_order=class_order,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=25,
    tensorboard_log_dir="runs/feature_mlp_cv",
)
print("Mean best val acc:", train_out["mean_best_val_acc"])
print("Class index mapping:", train_out["label_to_idx"])
print("Model name:", model_name)


PLOT_OPTIONS = {
    "threshold_hits": False,
    "classification_examples": False,
    "certain_samples": True,
    "mean_std_overview": False,
    "random_sample_overview": False,
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

if PLOT_OPTIONS["threshold_hits"]:
    Viewing.plot_threshold_hits(
        cv_folds=signal_cv_folds,
        **PLOT_CONFIG["threshold_hits"],
    )

if PLOT_OPTIONS["classification_examples"]:
    Viewing.plot_classification_examples(
        train_out=train_out,
        cv_folds=signal_cv_folds,
        class_order=class_order,
        **PLOT_CONFIG["classification_examples"],
    )

if PLOT_OPTIONS["certain_samples"]:
    Viewing.plot_certain_samples(
        categorized_dict=categorized_dict,
        sliced_dict=sliced_dict,
        sliced_filtered_dict=sliced_filtered_dict,
        x_trainval=X_trainval_signal,
        y_trainval=y_trainval,
        cv_folds=signal_cv_folds,
        train_out=train_out,
        **PLOT_CONFIG["certain_samples"],
    )

if PLOT_OPTIONS["mean_std_overview"]:
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

if PLOT_OPTIONS["random_sample_overview"]:
    Viewing.plot_random_sample_overview(
        class_order=class_order,
        random_seed=RANDOM_SEED,
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
        **PLOT_CONFIG["random_sample_overview"],
    )

if PLOT_OPTIONS["normalized_data_inspection"]:
    Viewing.inspect_normalized_data(
        cv_folds=signal_cv_folds,
        class_order=class_order,
        selected_value_types=selected_value_types,
        **PLOT_CONFIG["normalized_data_inspection"],
    )
