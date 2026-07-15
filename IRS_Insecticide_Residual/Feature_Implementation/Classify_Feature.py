# import os
# import sys
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import pandas as pd
from IRS_Insecticide_Residual.Feature_Implementation import Feature_Preprocessing
from IRS_Insecticide_Residual.Feature_Implementation.Models import Feature_Sklearn_Training
from IRS_Insecticide_Residual.Feature_Implementation.Models import Feature_Training
from IRS_Insecticide_Residual.Feature_Implementation.Functions import Feature_Extraction
from IRS_Insecticide_Residual.Utility_Functions import Preprocessing
from IRS_Insecticide_Residual.Raw_Data_Implementation.Functions import Viewing



## load data
excel_path = "/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"
df = pd.read_excel(excel_path, sheet_name=0)
label_col = df.columns[0]
df_clean, removed_zero_sample_indices = Preprocessing.remove_zero_samples(df, label_col=df.columns[0], reset_index=True)
print("Removed all-zero sample indices:", removed_zero_sample_indices)
class_order = ("LOW", "TARGET", "HIGH")
categorized_dict = {key: group.drop(columns=[label_col]).reset_index(drop=True) for key, group in df_clean.groupby(label_col)}


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
DOWNSAMPLING_RATIO = 5
original_filtered_dict = Preprocessing.downsample_dict_signals(original_filtered_dict, step=DOWNSAMPLING_RATIO, offset=0)

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


## input signal construction
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

x_trainval_signal, x_test_signal, y_trainval, y_test = Preprocessing.split_holdout(
    x_all,
    y_all,
    test_size=0.15,
    random_seed=RANDOM_SEED,
)


## global, peak/dip, and derivative feature extraction
ORIGINAL_CHANNEL_INDEX = selected_value_types.index("original")
FIRST_DERIVATIVE_CHANNEL_INDEX = selected_value_types.index("first_diff_filtered")
SECOND_DERIVATIVE_CHANNEL_INDEX = selected_value_types.index("second_diff_filtered")

ORIGINAL_BAND_EDGES = [(0, 1000), (1000, 1750), (1750, 2700)]
BAND_EDGES = [(start // DOWNSAMPLING_RATIO, end // DOWNSAMPLING_RATIO) for start, end in ORIGINAL_BAND_EDGES]

PEAK_SELECTION = "amplitude"
DIP_SELECTION = "amplitude"
MIN_PROMINENCE_FRAC = 0.20
MIN_DISTANCE = 1
MIN_WIDTH = 1
PERCENTILE_METHOD = "histogram"
GENERAL_PEAK_REL_HEIGHT = 0.5
MAIN_PEAK_REL_HEIGHT = 0.9
DIP_REL_HEIGHT = 0.5
DERIVATIVE_WINDOW_RADIUS = 5

x_trainval_features, x_test_features, feature_names, feature_metadata = (
    Feature_Extraction.extract_combined_features_for_train_test(
        x_trainval_signal=x_trainval_signal,
        x_test_signal=x_test_signal,
        band_edges=BAND_EDGES,
        channel_names=selected_value_types,
        signal_channel_index=ORIGINAL_CHANNEL_INDEX,
        first_derivative_channel_index=FIRST_DERIVATIVE_CHANNEL_INDEX,
        second_derivative_channel_index=SECOND_DERIVATIVE_CHANNEL_INDEX,
        peak_selection=PEAK_SELECTION,
        dip_selection=DIP_SELECTION,
        min_prominence_frac=MIN_PROMINENCE_FRAC,
        min_distance=MIN_DISTANCE,
        min_width=MIN_WIDTH,
        percentile_method=PERCENTILE_METHOD,
        general_peak_rel_height=GENERAL_PEAK_REL_HEIGHT,
        main_peak_rel_height=MAIN_PEAK_REL_HEIGHT,
        dip_rel_height=DIP_REL_HEIGHT,
        derivative_window_radius=DERIVATIVE_WINDOW_RADIUS,
        include_area_features=True,
        include_inter_band=True,
        include_broad_transition=True,
        include_second_derivative=True,
    )
)

# access feature information
# trainval_detected_peak_dip = feature_metadata["trainval"]["detected_peak_dip"]
# trainval_peak_dip_pairs = feature_metadata["trainval"]["peak_dip_pairs"]
# test_detected_peak_dip = feature_metadata["test"]["detected_peak_dip"]
# test_peak_dip_pairs = feature_metadata["test"]["peak_dip_pairs"]
# global_feature_names = feature_metadata["trainval"]["global_feature_names"]
# peak_dip_feature_names = feature_metadata["trainval"]["peak_dip_feature_names"]
# derivative_feature_names = feature_metadata["trainval"]["derivative_feature_names"]


## create dataset
FEATURE_REDUCTION_METHOD = "pca"  # use "none" or "pca"
PCA_N_COMPONENTS = 0.97  # keep enough PCA components to explain this fraction of variance
if FEATURE_REDUCTION_METHOD not in ("none", "pca"):
    raise ValueError("FEATURE_REDUCTION_METHOD must be 'none' or 'pca'")

feature_cv_folds = Feature_Preprocessing.build_feature_cv_folds(
    x_trainval_features,
    y_trainval,
    n_splits=5,
    random_seed=RANDOM_SEED,
    use_pca=(FEATURE_REDUCTION_METHOD == "pca"),
    pca_n_components=PCA_N_COMPONENTS,
)

# pca result details
if FEATURE_REDUCTION_METHOD == "pca":
    for fold_data in feature_cv_folds:
        pca_details = fold_data["pca_details"]
        print(
            f"PCA fold {fold_data['fold']}: "
            f"n_components={pca_details['n_components']}, "
            f"total_explained_variance_ratio={pca_details['total_explained_variance_ratio']:.4f}"
        )


## model training
MODEL_NAME = "mlp"  # use "mlp", "lda", "svm", "random_forest", or "knn"
if MODEL_NAME not in ("mlp", "lda", "svm", "random_forest", "rf", "knn"):
    raise ValueError("MODEL_NAME must be 'mlp', 'lda', 'svm', 'random_forest', 'rf', or 'knn'")

if MODEL_NAME == "mlp":
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
else:
    train_out = Feature_Sklearn_Training.train_sklearn_feature_cv(
        cv_folds=feature_cv_folds,
        model_name=MODEL_NAME,
        class_order=class_order,
        random_seed=RANDOM_SEED,
    )
print("Mean best val acc:", train_out["mean_best_val_acc"])
print("Class index mapping:", train_out["label_to_idx"])
print("Model name:", MODEL_NAME)


## holdout test evaluation using CV model ensemble
test_prob_by_fold = []
for fold_data, fold_result in zip(feature_cv_folds, train_out["fold_results"]):
    x_test_fold = fold_data["feature_scaler"].transform(x_test_features).astype(np.float32, copy=False)
    if fold_data.get("feature_reducer") is not None:
        x_test_fold = fold_data["feature_reducer"].transform(x_test_fold).astype(np.float32, copy=False)

    test_prob_by_fold.append(
        (
            Feature_Training.predict_prob(
                fold_result.model,
                x_test_fold,
                device=train_out["device"],
            )
            if MODEL_NAME == "mlp"
            else Feature_Sklearn_Training.predict_prob(
                fold_result.model,
                x_test_fold,
                label_to_idx=train_out["label_to_idx"],
            )
        )
    )

test_prob = np.mean(np.stack(test_prob_by_fold, axis=0), axis=0)
test_pred_idx = np.argmax(test_prob, axis=1)
test_true_idx = np.asarray([train_out["label_to_idx"][label] for label in y_test], dtype=np.int64)
test_acc = float(np.mean(test_pred_idx == test_true_idx))
test_pred_label = [train_out["idx_to_label"][int(idx)] for idx in test_pred_idx]

print("Holdout test acc:", test_acc)


## plotting
# signal_cv_folds = Preprocessing.build_normalized_cv_folds(
#     x_trainval_signal,
#     y_trainval,
#     n_splits=5,
#     random_seed=RANDOM_SEED,
#     clip_max_value=None,
#     cv_indices=cv_indices,
# )
# PLOT_OPTIONS = {
#     "threshold_hits": False,
#     "classification_examples": False,
#     "certain_samples": False,
#     "mean_std_overview": False,
#     "random_sample_overview": False,
#     "normalized_data_inspection": False,
# }
#
# PLOT_CONFIG = {
#     "threshold_hits": {
#         "channel_idx": 0,
#         "threshold": 25.0,
#         "split": None,
#         "nrows": 4,
#         "ncols": 4,
#         "title_prefix": "Normalized threshold-hit signals",
#     },
#     "classification_examples": {
#         "classes": class_order,
#         "top_k": 10,
#         "channel_idx": 0,
#     },
#     "certain_samples": {
#         "fold_id": 4,
#         "sample_idx": 136,
#         "channel_idx": 0,
#     },
#     "random_sample_overview": {
#         "classes": ("LOW",),
#         "n_samples": 30,
#         "ncols": 6,
#     },
#     "normalized_data_inspection": {
#         "fold_id": 0,
#         "n_samples": 5,
#     },
# }
#
# if PLOT_OPTIONS["threshold_hits"]:
#     Viewing.plot_threshold_hits(
#         cv_folds=signal_cv_folds,
#         **PLOT_CONFIG["threshold_hits"],
#     )
#
# if PLOT_OPTIONS["classification_examples"]:
#     Viewing.plot_classification_examples(
#         train_out=train_out,
#         cv_folds=signal_cv_folds,
#         class_order=class_order,
#         **PLOT_CONFIG["classification_examples"],
#     )
#
# if PLOT_OPTIONS["certain_samples"]:
#     Viewing.plot_certain_samples(
#         categorized_dict=categorized_dict,
#         sliced_dict=sliced_dict,
#         sliced_filtered_dict=sliced_filtered_dict,
#         x_trainval=x_trainval_signal,
#         y_trainval=y_trainval,
#         cv_folds=signal_cv_folds,
#         train_out=train_out,
#         **PLOT_CONFIG["certain_samples"],
#     )
#
# if PLOT_OPTIONS["mean_std_overview"]:
#     Viewing.plot_mean_std_overview(
#         class_order=class_order,
#         original_dict=sliced_dict,
#         original_filtered_dict=original_filtered_dict,
#         central_diff_dict=central_diff_dict,
#         central_diff_filtered_dict=central_diff_filtered_dict,
#         second_diff_dict=second_diff_dict,
#         second_diff_filtered_dict=second_diff_filtered_dict,
#         rolling_variance_dict=rolling_variance_dict,
#         derivative_energy_dict=derivative_energy_dict,
#     )
#
# if PLOT_OPTIONS["random_sample_overview"]:
#     Viewing.plot_random_sample_overview(
#         class_order=class_order,
#         random_seed=RANDOM_SEED,
#         categorized_dict=categorized_dict,
#         sliced_dict=sliced_dict,
#         sliced_filtered_dict=sliced_filtered_dict,
#         original_filtered_dict=original_filtered_dict,
#         original_envelope_dict=original_envelope_dict,
#         original_residual_dict=original_residual_dict,
#         central_diff_dict=central_diff_dict,
#         central_diff_filtered_dict=central_diff_filtered_dict,
#         second_diff_dict=second_diff_dict,
#         second_diff_filtered_dict=second_diff_filtered_dict,
#         rolling_variance_dict=rolling_variance_dict,
#         derivative_energy_dict=derivative_energy_dict,
#         **PLOT_CONFIG["random_sample_overview"],
#     )
#
# if PLOT_OPTIONS["normalized_data_inspection"]:
#     Viewing.inspect_normalized_data(
#         cv_folds=signal_cv_folds,
#         class_order=class_order,
#         selected_value_types=selected_value_types,
#         **PLOT_CONFIG["normalized_data_inspection"],
#     )
