# import os
# import sys
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
from IRS_Insecticide_Residual.Feature_Implementation import Feature_Preprocessing
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
        signal_channel_index=ORIGINAL_CHANNEL_INDEFeatureX,
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

trainval_detected_peak_dip = feature_metadata["trainval"]["detected_peak_dip"]
trainval_peak_dip_pairs = feature_metadata["trainval"]["peak_dip_pairs"]
test_detected_peak_dip = feature_metadata["test"]["detected_peak_dip"]
test_peak_dip_pairs = feature_metadata["test"]["peak_dip_pairs"]
global_feature_names = feature_metadata["trainval"]["global_feature_names"]
peak_dip_feature_names = feature_metadata["trainval"]["peak_dip_feature_names"]
derivative_feature_names = feature_metadata["trainval"]["derivative_feature_names"]

print("Signal train/val shape:", x_trainval_signal.shape)
print("Feature train/val shape:", x_trainval_features.shape)
print("Signal holdout shape:", x_test_signal.shape)
print("Feature holdout shape:", x_test_features.shape)
print("Global feature count:", len(global_feature_names))
print("Peak/dip feature count:", len(peak_dip_feature_names))
print("Derivative feature count:", len(derivative_feature_names))
print("Total extracted feature count:", len(feature_names))


## create dataset
# cv_indices = Preprocessing.build_stratified_cv_indices(
#     y_trainval,
#     n_splits=5,
#     random_seed=RANDOM_SEED,
# )
# signal_cv_folds = Preprocessing.build_normalized_cv_folds(
#     x_trainval_signal,
#     y_trainval,
#     n_splits=5,
#     random_seed=RANDOM_SEED,
#     clip_max_value=None,
#     cv_indices=cv_indices,
# )
# feature_cv_folds = Feature_Preprocessing.build_normalized_feature_cv_folds(
#     x_trainval_features,
#     y_trainval,
#     n_splits=5,
#     random_seed=RANDOM_SEED,
#     cv_indices=cv_indices,
# )
#
#
# ## model training
# model_name = "feature_mlp_classifier"
# train_out = Feature_Training.train_feature_mlp_cv(
#     cv_folds=feature_cv_folds,
#     class_order=class_order,
#     epochs=100,
#     batch_size=32,
#     lr=1e-3,
#     weight_decay=1e-4,
#     label_smoothing=0.1,
#     patience=25,
#     tensorboard_log_dir="runs/feature_mlp_cv",
# )
# print("Mean best val acc:", train_out["mean_best_val_acc"])
# print("Class index mapping:", train_out["label_to_idx"])
# print("Model name:", model_name)
#
#
# ## plotting
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
