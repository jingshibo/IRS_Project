##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IRS_Insecticide_Residual.Feature_Implementation.Functions import Derivative_Features, Peak_Dip_Features, Plotting, \
    Global_Features
from IRS_Insecticide_Residual.Utility_Functions import Preprocessing

## load data
excel_path = "/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"
df = pd.read_excel(excel_path, sheet_name=0)
label_col = df.columns[0]
df_clean, removed_zero_sample_indices = Preprocessing.remove_zero_samples(df, label_col=df.columns[0], reset_index=True)
print("Removed all-zero sample indices:", removed_zero_sample_indices)
class_order = ("LOW", "TARGET", "HIGH")
categorized_dict = {key: group.drop(columns=[label_col]).reset_index(drop=True) for key, group in df_clean.groupby(label_col)}


## preprocesing
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

## build dataset
value_type_dicts = {
    "original": original_filtered_dict,
    "first_diff_filtered": central_diff_filtered_dict,
    "second_diff_filtered": second_diff_filtered_dict,
}
SELECTED_VALUE_TYPES = ("original", "first_diff_filtered", "second_diff_filtered")
x_all, y_all = Preprocessing.build_multi_channel_dataset(
    data_dict_map=value_type_dicts,
    selected_types=SELECTED_VALUE_TYPES,
)

## feature and plotting
RANDOM_SEED = 152
rng = np.random.default_rng(RANDOM_SEED)
sample_indices = rng.choice(x_all.shape[0], size=20, replace=False)
ORIGINAL_BAND_EDGES = [(0, 1000), (1000, 1750), (1750, 2700)]
# When the downsampling ratio is 5, the original band edges become [(0, 200), (200, 350), (350, 540)]
BAND_EDGES = [(start // DOWNSAMPLING_RATIO, end // DOWNSAMPLING_RATIO) for start, end in ORIGINAL_BAND_EDGES]
PEAK_SELECTION = "amplitude"
DIP_SELECTION = "amplitude"
GENERAL_PEAK_REL_HEIGHT = 0.5
MAIN_PEAK_REL_HEIGHT = 0.9
DIP_REL_HEIGHT = 0.5
USE_DIP_REFERENCE_WIDTH = True
SHOW_MAIN_PEAK = True
SHOW_PAIR_WIDTHS = True
SHOW_DIP_WIDTH = True
SHOW_PROMINENCE_LINES = True
SHOW_LABELS = True

fig, axes = plt.subplots(4, 5, figsize=(20, 12))
axes = axes.ravel()

for ax, sample_idx in zip(axes, sample_indices): # loop and plot each sample separately
    signal = x_all[sample_idx, 1, :]
    Plotting.plot_peak_dip_summary(
        ax=ax,
        signal=signal,
        band_edges=BAND_EDGES,
        sample_idx=int(sample_idx),
        label=y_all[sample_idx],
        peak_selection=PEAK_SELECTION,
        dip_selection=DIP_SELECTION,
        min_prominence_frac=0.20,
        min_distance=1,
        min_width=1,
        percentile_method="histogram",
        general_peak_rel_height=GENERAL_PEAK_REL_HEIGHT,
        main_peak_rel_height=MAIN_PEAK_REL_HEIGHT,
        dip_rel_height=DIP_REL_HEIGHT,
        use_dip_reference_width=USE_DIP_REFERENCE_WIDTH,
        show_main_peak=SHOW_MAIN_PEAK,
        show_pair_widths=SHOW_PAIR_WIDTHS,
        show_dip_width=SHOW_DIP_WIDTH,
        show_prominence_lines=SHOW_PROMINENCE_LINES,
        show_labels=SHOW_LABELS,
    )

handles, labels = axes[0].get_legend_handles_labels()
unique_handles = {}
for handle, label in zip(handles, labels):
    if label not in unique_handles:
        unique_handles[label] = handle
fig.legend(unique_handles.values(), unique_handles.keys(), loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()


## doublet feature calculation
sample_signal = x_all[1698, 0, :]
sample_peaks_and_dips = Peak_Dip_Features.detect_peaks_and_dips(
    sample_signal,
    min_prominence_frac=0.20,
    min_distance=1,
    min_width=1,
    percentile_method="histogram",
    general_peak_rel_height=GENERAL_PEAK_REL_HEIGHT,
    main_peak_rel_height=MAIN_PEAK_REL_HEIGHT,
    dip_rel_height=DIP_REL_HEIGHT,
)
sample_peak_dip_pairs = Peak_Dip_Features.select_band_peak_dip_pairs(
    sample_peaks_and_dips,
    band_edges=BAND_EDGES,
    peak_selection=PEAK_SELECTION,
    dip_selection=DIP_SELECTION,
)
doublet_features = Peak_Dip_Features.calculate_doublet_features(sample_peak_dip_pairs)
area_features = Peak_Dip_Features.calculate_doublet_area_features(sample_peak_dip_pairs, sample_signal)


## derivative feature calculation
sample_first_derivative = x_all[1698, 1, :]
sample_second_derivative = x_all[1698, 2, :]
first_derivative_features = Derivative_Features.calculate_first_derivative_features(
    selected_pairs=sample_peak_dip_pairs,
    signal=sample_signal,
    first_derivative=sample_first_derivative,
    detected_peak_dip=sample_peaks_and_dips,
    window_radius=5,
    include_inter_band=True,
    include_broad_transition=True,
)
second_derivative_features = Derivative_Features.calculate_second_derivative_features(
    selected_pairs=sample_peak_dip_pairs,
    signal=sample_signal,
    first_derivative=sample_first_derivative,
    second_derivative=sample_second_derivative,
    detected_peak_dip=sample_peaks_and_dips,
    window_radius=5,
    include_inter_band=True,
    include_broad_transition=True,
)

## global features
feature_block, feature_names = Global_Features.calculate_global_features(x_all[:, 0, :], channel_name="original")