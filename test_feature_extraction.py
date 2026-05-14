##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Feature_Implementation.Functions import Peak_Dip_Features
from Utility_Functions import Preprocessing

## load data
EXCEL_PATH = "/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"
df = pd.read_excel(EXCEL_PATH, sheet_name=0)
label_col = df.columns[0]
categorized_dict = {
    key: group.drop(columns=[label_col]).reset_index(drop=True)
    for key, group in df.groupby(label_col)
}

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
sample_indices = rng.choice(x_all.shape[0], size=30, replace=False)
BAND_EDGES = [(0, 200), (200, 350), (350, 540)]

fig, axes = plt.subplots(5, 6, figsize=(20, 12))
axes = axes.ravel()

for ax, sample_idx in zip(axes, sample_indices):
    signal = x_all[sample_idx, 0, :]
    x_axis = np.arange(signal.shape[0])
    signal_max = float(np.max(signal))
    signal_min = float(np.min(signal))
    signal_range = max(signal_max - signal_min, 1.0)
    peaks_and_dips = Peak_Dip_Features.detect_peaks_and_dips(signal, min_prominence_frac=0.20, min_distance=1, min_width=1,
                                                             percentile_method="histogram")
    peak_dip_pairs = Peak_Dip_Features.select_band_peak_dip_pairs(
        peaks_and_dips,
        band_edges=BAND_EDGES,
        peak_selection="amplitude",
        dip_selection="amplitude",
    )
    peaks = peaks_and_dips["peak_frequencies"]
    dips = peaks_and_dips["dip_frequencies"]
    peak_width_heights = peaks_and_dips["peak_width_heights"]
    peak_left_ips = peaks_and_dips["peak_left_ips"]
    peak_right_ips = peaks_and_dips["peak_right_ips"]
    peak_left_bases = peaks_and_dips["peak_left_bases"]
    peak_right_bases = peaks_and_dips["peak_right_bases"]

    ax.plot(x_axis, signal, color="black", linewidth=1.0)
    ax.scatter(peaks, signal[peaks], marker="^", color="red", s=40, label="peaks")
    ax.scatter(dips, signal[dips], marker="o", color="blue", s=30, label="dips")

    for pair in peak_dip_pairs:
        if pair.get("main_peak_exists", False):
            main_peak_freq = int(pair["main_peak_freq"])
            main_peak_amp = float(signal[main_peak_freq])
            main_peak_ids = np.where(peaks == main_peak_freq)[0]
            if main_peak_ids.size > 0:
                main_peak_id = int(main_peak_ids[0])
                width_height = float(pair["main_peak_width_height"])
                width_left = float(pair["main_peak_left_ips"])
                width_right = float(pair["main_peak_right_ips"])
                prom_left = int(peak_left_bases[main_peak_id])
                prom_right = int(peak_right_bases[main_peak_id])
                prom_base_level = main_peak_amp - float(pair["main_peak_prominence"])

                ax.hlines(
                    y=width_height,
                    xmin=width_left,
                    xmax=width_right,
                    color="teal",
                    linewidth=1.3,
                    linestyle="-",
                    alpha=0.9,
                    label="peak width",
                )
                ax.vlines(
                    x=[width_left, width_right],
                    ymin=width_height - 0.35,
                    ymax=width_height + 0.35,
                    color="teal",
                    linewidth=1.0,
                    alpha=0.9,
                )

                ax.hlines(
                    y=prom_base_level,
                    xmin=prom_left,
                    xmax=prom_right,
                    color="green",
                    linewidth=1.1,
                    linestyle=":",
                    alpha=0.9,
                    label="peak prominence span",
                )
                ax.vlines(
                    x=main_peak_freq,
                    ymin=prom_base_level,
                    ymax=main_peak_amp,
                    color="green",
                    linewidth=1.1,
                    alpha=0.9,
                    label="peak prominence",
                )

            peak_note = (
                f"a={main_peak_amp:.1f}\n"
                f"w={pair['main_peak_width']:.1f}\n"
                f"p={pair['main_peak_prominence']:.1f}"
            )
            place_below = main_peak_amp > signal_min + 0.82 * signal_range
            y_offset = -10 if place_below else 8
            vertical_align = "top" if place_below else "bottom"
            ax.annotate(
                peak_note,
                xy=(main_peak_freq, main_peak_amp),
                xytext=(5, y_offset),
                textcoords="offset points",
                fontsize=7,
                color="darkorange",
                ha="left",
                va=vertical_align,
                annotation_clip=False,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkorange", alpha=0.75),
            )

        if pair.get("pair_exists", False):
            left_freq = int(pair["left_peak_freq"])
            right_freq = int(pair["right_peak_freq"])
            pair_x = [left_freq, right_freq]
            pair_y = [signal[left_freq], signal[right_freq]]

            ax.plot(pair_x, pair_y, color="darkorange", linewidth=1.2, linestyle="--", alpha=0.9, label="peak pair")
            ax.scatter(
                pair_x,
                pair_y,
                marker="s",
                color="darkorange",
                s=36,
                zorder=4,
                label="selected pair peaks",
            )

            for peak_freq, peak_width, peak_prominence, width_color, prom_color, width_label, prom_span_label, prom_label in (
                (
                    left_freq,
                    pair["left_peak_width"],
                    pair["left_peak_prominence"],
                    "deepskyblue",
                    "limegreen",
                    "left peak width",
                    "left peak prominence span",
                    "left peak prominence",
                ),
                (
                    right_freq,
                    pair["right_peak_width"],
                    pair["right_peak_prominence"],
                    "mediumorchid",
                    "goldenrod",
                    "right peak width",
                    "right peak prominence span",
                    "right peak prominence",
                ),
            ):
                peak_amp = float(signal[peak_freq])
                peak_ids = np.where(peaks == peak_freq)[0]
                if peak_ids.size == 0:
                    continue

                peak_id = int(peak_ids[0])
                width_height = float(peak_width_heights[peak_id])
                width_left = float(peak_left_ips[peak_id])
                width_right = float(peak_right_ips[peak_id])
                prom_left = int(peak_left_bases[peak_id])
                prom_right = int(peak_right_bases[peak_id])
                prom_base_level = peak_amp - float(peak_prominence)

                ax.hlines(
                    y=width_height,
                    xmin=width_left,
                    xmax=width_right,
                    color=width_color,
                    linewidth=1.1,
                    linestyle="--",
                    alpha=0.8,
                    label=width_label,
                )
                ax.hlines(
                    y=prom_base_level,
                    xmin=prom_left,
                    xmax=prom_right,
                    color=prom_color,
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.75,
                    label=prom_span_label,
                )
                ax.vlines(
                    x=peak_freq,
                    ymin=prom_base_level,
                    ymax=peak_amp,
                    color=prom_color,
                    linewidth=1.0,
                    alpha=0.75,
                    label=prom_label,
                )

            if pair.get("middle_dip_exists", False):
                dip_freq = int(pair["middle_dip_freq"])
                dip_amp = signal[dip_freq]
                ax.scatter(
                    [dip_freq],
                    [dip_amp],
                    marker="D",
                    color="purple",
                    s=34,
                    zorder=5,
                    label="middle dip",
                )
                ax.plot(
                    [left_freq, dip_freq, right_freq],
                    [signal[left_freq], dip_amp, signal[right_freq]],
                    color="purple",
                    linewidth=1.0,
                    alpha=0.6,
                )
        elif pair.get("main_peak_exists", False):
            peak_freq = int(pair["main_peak_freq"])
            ax.scatter(
                [peak_freq],
                [signal[peak_freq]],
                marker="s",
                facecolors="none",
                edgecolors="darkorange",
                s=42,
                linewidths=1.2,
                zorder=4,
                label="selected single peak",
            )

    for band_start, band_end in BAND_EDGES[1:]:
        ax.axvline(band_start, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_ylim(signal_min - 0.08 * signal_range, signal_max + 0.18 * signal_range)
    ax.set_title(f"idx={sample_idx}, label={y_all[sample_idx]}", fontsize=9)

handles, labels = axes[0].get_legend_handles_labels()
unique_handles = {}
for handle, label in zip(handles, labels):
    if label not in unique_handles:
        unique_handles[label] = handle
fig.legend(unique_handles.values(), unique_handles.keys(), loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()


##
doublet_features = Peak_Dip_Features.calculate_doublet_features(peak_dip_pairs)
Peak_Dip_Features.calculate_doublet_area_features(peak_dip_pairs, signal)


peaks_and_dips = Peak_Dip_Features.detect_peaks_and_dips(x_all[2395, 0, :], min_prominence_frac=0.20, min_distance=1, min_width=1,
                                                         percentile_method="histogram")