import numpy as np

from IRS_Insecticide_Residual.Feature_Implementation.Functions import Peak_Dip_Features


def plot_peak_dip_summary(
    ax,
    signal: np.ndarray,
    band_edges: list[tuple[int, int]],
    sample_idx: int | None = None,
    label: str | int | None = None,
    peak_selection: str = "amplitude",
    dip_selection: str = "amplitude",
    min_prominence_frac: float = 0.20,
    min_distance: int = 1,
    min_width: int = 1,
    percentile_method: str = "histogram",
    general_peak_rel_height: float = 0.5,
    main_peak_rel_height: float = 0.9,
    dip_rel_height: float = 0.5,
    use_dip_reference_width: bool = True,
    show_main_peak: bool = True,
    show_pair_widths: bool = True,
    show_dip_width: bool = True,
    show_prominence_lines: bool = True,
    show_labels: bool = True,
):
    signal = np.asarray(signal, dtype=np.float32)
    x_axis = np.arange(signal.shape[0])
    signal_max = float(np.max(signal))
    signal_min = float(np.min(signal))
    signal_range = max(signal_max - signal_min, 1.0)

    peaks_and_dips = Peak_Dip_Features.detect_peaks_and_dips(
        signal,
        min_prominence_frac=min_prominence_frac,
        min_distance=min_distance,
        min_width=min_width,
        percentile_method=percentile_method,
        general_peak_rel_height=general_peak_rel_height,
        main_peak_rel_height=main_peak_rel_height,
        dip_rel_height=dip_rel_height,
    )
    peak_dip_pairs = Peak_Dip_Features.select_band_peak_dip_pairs(
        peaks_and_dips,
        band_edges=band_edges,
        peak_selection=peak_selection,
        dip_selection=dip_selection,
    )
    area_features = Peak_Dip_Features.calculate_doublet_area_features(peak_dip_pairs, signal)

    peaks = peaks_and_dips["peak_frequencies"]
    dips = peaks_and_dips["dip_frequencies"]
    peak_width_heights = peaks_and_dips["peak_width_heights"]
    peak_left_ips = peaks_and_dips["peak_left_ips"]
    peak_right_ips = peaks_and_dips["peak_right_ips"]
    peak_left_bases = peaks_and_dips["peak_left_bases"]
    peak_right_bases = peaks_and_dips["peak_right_bases"]
    dip_width_heights = peaks_and_dips["dip_width_heights"]
    dip_left_ips = peaks_and_dips["dip_left_ips"]
    dip_right_ips = peaks_and_dips["dip_right_ips"]
    dip_left_bases = peaks_and_dips["dip_left_bases"]
    dip_right_bases = peaks_and_dips["dip_right_bases"]

    ax.plot(x_axis, signal, color="black", linewidth=1.0)
    ax.scatter(peaks, signal[peaks], marker="^", color="red", s=40, label="peaks")
    ax.scatter(dips, signal[dips], marker="o", color="blue", s=30, label="dips")

    for pair in peak_dip_pairs:
        if show_main_peak and pair.get("main_peak_exists", False):
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
                if show_prominence_lines:
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

        if pair.get("pair_exists", False):
            left_freq = int(pair["left_peak_freq"])
            right_freq = int(pair["right_peak_freq"])
            pair_x = [left_freq, right_freq]
            pair_y = [signal[left_freq], signal[right_freq]]

            ax.plot(pair_x, pair_y, color="darkorange", linewidth=1.2, linestyle="--", alpha=0.9, label="peak pair")
            ax.scatter(pair_x, pair_y, marker="s", color="darkorange", s=36, zorder=4, label="selected pair peaks")

            use_dip_reference_width_for_pair = use_dip_reference_width and pair.get("middle_dip_exists", False)
            for side, peak_freq, peak_width, peak_prominence, width_color, label_color, prom_color, width_label, prom_span_label, prom_label in (
                (
                    "left",
                    left_freq,
                    pair["left_peak_width"],
                    pair["left_peak_prominence"],
                    "deepskyblue",
                    "deepskyblue",
                    "limegreen",
                    "left peak width at dip",
                    "left peak prominence span",
                    "left peak prominence",
                ),
                (
                    "right",
                    right_freq,
                    pair["right_peak_width"],
                    pair["right_peak_prominence"],
                    "mediumorchid",
                    "firebrick",
                    "goldenrod",
                    "right peak width at dip",
                    "right peak prominence span",
                    "right peak prominence",
                ),
            ):
                peak_amp = float(signal[peak_freq])
                peak_ids = np.where(peaks == peak_freq)[0]
                if peak_ids.size == 0:
                    continue

                peak_id = int(peak_ids[0])
                if use_dip_reference_width_for_pair:
                    width_height = float(pair["middle_dip_amp"])
                    width_left = float(pair[f"{side}_peak_left_ips_at_middle_dip"])
                    width_right = float(pair[f"{side}_peak_right_ips_at_middle_dip"])
                    peak_width_value = float(pair[f"{side}_peak_width_at_middle_dip"])
                else:
                    width_height = float(peak_width_heights[peak_id])
                    width_left = float(peak_left_ips[peak_id])
                    width_right = float(peak_right_ips[peak_id])
                    peak_width_value = float(peak_width)
                prom_left = int(peak_left_bases[peak_id])
                prom_right = int(peak_right_bases[peak_id])
                prom_base_level = peak_amp - float(peak_prominence)
                area_prefix = f"band{pair['band_id']}"
                if use_dip_reference_width_for_pair:
                    peak_area_value = float(area_features[f"{area_prefix}_{side}_peak_area_at_middle_dip"])
                else:
                    peak_area_value = float(area_features[f"{area_prefix}_{side}_peak_area"])

                if show_pair_widths:
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
                if show_prominence_lines:
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

                if show_labels:
                    peak_note = (
                        f"{side[0]} a={peak_amp:.1f}\n"
                        f"w={peak_width_value:.1f}\n"
                        f"A={peak_area_value:.1f}\n"
                        f"p={peak_prominence:.1f}"
                    )
                    place_below = peak_amp > signal_min + 0.82 * signal_range
                    y_offset = -10 if place_below else 8
                    x_offset = -28 if side == "left" else 6
                    horizontal_align = "right" if side == "left" else "left"
                    vertical_align = "top" if place_below else "bottom"
                    ax.annotate(
                        peak_note,
                        xy=(peak_freq, peak_amp),
                        xytext=(x_offset, y_offset),
                        textcoords="offset points",
                        fontsize=6.8,
                        color=label_color,
                        ha=horizontal_align,
                        va=vertical_align,
                        annotation_clip=False,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=label_color, alpha=0.75),
                    )

            if pair.get("middle_dip_exists", False):
                dip_freq = int(pair["middle_dip_freq"])
                dip_amp = float(signal[dip_freq])
                ax.scatter([dip_freq], [dip_amp], marker="D", color="purple", s=34, zorder=5, label="middle dip")
                ax.plot([left_freq, dip_freq, right_freq], [signal[left_freq], dip_amp, signal[right_freq]], color="purple", linewidth=1.0, alpha=0.6)

                dip_ids = np.where(dips == dip_freq)[0]
                if dip_ids.size > 0:
                    dip_id = int(dip_ids[0])
                    dip_width_height = float(dip_width_heights[dip_id])
                    dip_width_left = float(dip_left_ips[dip_id])
                    dip_width_right = float(dip_right_ips[dip_id])
                    dip_prom_left = int(dip_left_bases[dip_id])
                    dip_prom_right = int(dip_right_bases[dip_id])
                    dip_prom_base_level = dip_amp + float(pair["middle_dip_prominence"])

                    if show_dip_width:
                        ax.hlines(
                            y=dip_width_height,
                            xmin=dip_width_left,
                            xmax=dip_width_right,
                            color="purple",
                            linewidth=1.2,
                            linestyle="-.",
                            alpha=0.9,
                            label="middle dip width",
                        )
                        ax.vlines(
                            x=[dip_width_left, dip_width_right],
                            ymin=dip_width_height - 0.35,
                            ymax=dip_width_height + 0.35,
                            color="purple",
                            linewidth=1.0,
                            alpha=0.9,
                        )
                    if show_prominence_lines:
                        ax.hlines(
                            y=dip_prom_base_level,
                            xmin=dip_prom_left,
                            xmax=dip_prom_right,
                            color="magenta",
                            linewidth=1.0,
                            linestyle=":",
                            alpha=0.85,
                            label="middle dip prominence span",
                        )
                        ax.vlines(
                            x=dip_freq,
                            ymin=dip_amp,
                            ymax=dip_prom_base_level,
                            color="magenta",
                            linewidth=1.0,
                            alpha=0.85,
                            label="middle dip prominence",
                        )

                if show_labels:
                    dip_area_value = float(area_features[f"band{pair['band_id']}_middle_dip_area"])
                    dip_note = (
                        f"d a={dip_amp:.1f}\n"
                        f"w={pair['middle_dip_width']:.1f}\n"
                        f"A={dip_area_value:.1f}\n"
                        f"p={pair['middle_dip_prominence']:.1f}"
                    )
                    place_above = dip_amp < signal_min + 0.22 * signal_range
                    y_offset = 10 if place_above else -10
                    vertical_align = "bottom" if place_above else "top"
                    ax.annotate(
                        dip_note,
                        xy=(dip_freq, dip_amp),
                        xytext=(6, y_offset),
                        textcoords="offset points",
                        fontsize=6.5,
                        color="purple",
                        ha="left",
                        va=vertical_align,
                        annotation_clip=False,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="purple", alpha=0.72),
                    )
        elif pair.get("main_peak_exists", False):
            peak_freq = int(pair["main_peak_freq"])
            ax.scatter([peak_freq], [signal[peak_freq]], marker="s", facecolors="none", edgecolors="darkorange", s=42, linewidths=1.2, zorder=4, label="selected single peak")

    for band_start, _ in band_edges[1:]:
        ax.axvline(band_start, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_ylim(signal_min - 0.08 * signal_range, signal_max + 0.18 * signal_range)
    if sample_idx is not None or label is not None:
        ax.set_title(f"idx={sample_idx}, label={label}", fontsize=9)

    return {
        "peaks_and_dips": peaks_and_dips,
        "peak_dip_pairs": peak_dip_pairs,
        "area_features": area_features,
    }
