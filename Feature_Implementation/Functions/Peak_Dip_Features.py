from scipy.signal import find_peaks, peak_widths, peak_prominences
import numpy as np


## compute approximate percentile using histogram binning (returns float32)
def _histogram_percentile(x: np.ndarray, q: float, bins: int = 128) -> np.float32:
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        raise ValueError("x must contain at least one value")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    x_min = np.min(x)
    x_max = np.max(x)
    if x_min == x_max:
        return np.float32(x_min)
    if q == 0:
        return np.float32(x_min)
    if q == 100:
        return np.float32(x_max)

    counts, bin_edges = np.histogram(x, bins=bins, range=(x_min, x_max))
    cumulative_counts = np.cumsum(counts, dtype=np.int64)
    target_count = (q / 100.0) * x.size
    bin_idx = int(np.searchsorted(cumulative_counts, target_count, side="left"))
    bin_idx = min(max(bin_idx, 0), len(bin_edges) - 2)

    left_edge = bin_edges[bin_idx]
    right_edge = bin_edges[bin_idx + 1]
    return np.float32((left_edge + right_edge) * 0.5)


##  detect peaks and dips in a single sample (1D signal)
def detect_peaks_and_dips(
        signal,
        min_prominence_frac=0.08,
        min_distance=30,
        min_width=3,
        percentile_method="histogram",
        histogram_bins=50,
):
    x = np.asarray(signal, dtype=np.float32)

    # signal scale used to convert fractional prominence into an absolute threshold
    if percentile_method == "histogram":
        signal_min = _histogram_percentile(x, 5.0, bins=histogram_bins)
        signal_max = _histogram_percentile(x, 95.0, bins=histogram_bins)
    elif percentile_method == "exact":
        signal_min = np.percentile(x, 5.0)
        signal_max = np.percentile(x, 95.0)
    else:
        raise ValueError(
            f"Unsupported percentile_method={percentile_method!r}. "
            "Use 'histogram' or 'exact'."
        )

    signal_range = signal_max - signal_min
    min_prominence = min_prominence_frac * signal_range

    # peaks
    peaks, peak_props = find_peaks(x, prominence=min_prominence, distance=min_distance, width=min_width)
    # dips: peaks in the negative residual
    dips, dip_props = find_peaks(-x, prominence=min_prominence, distance=min_distance, width=min_width)

    # widths at half-prominence for general peaks / dips
    peak_width_result = peak_widths(x, peaks, rel_height=0.5)
    dip_width_result = peak_widths(-x, dips, rel_height=0.5)
    # main peak width uses a deeper evaluation level
    main_peak_width_result = peak_widths(x, peaks, rel_height=0.9)

    peak_amplitudes = x[peaks]
    dip_amplitudes = x[dips]

    return {
        "peak_frequencies": peaks,
        "peak_amplitudes": peak_amplitudes,
        "peak_prominences": peak_props["prominences"],
        "peak_widths": peak_width_result[0],
        "peak_width_heights": peak_width_result[1],
        "peak_left_ips": peak_width_result[2],
        "peak_right_ips": peak_width_result[3],
        "main_peak_widths": main_peak_width_result[0],
        "main_peak_width_heights": main_peak_width_result[1],
        "main_peak_left_ips": main_peak_width_result[2],
        "main_peak_right_ips": main_peak_width_result[3],
        "peak_left_bases": peak_props["left_bases"],
        "peak_right_bases": peak_props["right_bases"],
        "dip_frequencies": dips,
        "dip_amplitudes": dip_amplitudes,
        "dip_prominences": dip_props["prominences"],
        "dip_widths": dip_width_result[0],
        "dip_width_heights": dip_width_result[1],
        "dip_left_ips": dip_width_result[2],
        "dip_right_ips": dip_width_result[3],
        "dip_left_bases": dip_props["left_bases"],
        "dip_right_bases": dip_props["right_bases"],
    }


##
import numpy as np


def select_band_peak_dip_pairs(
    detected_peak_dip: dict,
    band_edges: list[tuple[float, float]],
    peak_selection: str = "amplitude",
    dip_selection: str = "amplitude",
) -> list[dict]:
    """
    For each band:
        - if 0 peaks: save empty candidate
        - if 1 peak: save single main peak information
        - if >=2 peaks: save top two peaks and one dip between them

    Also stores exact width boundary points:
        - *_left_ips
        - *_right_ips

    Returns
    -------
    peak_dip_pairs:
        List of candidate dictionaries, one per band.
    """

    peak_freq = detected_peak_dip["peak_frequencies"]
    peak_amp = detected_peak_dip["peak_amplitudes"]
    peak_promin = detected_peak_dip["peak_prominences"]
    peak_width = detected_peak_dip["peak_widths"]
    peak_left_ips = detected_peak_dip["peak_left_ips"]
    peak_right_ips = detected_peak_dip["peak_right_ips"]
    main_peak_width = detected_peak_dip["main_peak_widths"]
    main_peak_width_heights = detected_peak_dip["main_peak_width_heights"]
    main_peak_left_ips = detected_peak_dip["main_peak_left_ips"]
    main_peak_right_ips = detected_peak_dip["main_peak_right_ips"]

    dip_freq = detected_peak_dip["dip_frequencies"]
    dip_amp = detected_peak_dip["dip_amplitudes"]
    dip_promin = detected_peak_dip["dip_prominences"]
    dip_width = detected_peak_dip["dip_widths"]
    dip_left_ips = detected_peak_dip["dip_left_ips"]
    dip_right_ips = detected_peak_dip["dip_right_ips"]

    assert len(peak_freq) == len(peak_amp) == len(peak_promin) == len(peak_width)
    assert len(peak_freq) == len(peak_left_ips) == len(peak_right_ips)

    assert len(dip_freq) == len(dip_amp) == len(dip_promin) == len(dip_width)
    assert len(dip_freq) == len(dip_left_ips) == len(dip_right_ips)

    peak_dip_pairs = []

    for band_id, (start, end) in enumerate(band_edges):
        candidate = {
            "band_id": band_id + 1,
            "band_start": start,
            "band_end": end,

            "num_peaks_in_band": 0,
            "num_dips_in_band": 0,
            "num_middle_dips": 0,

            "main_peak_exists": False,
            "pair_exists": False,
            "middle_dip_exists": False,
        }

        peak_ids = np.where((peak_freq >= start) & (peak_freq < end))[0]
        dip_ids = np.where((dip_freq >= start) & (dip_freq < end))[0]

        candidate["num_peaks_in_band"] = int(len(peak_ids))
        candidate["num_dips_in_band"] = int(len(dip_ids))

        # -------------------------
        # Case 0: no peak in band
        # -------------------------
        if len(peak_ids) == 0:
            peak_dip_pairs.append(candidate)
            continue

        # Sort peaks by selection criterion
        if peak_selection == "prominence":
            order = np.argsort(peak_promin[peak_ids])[::-1]
        elif peak_selection == "amplitude":
            order = np.argsort(peak_amp[peak_ids])[::-1]
        else:
            raise ValueError("peak_selection must be 'prominence' or 'amplitude'")

        sorted_peak_ids = peak_ids[order]

        # -------------------------
        # Always save the strongest / main peak
        # -------------------------
        main_peak_id = sorted_peak_ids[0]

        candidate.update({
            "main_peak_exists": True,

            "main_peak_freq": peak_freq[main_peak_id],
            "main_peak_amp": peak_amp[main_peak_id],
            "main_peak_prominence": peak_promin[main_peak_id],
            "main_peak_width": main_peak_width[main_peak_id],
            "main_peak_width_height": main_peak_width_heights[main_peak_id],
            "main_peak_left_ips": main_peak_left_ips[main_peak_id],
            "main_peak_right_ips": main_peak_right_ips[main_peak_id],
        })

        # -------------------------
        # Case 1: only one peak
        # -------------------------
        if len(sorted_peak_ids) == 1:
            peak_dip_pairs.append(candidate)
            continue

        # -------------------------
        # Case 2: at least two peaks
        # -------------------------
        top2_peak_ids = sorted_peak_ids[:2]

        # Sort selected peaks left-to-right by frequency
        top2_peak_ids = top2_peak_ids[np.argsort(peak_freq[top2_peak_ids])]
        left_id, right_id = top2_peak_ids

        f_left = peak_freq[left_id]
        f_right = peak_freq[right_id]

        candidate.update({
            "pair_exists": True,

            "left_peak_freq": f_left,
            "left_peak_amp": peak_amp[left_id],
            "left_peak_prominence": peak_promin[left_id],
            "left_peak_width": peak_width[left_id],
            "left_peak_left_ips": peak_left_ips[left_id],
            "left_peak_right_ips": peak_right_ips[left_id],

            "right_peak_freq": f_right,
            "right_peak_amp": peak_amp[right_id],
            "right_peak_prominence": peak_promin[right_id],
            "right_peak_width": peak_width[right_id],
            "right_peak_left_ips": peak_left_ips[right_id],
            "right_peak_right_ips": peak_right_ips[right_id],
        })

        # -------------------------
        # Find dip between selected two peaks
        # -------------------------
        middle_dip_ids = np.where((dip_freq > f_left) & (dip_freq < f_right))[0]
        candidate["num_middle_dips"] = int(len(middle_dip_ids))

        if len(middle_dip_ids) > 0:
            if dip_selection == "amplitude":
                # For dips, lower amplitude means deeper dip
                best_dip_id = middle_dip_ids[np.argmin(dip_amp[middle_dip_ids])]
            elif dip_selection == "prominence":
                best_dip_id = middle_dip_ids[np.argmax(dip_promin[middle_dip_ids])]
            else:
                raise ValueError("dip_selection must be 'amplitude' or 'prominence'")

            candidate.update({
                "middle_dip_exists": True,

                "middle_dip_freq": dip_freq[best_dip_id],
                "middle_dip_amp": dip_amp[best_dip_id],
                "middle_dip_prominence": dip_promin[best_dip_id],
                "middle_dip_width": dip_width[best_dip_id],
                "middle_dip_left_ips": dip_left_ips[best_dip_id],
                "middle_dip_right_ips": dip_right_ips[best_dip_id],
            })

        peak_dip_pairs.append(candidate)

    return peak_dip_pairs


##
def calculate_doublet_features(
    selected_pairs: list[dict],
    eps: float = 1e-8,
) -> dict:
    """
    Calculate fixed-length doublet features from selected band-wise peak-dip-pair output.

    Input:
        selected_pairs:
            Output from select_band_peak_dip_pairs()

    Output:
        features: peak/dip frequency, amplitude, prominence, width, Q, balance, imbalance, doublet score.
            Dictionary of fixed-length features.
    """

    features = {}

    for pair in selected_pairs:
        band_id = pair["band_id"]
        prefix = f"band{band_id}"

        # -------------------------
        # Basic existence/count flags
        # -------------------------
        features[f"{prefix}_main_peak_exists"] = float(pair.get("main_peak_exists", False))
        features[f"{prefix}_pair_exists"] = float(pair.get("pair_exists", False))
        features[f"{prefix}_middle_dip_exists"] = float(pair.get("middle_dip_exists", False))

        features[f"{prefix}_num_peaks_in_band"] = float(pair.get("num_peaks_in_band", 0))
        features[f"{prefix}_num_dips_in_band"] = float(pair.get("num_dips_in_band", 0))
        features[f"{prefix}_num_middle_dips"] = float(pair.get("num_middle_dips", 0))

        # -------------------------
        # Default values
        # -------------------------
        default_keys = [
            # main peak
            "main_peak_freq", "main_peak_amp", "main_peak_prominence", "main_peak_width", "main_peak_Q",
            # left peak
            "left_peak_freq", "left_peak_amp", "left_peak_prominence", "left_peak_width", "left_peak_Q",
            # right peak
            "right_peak_freq", "right_peak_amp", "right_peak_prominence", "right_peak_width", "right_peak_Q",
            # peak-pair geometry
            "peak_separation", "peak_balance_amp", "peak_balance_prominence", "peak_balance_width",
            "signed_amp_imbalance", "signed_prominence_imbalance", "signed_width_imbalance",
            # middle dip
            "middle_dip_freq", "middle_dip_amp", "middle_dip_prominence", "middle_dip_width", "middle_dip_Q",
            # doublet morphology
            "dip_depth", "dip_depth_norm", "dip_relative_position", "doublet_score",
        ]

        for key in default_keys:
            features[f"{prefix}_{key}"] = 0.0

        # -------------------------
        # Main single peak features
        # -------------------------
        if pair.get("main_peak_exists", False):
            main_f = pair["main_peak_freq"]
            main_w = pair["main_peak_width"]

            features[f"{prefix}_main_peak_freq"] = main_f
            features[f"{prefix}_main_peak_amp"] = pair["main_peak_amp"]
            features[f"{prefix}_main_peak_prominence"] = pair["main_peak_prominence"]
            features[f"{prefix}_main_peak_width"] = main_w
            features[f"{prefix}_main_peak_Q"] = main_f / max(main_w, eps)

        # -------------------------
        # Pair-based features
        # -------------------------
        if not pair.get("pair_exists", False):
            continue

        left_f = pair["left_peak_freq"]
        right_f = pair["right_peak_freq"]

        left_amp = pair["left_peak_amp"]
        right_amp = pair["right_peak_amp"]

        left_prom = pair["left_peak_prominence"]
        right_prom = pair["right_peak_prominence"]

        left_w = pair["left_peak_width"]
        right_w = pair["right_peak_width"]

        separation = right_f - left_f
        # Balance features: 1 = balanced, 0 = highly imbalanced
        amp_balance = min(abs(left_amp), abs(right_amp)) / max(max(abs(left_amp), abs(right_amp)), eps)
        prom_balance = min(abs(left_prom), abs(right_prom)) / max(max(abs(left_prom), abs(right_prom)), eps)
        width_balance = min(abs(left_w), abs(right_w)) / max(max(abs(left_w), abs(right_w)), eps)

        # Signed imbalance features: positive = left larger, negative = right larger
        signed_amp_imbalance = (left_amp - right_amp) / max(abs(left_amp) + abs(right_amp), eps)
        signed_prominence_imbalance = (left_prom - right_prom) / max(abs(left_prom) + abs(right_prom), eps)
        signed_width_imbalance = (left_w - right_w) / max(abs(left_w) + abs(right_w), eps)

        features[f"{prefix}_left_peak_freq"] = left_f
        features[f"{prefix}_left_peak_amp"] = left_amp
        features[f"{prefix}_left_peak_prominence"] = left_prom
        features[f"{prefix}_left_peak_width"] = left_w
        features[f"{prefix}_left_peak_Q"] = left_f / max(left_w, eps)

        features[f"{prefix}_right_peak_freq"] = right_f
        features[f"{prefix}_right_peak_amp"] = right_amp
        features[f"{prefix}_right_peak_prominence"] = right_prom
        features[f"{prefix}_right_peak_width"] = right_w
        features[f"{prefix}_right_peak_Q"] = right_f / max(right_w, eps)

        features[f"{prefix}_peak_separation"] = separation
        features[f"{prefix}_peak_balance_amp"] = amp_balance
        features[f"{prefix}_peak_balance_prominence"] = prom_balance
        features[f"{prefix}_peak_balance_width"] = width_balance

        features[f"{prefix}_signed_amp_imbalance"] = signed_amp_imbalance
        features[f"{prefix}_signed_prominence_imbalance"] = signed_prominence_imbalance
        features[f"{prefix}_signed_width_imbalance"] = signed_width_imbalance

        # -------------------------
        # Middle dip-based features
        # -------------------------
        if not pair.get("middle_dip_exists", False):
            continue

        dip_f = pair["middle_dip_freq"]
        dip_amp = pair["middle_dip_amp"]
        dip_w = pair["middle_dip_width"]

        dip_depth = max(min(left_amp, right_amp) - dip_amp, 0.0)
        dip_depth_norm = dip_depth / max(min(abs(left_amp), abs(right_amp)), eps)
        dip_relative_position = (dip_f - left_f) / max(separation, eps)
        # Uses amp_balance so the score rewards balanced peak-dip-peak structures
        doublet_score = dip_depth * amp_balance / max(separation, eps)

        features[f"{prefix}_middle_dip_freq"] = dip_f
        features[f"{prefix}_middle_dip_amp"] = dip_amp
        features[f"{prefix}_middle_dip_prominence"] = pair["middle_dip_prominence"]
        features[f"{prefix}_middle_dip_width"] = dip_w
        features[f"{prefix}_middle_dip_Q"] = dip_f / max(dip_w, eps)

        features[f"{prefix}_dip_depth"] = dip_depth
        features[f"{prefix}_dip_depth_norm"] = dip_depth_norm
        features[f"{prefix}_dip_relative_position"] = dip_relative_position
        features[f"{prefix}_doublet_score"] = doublet_score

    return features


def _safe_index(idx: float, n: int) -> int:
    """Convert a float index to an integer index within a valid range ."""
    return int(np.clip(round(idx), 0, n - 1))


def _positive_area(
    x: np.ndarray,
    start_idx: float,
    end_idx: float,
    baseline: float,
    mode: str,
) -> float:
    """
    Calculate positive area relative to a baseline.

    mode="peak":
        area = sum(max(x - baseline, 0))

    mode="dip":
        area = sum(max(baseline - x, 0))
    """

    n = len(x)

    start = _safe_index(start_idx, n)
    end = _safe_index(end_idx, n)

    segment = x[start:end + 1]

    if mode == "peak":
        area = np.sum(np.maximum(segment - baseline, 0.0))
    elif mode == "dip":
        area = np.sum(np.maximum(baseline - segment, 0.0))
    else:
        raise ValueError("mode must be 'peak' or 'dip'")

    return float(area)


##
def calculate_doublet_area_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    eps: float = 1e-8,
) -> dict:
    """
    Calculate area-based features using exact width boundaries:
        left_ips and right_ips.

    Requires selected_pairs to contain:
        main_peak_left_ips, main_peak_right_ips
        left_peak_left_ips, left_peak_right_ips
        right_peak_left_ips, right_peak_right_ips
        middle_dip_left_ips, middle_dip_right_ips
    """

    signal = np.asarray(signal, dtype=np.float64)
    features = {}

    for pair in selected_pairs:
        band_id = pair["band_id"]
        prefix = f"band{band_id}"

        # -------------------------
        # Default values
        # -------------------------
        default_keys = [
            "main_peak_area", "main_peak_area_norm", "left_peak_area", "left_peak_area_norm",
            "right_peak_area", "right_peak_area_norm", "middle_dip_area", "middle_dip_area_norm",
            "doublet_valley_area", "doublet_valley_area_norm",
        ]

        for key in default_keys:
            features[f"{prefix}_{key}"] = 0.0

        # -------------------------
        # Main peak area
        # -------------------------
        if pair.get("main_peak_exists", False):
            peak_amp = pair["main_peak_amp"]
            peak_prom = pair["main_peak_prominence"]
            peak_width = pair["main_peak_width"]

            # prominence = peak_amp - local_baseline
            baseline = peak_amp - peak_prom

            area = _positive_area(
                x=signal,
                start_idx=pair["main_peak_left_ips"],
                end_idx=pair["main_peak_right_ips"],
                baseline=baseline,
                mode="peak",
            )

            features[f"{prefix}_main_peak_area"] = area
            features[f"{prefix}_main_peak_area_norm"] = area / max(peak_prom * peak_width, eps)

        # -------------------------
        # Peak pair for left/right peak area
        # -------------------------
        if not pair.get("pair_exists", False):
            continue

        left_f = pair["left_peak_freq"]
        right_f = pair["right_peak_freq"]

        left_amp = pair["left_peak_amp"]
        right_amp = pair["right_peak_amp"]

        left_prom = pair["left_peak_prominence"]
        right_prom = pair["right_peak_prominence"]

        left_width = pair["left_peak_width"]
        right_width = pair["right_peak_width"]

        # -------------------------
        # Left peak area
        # -------------------------
        left_baseline = left_amp - left_prom

        left_area = _positive_area(
            x=signal,
            start_idx=pair["left_peak_left_ips"],
            end_idx=pair["left_peak_right_ips"],
            baseline=left_baseline,
            mode="peak",
        )

        features[f"{prefix}_left_peak_area"] = left_area
        features[f"{prefix}_left_peak_area_norm"] = left_area / max(left_prom * left_width, eps)

        # -------------------------
        # Right peak area
        # -------------------------
        right_baseline = right_amp - right_prom

        right_area = _positive_area(
            x=signal,
            start_idx=pair["right_peak_left_ips"],
            end_idx=pair["right_peak_right_ips"],
            baseline=right_baseline,
            mode="peak",
        )

        features[f"{prefix}_right_peak_area"] = right_area
        features[f"{prefix}_right_peak_area_norm"] = right_area / max(right_prom * right_width, eps)

        # -------------------------
        # Middle dip area
        # -------------------------
        if not pair.get("middle_dip_exists", False):
            continue

        dip_amp = pair["middle_dip_amp"]
        dip_prom = pair["middle_dip_prominence"]
        dip_width = pair["middle_dip_width"]

        # For dips, prominence is measured on -signal.
        # So local baseline is approximately:
        # baseline = dip_amp + dip_prom
        dip_baseline = dip_amp + dip_prom

        dip_area = _positive_area(
            x=signal,
            start_idx=pair["middle_dip_left_ips"],
            end_idx=pair["middle_dip_right_ips"],
            baseline=dip_baseline,
            mode="dip",
        )

        features[f"{prefix}_middle_dip_area"] = dip_area
        features[f"{prefix}_middle_dip_area_norm"] = dip_area / max(dip_prom * dip_width, eps)

        # -------------------------
        # Doublet valley area
        # -------------------------
        # This is not limited to the dip_width only.
        # It measures the whole valley between the two selected peaks.
        valley_baseline = min(left_amp, right_amp)
        peak_separation = right_f - left_f

        doublet_valley_area = _positive_area(
            x=signal,
            start_idx=left_f,
            end_idx=right_f,
            baseline=valley_baseline,
            mode="dip",
        )

        dip_depth = max(min(left_amp, right_amp) - dip_amp, 0.0)

        features[f"{prefix}_doublet_valley_area"] = doublet_valley_area
        features[f"{prefix}_doublet_valley_area_norm"] = (
            doublet_valley_area / max(dip_depth * peak_separation, eps)
        )

    return features


