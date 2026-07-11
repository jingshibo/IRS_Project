from typing import Optional

import numpy as np


def _safe_index(idx: float, n: int) -> int:
    """Convert a float index to an integer index within a valid range."""
    return int(np.clip(round(float(idx)), 0, n - 1))


def _segment_bounds(start_idx: float, end_idx: float, n: int) -> tuple[int, int]:
    start = _safe_index(start_idx, n)
    end = _safe_index(end_idx, n)
    if start > end:
        start, end = end, start
    return start, end


def _window_bounds(center_idx: float, radius: int, n: int) -> tuple[int, int]:
    center = _safe_index(center_idx, n)
    radius = max(int(radius), 0)
    start = max(center - radius, 0)
    end = min(center + radius, n - 1)
    return start, end


def _safe_ratio(numerator: float, denominator: float, eps: float) -> float:
    return float(numerator) / max(abs(float(denominator)), eps)


def _prepare_signal_and_derivative(
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32).ravel()
    if first_derivative is None:
        first_derivative = np.gradient(signal).astype(np.float32)
    else:
        first_derivative = np.asarray(first_derivative, dtype=np.float32).ravel()

    if signal.shape != first_derivative.shape:
        raise ValueError(
            "signal and first_derivative must have the same shape, "
            f"got signal={signal.shape}, first_derivative={first_derivative.shape}"
        )

    return signal, first_derivative


def _inter_band_endpoint(pair: dict, side: str) -> Optional[tuple[float, float]]:
    if side == "right" and pair.get("pair_exists", False):
        return float(pair["right_peak_freq"]), float(pair["right_peak_amp"])
    if side == "left" and pair.get("pair_exists", False):
        return float(pair["left_peak_freq"]), float(pair["left_peak_amp"])
    if pair.get("main_peak_exists", False):
        return float(pair["main_peak_freq"]), float(pair["main_peak_amp"])
    return None


def _abs_derivative_summary(
    first_derivative: np.ndarray,
    start_idx: float,
    end_idx: float,
) -> tuple[float, float, float]:
    start, end = _segment_bounds(start_idx, end_idx, len(first_derivative))
    segment = first_derivative[start:end + 1]
    abs_segment = np.abs(segment)
    return (
        float(np.max(abs_segment)),
        float(np.mean(abs_segment)),
        float(np.sum(segment * segment)),
    )


def _signed_derivative_summary(
    first_derivative: np.ndarray,
    start_idx: float,
    end_idx: float,
) -> tuple[float, float]:
    start, end = _segment_bounds(start_idx, end_idx, len(first_derivative))
    segment = first_derivative[start:end + 1]
    return float(np.max(segment)), float(np.min(segment))


def _derivative_variability_stats(
    derivative_segment: np.ndarray,
    eps: float,
) -> tuple[float, float, float]:
    derivative_segment = np.asarray(derivative_segment, dtype=np.float32).ravel()
    if derivative_segment.size == 0:
        return 0.0, 0.0, 0.0

    derivative_std = float(np.std(derivative_segment))
    nonzero_values = derivative_segment[np.abs(derivative_segment) > eps]
    if nonzero_values.size < 2:
        zero_crossing_count = 0
    else:
        signs = np.sign(nonzero_values)
        zero_crossing_count = int(np.count_nonzero(signs[:-1] * signs[1:] < 0))
    zero_crossing_rate = zero_crossing_count / max(derivative_segment.size - 1, 1)
    return derivative_std, float(zero_crossing_count), float(zero_crossing_rate)


def _derivative_variability_summary(
    first_derivative: np.ndarray,
    start_idx: float,
    end_idx: float,
    eps: float,
) -> tuple[float, float, float]:
    start, end = _segment_bounds(start_idx, end_idx, len(first_derivative))
    return _derivative_variability_stats(first_derivative[start:end + 1], eps)


def _derivative_extrema_location_summary(
    first_derivative: np.ndarray,
    start_idx: float,
    end_idx: float,
) -> tuple[float, float, float, float, float, float]:
    start, end = _segment_bounds(start_idx, end_idx, len(first_derivative))
    derivative_segment = first_derivative[start:end + 1]
    if derivative_segment.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    max_abs_pos = start + int(np.argmax(np.abs(derivative_segment)))
    max_positive_pos = start + int(np.argmax(derivative_segment))
    min_negative_pos = start + int(np.argmin(derivative_segment))
    denom = max(end - start, 1)

    return (
        float(max_abs_pos),
        float((max_abs_pos - start) / denom),
        float(max_positive_pos),
        float((max_positive_pos - start) / denom),
        float(min_negative_pos),
        float((min_negative_pos - start) / denom),
    )


def _add_segment_derivative_features(
    features: dict,
    prefix: str,
    segment_name: str,
    start_freq: float,
    start_amp: float,
    end_freq: float,
    end_amp: float,
    first_derivative: np.ndarray,
    eps: float,
) -> None:
    start_idx, end_idx = _segment_bounds(start_freq, end_freq, len(first_derivative))
    derivative_segment = first_derivative[start_idx:end_idx + 1]
    if derivative_segment.size == 0:
        return

    length = abs(end_freq - start_freq)
    net_amplitude_change = end_amp - start_amp
    slope = net_amplitude_change / max(length, eps)
    derivative_std, zero_crossing_count, zero_crossing_rate = _derivative_variability_stats(
        derivative_segment,
        eps,
    )
    (
        max_abs_pos,
        max_abs_rel_pos,
        max_positive_pos,
        max_positive_rel_pos,
        min_negative_pos,
        min_negative_rel_pos,
    ) = _derivative_extrema_location_summary(first_derivative, start_freq, end_freq)

    features[f"{prefix}_{segment_name}_exists"] = np.float32(1.0)
    features[f"{prefix}_{segment_name}_start_freq"] = np.float32(start_freq)
    features[f"{prefix}_{segment_name}_end_freq"] = np.float32(end_freq)
    features[f"{prefix}_{segment_name}_length"] = np.float32(length)
    features[f"{prefix}_{segment_name}_net_amplitude_change"] = np.float32(net_amplitude_change)
    features[f"{prefix}_{segment_name}_slope"] = np.float32(slope)
    features[f"{prefix}_{segment_name}_mean_first_derivative"] = np.float32(np.mean(derivative_segment))
    features[f"{prefix}_{segment_name}_mean_abs_first_derivative"] = np.float32(np.mean(np.abs(derivative_segment)))
    features[f"{prefix}_{segment_name}_std_first_derivative"] = np.float32(derivative_std)
    features[f"{prefix}_{segment_name}_zero_crossing_count"] = np.float32(zero_crossing_count)
    features[f"{prefix}_{segment_name}_zero_crossing_rate"] = np.float32(zero_crossing_rate)
    features[f"{prefix}_{segment_name}_max_abs_first_derivative"] = np.float32(np.max(np.abs(derivative_segment)))
    features[f"{prefix}_{segment_name}_max_abs_first_derivative_position"] = np.float32(max_abs_pos)
    features[f"{prefix}_{segment_name}_max_abs_first_derivative_relative_position"] = np.float32(max_abs_rel_pos)
    features[f"{prefix}_{segment_name}_first_derivative_energy"] = np.float32(np.sum(derivative_segment * derivative_segment))
    features[f"{prefix}_{segment_name}_max_positive_first_derivative"] = np.float32(np.max(derivative_segment))
    features[f"{prefix}_{segment_name}_max_positive_first_derivative_position"] = np.float32(max_positive_pos)
    features[f"{prefix}_{segment_name}_max_positive_first_derivative_relative_position"] = np.float32(max_positive_rel_pos)
    features[f"{prefix}_{segment_name}_min_negative_first_derivative"] = np.float32(np.min(derivative_segment))
    features[f"{prefix}_{segment_name}_min_negative_first_derivative_position"] = np.float32(min_negative_pos)
    features[f"{prefix}_{segment_name}_min_negative_first_derivative_relative_position"] = np.float32(min_negative_rel_pos)
    features[f"{prefix}_{segment_name}_positive_derivative_area"] = np.float32(np.sum(np.maximum(derivative_segment, 0.0)))
    features[f"{prefix}_{segment_name}_negative_derivative_area"] = np.float32(np.sum(np.maximum(-derivative_segment, 0.0)))


def _add_empty_segment_derivative_features(features: dict, prefix: str, segment_name: str) -> None:
    keys = [
        "exists",
        "start_freq",
        "end_freq",
        "length",
        "net_amplitude_change",
        "slope",
        "mean_first_derivative",
        "mean_abs_first_derivative",
        "std_first_derivative",
        "zero_crossing_count",
        "zero_crossing_rate",
        "max_abs_first_derivative",
        "max_abs_first_derivative_position",
        "max_abs_first_derivative_relative_position",
        "first_derivative_energy",
        "max_positive_first_derivative",
        "max_positive_first_derivative_position",
        "max_positive_first_derivative_relative_position",
        "min_negative_first_derivative",
        "min_negative_first_derivative_position",
        "min_negative_first_derivative_relative_position",
        "positive_derivative_area",
        "negative_derivative_area",
    ]
    for key in keys:
        features[f"{prefix}_{segment_name}_{key}"] = np.float32(0.0)


def calculate_within_band_derivative_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray] = None,
    window_radius: int = 5, # 5 corresponds to 55 original points
    eps: float = 1e-8,
) -> dict:
    """
    Calculate first-derivative shape features around selected band-wise peak/dip pairs.

    Parameters
    ----------
    selected_pairs:
        Output from Peak_Dip_Features.select_band_peak_dip_pairs().
    signal:
        Original 1D signal used for peak/dip detection.
    first_derivative:
        Optional first-derivative signal aligned with `signal`. If not provided,
        np.gradient(signal) is used.
    window_radius:
        Number of points on each side of the peak used for local derivative summaries.
    eps:
        Small value used to avoid division by zero.

    Returns
    -------
    features:
        Fixed-length dictionary of first-derivative features per band.
    """
    signal, first_derivative = _prepare_signal_and_derivative(signal, first_derivative)

    features = {}

    default_keys = [
        "main_peak_left_slope",
        "main_peak_right_slope",
        "main_peak_slope_balance",
        "main_peak_signed_slope_imbalance",
        "main_peak_max_abs_first_derivative",
        "main_peak_mean_abs_first_derivative",
        "main_peak_std_first_derivative",
        "main_peak_zero_crossing_count",
        "main_peak_zero_crossing_rate",
        "main_peak_first_derivative_energy",
        "main_peak_max_positive_first_derivative",
        "main_peak_max_positive_first_derivative_position",
        "main_peak_max_positive_first_derivative_relative_position",
        "main_peak_min_negative_first_derivative",
        "main_peak_min_negative_first_derivative_position",
        "main_peak_min_negative_first_derivative_relative_position",
        "main_peak_max_abs_first_derivative_position",
        "main_peak_max_abs_first_derivative_relative_position",
        "left_dip_to_peak_slope", # left peak
        "right_dip_to_peak_slope", # right peak
        "dip_to_peak_slope_balance",
        "dip_to_peak_signed_slope_imbalance",
        "left_dip_to_peak_max_abs_first_derivative",
        "left_dip_to_peak_mean_abs_first_derivative",
        "left_dip_to_peak_std_first_derivative",
        "left_dip_to_peak_zero_crossing_count",
        "left_dip_to_peak_zero_crossing_rate",
        "left_dip_to_peak_first_derivative_energy",
        "left_dip_to_peak_max_positive_first_derivative",
        "left_dip_to_peak_max_abs_first_derivative_position",
        "left_dip_to_peak_max_abs_first_derivative_relative_position",
        "left_dip_to_peak_max_positive_first_derivative_position",
        "left_dip_to_peak_max_positive_first_derivative_relative_position",
        "left_dip_to_peak_min_negative_first_derivative",
        "left_dip_to_peak_min_negative_first_derivative_position",
        "left_dip_to_peak_min_negative_first_derivative_relative_position",
        "right_dip_to_peak_max_abs_first_derivative",
        "right_dip_to_peak_mean_abs_first_derivative",
        "right_dip_to_peak_std_first_derivative",
        "right_dip_to_peak_zero_crossing_count",
        "right_dip_to_peak_zero_crossing_rate",
        "right_dip_to_peak_first_derivative_energy",
        "right_dip_to_peak_max_positive_first_derivative",
        "right_dip_to_peak_max_abs_first_derivative_position",
        "right_dip_to_peak_max_abs_first_derivative_relative_position",
        "right_dip_to_peak_max_positive_first_derivative_position",
        "right_dip_to_peak_max_positive_first_derivative_relative_position",
        "right_dip_to_peak_min_negative_first_derivative",
        "right_dip_to_peak_min_negative_first_derivative_position",
        "right_dip_to_peak_min_negative_first_derivative_relative_position",
    ]

    for pair in selected_pairs:
        band_id = pair["band_id"]
        prefix = f"band{band_id}"

        for key in default_keys:
            features[f"{prefix}_{key}"] = np.float32(0.0)

        if pair.get("main_peak_exists", False):
            peak_amp = float(pair["main_peak_amp"])
            peak_freq = float(pair["main_peak_freq"])
            peak_width_height = float(pair["main_peak_width_height"])
            left_ips = float(pair["main_peak_left_ips"])
            right_ips = float(pair["main_peak_right_ips"])

            left_slope = _safe_ratio(peak_amp - peak_width_height, peak_freq - left_ips, eps)
            right_slope = _safe_ratio(peak_amp - peak_width_height, right_ips - peak_freq, eps)
            slope_balance = min(abs(left_slope), abs(right_slope)) / max(max(abs(left_slope), abs(right_slope)), eps)
            signed_slope_imbalance = (left_slope - right_slope) / max(abs(left_slope) + abs(right_slope), eps)

            win_start, win_end = _window_bounds(peak_freq, window_radius, len(first_derivative))
            max_abs_d, mean_abs_d, energy_d = _abs_derivative_summary(first_derivative, win_start, win_end)
            max_positive_d, min_negative_d = _signed_derivative_summary(first_derivative, win_start, win_end)
            std_d, zc_count_d, zc_rate_d = _derivative_variability_summary(
                first_derivative,
                win_start,
                win_end,
                eps,
            )
            (
                max_abs_pos_d,
                max_abs_rel_pos_d,
                max_positive_pos_d,
                max_positive_rel_pos_d,
                min_negative_pos_d,
                min_negative_rel_pos_d,
            ) = _derivative_extrema_location_summary(first_derivative, win_start, win_end)

            features[f"{prefix}_main_peak_left_slope"] = np.float32(left_slope)
            features[f"{prefix}_main_peak_right_slope"] = np.float32(right_slope)
            features[f"{prefix}_main_peak_slope_balance"] = np.float32(slope_balance)
            features[f"{prefix}_main_peak_signed_slope_imbalance"] = np.float32(signed_slope_imbalance)
            features[f"{prefix}_main_peak_max_abs_first_derivative"] = np.float32(max_abs_d)
            features[f"{prefix}_main_peak_mean_abs_first_derivative"] = np.float32(mean_abs_d)
            features[f"{prefix}_main_peak_std_first_derivative"] = np.float32(std_d)
            features[f"{prefix}_main_peak_zero_crossing_count"] = np.float32(zc_count_d)
            features[f"{prefix}_main_peak_zero_crossing_rate"] = np.float32(zc_rate_d)
            features[f"{prefix}_main_peak_max_abs_first_derivative_position"] = np.float32(max_abs_pos_d)
            features[f"{prefix}_main_peak_max_abs_first_derivative_relative_position"] = np.float32(max_abs_rel_pos_d)
            features[f"{prefix}_main_peak_first_derivative_energy"] = np.float32(energy_d)
            features[f"{prefix}_main_peak_max_positive_first_derivative"] = np.float32(max_positive_d)
            features[f"{prefix}_main_peak_max_positive_first_derivative_position"] = np.float32(max_positive_pos_d)
            features[f"{prefix}_main_peak_max_positive_first_derivative_relative_position"] = np.float32(max_positive_rel_pos_d)
            features[f"{prefix}_main_peak_min_negative_first_derivative"] = np.float32(min_negative_d)
            features[f"{prefix}_main_peak_min_negative_first_derivative_position"] = np.float32(min_negative_pos_d)
            features[f"{prefix}_main_peak_min_negative_first_derivative_relative_position"] = np.float32(min_negative_rel_pos_d)

        if not pair.get("pair_exists", False) or not pair.get("middle_dip_exists", False):
            continue

        left_peak_freq = float(pair["left_peak_freq"])
        right_peak_freq = float(pair["right_peak_freq"])
        left_peak_amp = float(pair["left_peak_amp"])
        right_peak_amp = float(pair["right_peak_amp"])
        dip_freq = float(pair["middle_dip_freq"])
        dip_amp = float(pair["middle_dip_amp"])

        left_dip_to_peak_slope = _safe_ratio(left_peak_amp - dip_amp, dip_freq - left_peak_freq, eps)
        right_dip_to_peak_slope = _safe_ratio(right_peak_amp - dip_amp, right_peak_freq - dip_freq, eps)
        dip_to_peak_slope_balance = min(abs(left_dip_to_peak_slope), abs(right_dip_to_peak_slope)) / max(
            max(abs(left_dip_to_peak_slope), abs(right_dip_to_peak_slope)),
            eps,
        )
        dip_to_peak_signed_slope_imbalance = (
            (left_dip_to_peak_slope - right_dip_to_peak_slope)
            / max(abs(left_dip_to_peak_slope) + abs(right_dip_to_peak_slope), eps)
        )

        left_max_abs_d, left_mean_abs_d, left_energy_d = _abs_derivative_summary(
            first_derivative,
            left_peak_freq,
            dip_freq,
        )
        left_max_positive_d, left_min_negative_d = _signed_derivative_summary(
            first_derivative,
            left_peak_freq,
            dip_freq,
        )
        left_std_d, left_zc_count_d, left_zc_rate_d = _derivative_variability_summary(
            first_derivative,
            left_peak_freq,
            dip_freq,
            eps,
        )
        (
            left_max_abs_pos_d,
            left_max_abs_rel_pos_d,
            left_max_positive_pos_d,
            left_max_positive_rel_pos_d,
            left_min_negative_pos_d,
            left_min_negative_rel_pos_d,
        ) = _derivative_extrema_location_summary(first_derivative, left_peak_freq, dip_freq)
        right_max_abs_d, right_mean_abs_d, right_energy_d = _abs_derivative_summary(
            first_derivative,
            dip_freq,
            right_peak_freq,
        )
        right_max_positive_d, right_min_negative_d = _signed_derivative_summary(
            first_derivative,
            dip_freq,
            right_peak_freq,
        )
        right_std_d, right_zc_count_d, right_zc_rate_d = _derivative_variability_summary(
            first_derivative,
            dip_freq,
            right_peak_freq,
            eps,
        )
        (
            right_max_abs_pos_d,
            right_max_abs_rel_pos_d,
            right_max_positive_pos_d,
            right_max_positive_rel_pos_d,
            right_min_negative_pos_d,
            right_min_negative_rel_pos_d,
        ) = _derivative_extrema_location_summary(first_derivative, dip_freq, right_peak_freq)

        features[f"{prefix}_left_dip_to_peak_slope"] = np.float32(left_dip_to_peak_slope)
        features[f"{prefix}_right_dip_to_peak_slope"] = np.float32(right_dip_to_peak_slope)
        features[f"{prefix}_dip_to_peak_slope_balance"] = np.float32(dip_to_peak_slope_balance)
        features[f"{prefix}_dip_to_peak_signed_slope_imbalance"] = np.float32(dip_to_peak_signed_slope_imbalance)
        features[f"{prefix}_left_dip_to_peak_max_abs_first_derivative"] = np.float32(left_max_abs_d)
        features[f"{prefix}_left_dip_to_peak_mean_abs_first_derivative"] = np.float32(left_mean_abs_d)
        features[f"{prefix}_left_dip_to_peak_std_first_derivative"] = np.float32(left_std_d)
        features[f"{prefix}_left_dip_to_peak_zero_crossing_count"] = np.float32(left_zc_count_d)
        features[f"{prefix}_left_dip_to_peak_zero_crossing_rate"] = np.float32(left_zc_rate_d)
        features[f"{prefix}_left_dip_to_peak_max_abs_first_derivative_position"] = np.float32(left_max_abs_pos_d)
        features[f"{prefix}_left_dip_to_peak_max_abs_first_derivative_relative_position"] = np.float32(left_max_abs_rel_pos_d)
        features[f"{prefix}_left_dip_to_peak_max_positive_first_derivative"] = np.float32(left_max_positive_d)
        features[f"{prefix}_left_dip_to_peak_max_positive_first_derivative_position"] = np.float32(left_max_positive_pos_d)
        features[f"{prefix}_left_dip_to_peak_max_positive_first_derivative_relative_position"] = np.float32(left_max_positive_rel_pos_d)
        features[f"{prefix}_left_dip_to_peak_min_negative_first_derivative"] = np.float32(left_min_negative_d)
        features[f"{prefix}_left_dip_to_peak_min_negative_first_derivative_position"] = np.float32(left_min_negative_pos_d)
        features[f"{prefix}_left_dip_to_peak_min_negative_first_derivative_relative_position"] = np.float32(left_min_negative_rel_pos_d)
        features[f"{prefix}_left_dip_to_peak_first_derivative_energy"] = np.float32(left_energy_d)
        features[f"{prefix}_right_dip_to_peak_max_abs_first_derivative"] = np.float32(right_max_abs_d)
        features[f"{prefix}_right_dip_to_peak_mean_abs_first_derivative"] = np.float32(right_mean_abs_d)
        features[f"{prefix}_right_dip_to_peak_std_first_derivative"] = np.float32(right_std_d)
        features[f"{prefix}_right_dip_to_peak_zero_crossing_count"] = np.float32(right_zc_count_d)
        features[f"{prefix}_right_dip_to_peak_zero_crossing_rate"] = np.float32(right_zc_rate_d)
        features[f"{prefix}_right_dip_to_peak_max_abs_first_derivative_position"] = np.float32(right_max_abs_pos_d)
        features[f"{prefix}_right_dip_to_peak_max_abs_first_derivative_relative_position"] = np.float32(right_max_abs_rel_pos_d)
        features[f"{prefix}_right_dip_to_peak_max_positive_first_derivative"] = np.float32(right_max_positive_d)
        features[f"{prefix}_right_dip_to_peak_max_positive_first_derivative_position"] = np.float32(right_max_positive_pos_d)
        features[f"{prefix}_right_dip_to_peak_max_positive_first_derivative_relative_position"] = np.float32(right_max_positive_rel_pos_d)
        features[f"{prefix}_right_dip_to_peak_min_negative_first_derivative"] = np.float32(right_min_negative_d)
        features[f"{prefix}_right_dip_to_peak_min_negative_first_derivative_position"] = np.float32(right_min_negative_pos_d)
        features[f"{prefix}_right_dip_to_peak_min_negative_first_derivative_relative_position"] = np.float32(right_min_negative_rel_pos_d)
        features[f"{prefix}_right_dip_to_peak_first_derivative_energy"] = np.float32(right_energy_d)

    return features


def calculate_inter_band_derivative_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray] = None,
    detected_peak_dip: Optional[dict] = None,
    include_broad_transition: bool = True,
    eps: float = 1e-8,
) -> dict:
    """
    Calculate first-derivative transition features between adjacent bands.

    For each adjacent pair of bands, the transition is measured from:
        previous band right peak if a pair exists, otherwise previous band main peak
    to:
        next band left peak if a pair exists, otherwise next band main peak.

    If detected_peak_dip is provided, the broad transition is also split into:
        peak_to_first_dip
        first_dip_to_last_dip
        last_dip_to_peak
    """
    signal, first_derivative = _prepare_signal_and_derivative(signal, first_derivative)
    sorted_pairs = sorted(selected_pairs, key=lambda pair: pair["band_id"]) # make sure the order is correct
    features = {}
    dip_freq = np.asarray([], dtype=np.float32)
    dip_amp = np.asarray([], dtype=np.float32)
    if detected_peak_dip is not None:
        dip_freq = np.asarray(detected_peak_dip.get("dip_frequencies", []), dtype=np.float32)
        dip_amp = np.asarray(detected_peak_dip.get("dip_amplitudes", []), dtype=np.float32)

    default_keys = [
        "transition_exists",
        "transition_start_freq",
        "transition_end_freq",
        "transition_length",
        "transition_net_amplitude_change",
        "transition_slope",
        "transition_mean_first_derivative",
        "transition_mean_abs_first_derivative",
        "transition_std_first_derivative",
        "transition_zero_crossing_count",
        "transition_zero_crossing_rate",
        "transition_max_abs_first_derivative",
        "transition_max_abs_first_derivative_position",
        "transition_max_abs_first_derivative_relative_position",
        "transition_first_derivative_energy",
        "transition_max_positive_first_derivative",
        "transition_max_positive_first_derivative_position",
        "transition_max_positive_first_derivative_relative_position",
        "transition_min_negative_first_derivative",
        "transition_min_negative_first_derivative_position",
        "transition_min_negative_first_derivative_relative_position",
        "transition_positive_derivative_area",
        "transition_negative_derivative_area",
        "num_inter_band_dips",
        "first_inter_band_dip_freq",
        "last_inter_band_dip_freq",
    ]
    piece_names = [
        "peak_to_first_dip",
        "first_dip_to_last_dip",
        "last_dip_to_peak",
    ]

    for left_pair, right_pair in zip(sorted_pairs[:-1], sorted_pairs[1:]): # i.e., zip([band1, band2], [band2, band3])
        left_band_id = left_pair["band_id"]
        right_band_id = right_pair["band_id"]
        prefix = f"band{left_band_id}_to_band{right_band_id}"

        for key in default_keys:
            features[f"{prefix}_{key}"] = np.float32(0.0)
        for piece_name in piece_names:
            _add_empty_segment_derivative_features(features, prefix, piece_name)

        start_endpoint = _inter_band_endpoint(left_pair, side="right")
        end_endpoint = _inter_band_endpoint(right_pair, side="left")
        if start_endpoint is None or end_endpoint is None:
            continue

        start_freq, start_amp = start_endpoint
        end_freq, end_amp = end_endpoint
        start_idx, end_idx = _segment_bounds(start_freq, end_freq, len(signal))
        derivative_segment = first_derivative[start_idx:end_idx + 1]

        if derivative_segment.size == 0: # if no derivative segment
            continue

        if include_broad_transition:
            transition_length = abs(end_freq - start_freq)
            net_amplitude_change = end_amp - start_amp
            transition_slope = net_amplitude_change / max(transition_length, eps)
            positive_derivative_area = np.sum(np.maximum(derivative_segment, 0.0))
            negative_derivative_area = np.sum(np.maximum(-derivative_segment, 0.0))
            derivative_std, zero_crossing_count, zero_crossing_rate = _derivative_variability_stats(
                derivative_segment,
                eps,
            )
            (
                max_abs_pos,
                max_abs_rel_pos,
                max_positive_pos,
                max_positive_rel_pos,
                min_negative_pos,
                min_negative_rel_pos,
            ) = _derivative_extrema_location_summary(first_derivative, start_freq, end_freq)

            features[f"{prefix}_transition_exists"] = np.float32(1.0)
            features[f"{prefix}_transition_start_freq"] = np.float32(start_freq)
            features[f"{prefix}_transition_end_freq"] = np.float32(end_freq)
            features[f"{prefix}_transition_length"] = np.float32(transition_length)
            features[f"{prefix}_transition_net_amplitude_change"] = np.float32(net_amplitude_change)
            features[f"{prefix}_transition_slope"] = np.float32(transition_slope)
            features[f"{prefix}_transition_mean_first_derivative"] = np.float32(np.mean(derivative_segment))
            features[f"{prefix}_transition_mean_abs_first_derivative"] = np.float32(np.mean(np.abs(derivative_segment)))
            features[f"{prefix}_transition_std_first_derivative"] = np.float32(derivative_std)
            features[f"{prefix}_transition_zero_crossing_count"] = np.float32(zero_crossing_count)
            features[f"{prefix}_transition_zero_crossing_rate"] = np.float32(zero_crossing_rate)
            features[f"{prefix}_transition_max_abs_first_derivative"] = np.float32(np.max(np.abs(derivative_segment)))
            features[f"{prefix}_transition_max_abs_first_derivative_position"] = np.float32(max_abs_pos)
            features[f"{prefix}_transition_max_abs_first_derivative_relative_position"] = np.float32(max_abs_rel_pos)
            features[f"{prefix}_transition_first_derivative_energy"] = np.float32(np.sum(derivative_segment * derivative_segment))
            features[f"{prefix}_transition_max_positive_first_derivative"] = np.float32(np.max(derivative_segment))
            features[f"{prefix}_transition_max_positive_first_derivative_position"] = np.float32(max_positive_pos)
            features[f"{prefix}_transition_max_positive_first_derivative_relative_position"] = np.float32(max_positive_rel_pos)
            features[f"{prefix}_transition_min_negative_first_derivative"] = np.float32(np.min(derivative_segment))
            features[f"{prefix}_transition_min_negative_first_derivative_position"] = np.float32(min_negative_pos)
            features[f"{prefix}_transition_min_negative_first_derivative_relative_position"] = np.float32(min_negative_rel_pos)
            features[f"{prefix}_transition_positive_derivative_area"] = np.float32(positive_derivative_area)
            features[f"{prefix}_transition_negative_derivative_area"] = np.float32(negative_derivative_area)

        if dip_freq.size == 0: # if no dips
            continue

        low_freq = min(start_freq, end_freq)
        high_freq = max(start_freq, end_freq)
        inter_band_dip_ids = np.where((dip_freq > low_freq) & (dip_freq < high_freq))[0]
        if inter_band_dip_ids.size == 0: # if no inter band dips
            continue

        inter_band_dip_ids = inter_band_dip_ids[np.argsort(dip_freq[inter_band_dip_ids])]
        first_dip_id = inter_band_dip_ids[0]
        last_dip_id = inter_band_dip_ids[-1]

        first_dip_freq = float(dip_freq[first_dip_id])
        first_dip_amp = float(dip_amp[first_dip_id])
        last_dip_freq = float(dip_freq[last_dip_id])
        last_dip_amp = float(dip_amp[last_dip_id])

        features[f"{prefix}_num_inter_band_dips"] = np.float32(inter_band_dip_ids.size)
        features[f"{prefix}_first_inter_band_dip_freq"] = np.float32(first_dip_freq)
        features[f"{prefix}_last_inter_band_dip_freq"] = np.float32(last_dip_freq)

        _add_segment_derivative_features(
            features=features,
            prefix=prefix,
            segment_name="peak_to_first_dip",
            start_freq=start_freq,
            start_amp=start_amp,
            end_freq=first_dip_freq,
            end_amp=first_dip_amp,
            first_derivative=first_derivative,
            eps=eps,
        )

        if first_dip_id != last_dip_id:
            _add_segment_derivative_features(
                features=features,
                prefix=prefix,
                segment_name="first_dip_to_last_dip",
                start_freq=first_dip_freq,
                start_amp=first_dip_amp,
                end_freq=last_dip_freq,
                end_amp=last_dip_amp,
                first_derivative=first_derivative,
                eps=eps,
            )

        _add_segment_derivative_features(
            features=features,
            prefix=prefix,
            segment_name="last_dip_to_peak",
            start_freq=last_dip_freq,
            start_amp=last_dip_amp,
            end_freq=end_freq,
            end_amp=end_amp,
            first_derivative=first_derivative,
            eps=eps,
        )

    return features


def calculate_first_derivative_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray] = None,
    detected_peak_dip: Optional[dict] = None,
    window_radius: int = 5,
    include_inter_band: bool = True,
    include_broad_transition: bool = True,
    eps: float = 1e-8,
) -> dict:
    """Calculate first-derivative features within bands and, optionally, between bands."""
    features = calculate_within_band_derivative_features(
        selected_pairs=selected_pairs,
        signal=signal,
        first_derivative=first_derivative,
        window_radius=window_radius,
        eps=eps,
    )
    if include_inter_band:
        features.update(
            calculate_inter_band_derivative_features(
                selected_pairs=selected_pairs,
                signal=signal,
                first_derivative=first_derivative,
                detected_peak_dip=detected_peak_dip,
                include_broad_transition=include_broad_transition,
                eps=eps,
            )
        )
    return features


def calculate_derivative_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray] = None,
    detected_peak_dip: Optional[dict] = None,
    window_radius: int = 5,
    include_inter_band: bool = True,
    include_broad_transition: bool = True,
    eps: float = 1e-8,
) -> dict:
    """Compatibility wrapper for first-derivative-only feature extraction."""
    return calculate_first_derivative_features(
        selected_pairs=selected_pairs,
        signal=signal,
        first_derivative=first_derivative,
        detected_peak_dip=detected_peak_dip,
        window_radius=window_radius,
        include_inter_band=include_inter_band,
        include_broad_transition=include_broad_transition,
        eps=eps,
    )
