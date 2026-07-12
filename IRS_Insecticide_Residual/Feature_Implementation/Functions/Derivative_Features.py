from typing import Optional

import numpy as np


"""
Derivative feature design notes
-------------------------------
The feature groups below are separated by the physical shape question they answer.

_MAIN_PEAK_SLOPE_KEYS:
    Captures whether the main peak rises and falls symmetrically around its width reference line.
    These features describe peak sharpness and left/right asymmetry using simple endpoint slopes.

_DIP_TO_PEAK_PAIR_KEYS:
    Compares the two sides of a selected doublet: left peak to middle dip and middle dip to right peak.
    These features describe whether the doublet has balanced or asymmetric side slopes.

_FIRST_DERIVATIVE_DOUBLET_SHAPE_KEYS:
    Describes first-derivative behavior inside each selected band/doublet region.
    These features focus on steepness, derivative variability, zero crossings, derivative energy,
    and where the strongest positive/negative derivative occurs.

_FIRST_DERIVATIVE_SEGMENT_KEYS:
    Describes endpoint-defined first-derivative transition segments, currently used between frequency bands.
    These include start/end metadata, net amplitude change, slope, derivative statistics, and
    positive/negative derivative area because transitions need explicit segment geometry.

_SECOND_DERIVATIVE_SEGMENT_KEYS:
    Describes compact curvature behavior for both within-band and inter-band regions.
    These omit start/end slope metadata because second derivative is used here mainly for curvature
    strength and sign, not for endpoint trend.

Why first derivative has two key lists but second derivative has one:
    First derivative separates within-doublet shape from endpoint-to-endpoint transitions because
    transition features need geometry fields such as start/end, length, amplitude change, and slope.
    Second derivative is kept as one compact curvature set because it is noisier and mainly used for
    curvature strength/sign; it omits first-derivative-style trend, zero-crossing, std, and position fields.

Main function groups:
    calculate_within_band_first_derivative_features() extracts within-band (doublet-shape) first-derivative features.
    calculate_inter_band_first_derivative_features() extracts inter-band first-derivative transition features.
    calculate_within_band_second_derivative_features() extracts within-band curvature features.
    calculate_inter_band_second_derivative_features() extracts inter-band curvature features.
"""


# Slope-only feature names for the main peak width reference line.
_MAIN_PEAK_SLOPE_KEYS = [
    "left_slope",  # slope from left width intersection to the main peak
    "right_slope",  # slope from the main peak to right width intersection
    "slope_balance",  # ratio of smaller absolute side slope to larger absolute side slope
    "signed_slope_imbalance",  # signed left-vs-right slope difference normalized by total slope magnitude
]


# Slope-only feature names comparing the left and right peak-to-dip sides of a doublet.
_DIP_TO_PEAK_PAIR_KEYS = [
    "left_dip_to_peak_slope",  # amplitude drop/rise rate between left peak and middle dip
    "right_dip_to_peak_slope",  # amplitude rise/drop rate between middle dip and right peak
    "dip_to_peak_slope_balance",  # ratio of smaller absolute dip-to-peak slope to larger absolute slope
    "dip_to_peak_signed_slope_imbalance",  # signed left-vs-right dip-to-peak slope difference
]


# Shared feature names for doublet shape first-derivative features around peaks/dip-to-peak segments.
_FIRST_DERIVATIVE_DOUBLET_SHAPE_KEYS = [
    "max_abs_first_derivative",  # strongest doublet shape first-derivative magnitude
    "mean_abs_first_derivative",  # average absolute first derivative in the segment
    "std_first_derivative",  # variability of first-derivative values in the segment
    "zero_crossing_count",  # number of sign changes in non-near-zero derivative values
    "zero_crossing_rate",  # zero-crossing count normalized by segment length
    "max_abs_first_derivative_position",  # absolute index where first-derivative magnitude is largest
    "max_abs_first_derivative_relative_position",  # relative segment position of largest derivative magnitude
    "first_derivative_energy",  # sum of squared first-derivative values in the segment
    "max_positive_first_derivative",  # largest positive first-derivative value
    "max_positive_first_derivative_position",  # absolute index of largest positive first derivative
    "max_positive_first_derivative_relative_position",  # relative segment position of largest positive derivative
    "min_negative_first_derivative",  # most negative first-derivative value
    "min_negative_first_derivative_position",  # absolute index of most negative first derivative
    "min_negative_first_derivative_relative_position",  # relative segment position of most negative derivative
]


# Shared feature names for first-derivative transition segments between two frequency positions.
_FIRST_DERIVATIVE_SEGMENT_KEYS = [
    "exists",  # flag indicating the segment was found and measured
    "start_freq",  # start index/frequency of the segment
    "end_freq",  # end index/frequency of the segment
    "length",  # absolute distance between start and end positions
    "net_amplitude_change",  # end amplitude minus start amplitude
    "slope",  # net amplitude change divided by segment length
    "mean_first_derivative",  # signed mean first derivative over the segment
    "mean_abs_first_derivative",  # mean absolute first derivative over the segment
    "std_first_derivative",  # standard deviation of first-derivative values
    "zero_crossing_count",  # number of sign changes in non-near-zero derivative values
    "zero_crossing_rate",  # zero-crossing count normalized by segment length
    "max_abs_first_derivative",  # strongest first-derivative magnitude in the segment
    "max_abs_first_derivative_position",  # absolute index of strongest derivative magnitude
    "max_abs_first_derivative_relative_position",  # relative segment position of strongest derivative magnitude
    "first_derivative_energy",  # sum of squared first-derivative values
    "max_positive_first_derivative",  # largest positive first-derivative value
    "max_positive_first_derivative_position",  # absolute index of largest positive derivative
    "max_positive_first_derivative_relative_position",  # relative segment position of largest positive derivative
    "min_negative_first_derivative",  # most negative first-derivative value
    "min_negative_first_derivative_position",  # absolute index of most negative derivative
    "min_negative_first_derivative_relative_position",  # relative segment position of most negative derivative
    "positive_derivative_area",  # sum of positive first-derivative values
    "negative_derivative_area",  # sum of absolute negative first-derivative values
]


# Compact curvature feature names for second-derivative segments.
_SECOND_DERIVATIVE_SEGMENT_KEYS = [
    "second_derivative_exists",  # flag indicating the curvature segment was found and measured
    "mean_second_derivative",  # signed mean curvature over the segment
    "mean_abs_second_derivative",  # average curvature magnitude over the segment
    "max_abs_second_derivative",  # strongest curvature magnitude in the segment
    "second_derivative_energy",  # sum of squared second-derivative values
    "max_positive_second_derivative",  # strongest positive curvature value
    "min_negative_second_derivative",  # strongest negative curvature value
    "positive_second_derivative_area",  # sum of positive second-derivative values
    "negative_second_derivative_area",  # sum of absolute negative second-derivative values
]


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


def _prepare_second_derivative(
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray],
    second_derivative: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal, first_derivative = _prepare_signal_and_derivative(signal, first_derivative)
    if second_derivative is None:
        second_derivative = np.gradient(first_derivative).astype(np.float32)
    else:
        second_derivative = np.asarray(second_derivative, dtype=np.float32).ravel()

    if signal.shape != second_derivative.shape:
        raise ValueError(
            "signal and second_derivative must have the same shape, "
            f"got signal={signal.shape}, second_derivative={second_derivative.shape}"
        )

    return signal, first_derivative, second_derivative


def _inter_band_endpoint(pair: dict, side: str) -> Optional[tuple[float, float]]:
    if side == "right" and pair.get("pair_exists", False):
        return float(pair["right_peak_freq"]), float(pair["right_peak_amp"])
    if side == "left" and pair.get("pair_exists", False):
        return float(pair["left_peak_freq"]), float(pair["left_peak_amp"])
    if pair.get("main_peak_exists", False):
        return float(pair["main_peak_freq"]), float(pair["main_peak_amp"])
    return None


def _abs_derivative_stats(
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


def _signed_derivative_stats(
    first_derivative: np.ndarray,
    start_idx: float,
    end_idx: float,
) -> tuple[float, float]:
    start, end = _segment_bounds(start_idx, end_idx, len(first_derivative))
    segment = first_derivative[start:end + 1]
    return float(np.max(segment)), float(np.min(segment))


def _derivative_variability_stats(
    first_derivative: np.ndarray,
    start_idx: float,
    end_idx: float,
    eps: float,
) -> tuple[float, float, float]:
    start, end = _segment_bounds(start_idx, end_idx, len(first_derivative))
    segment = first_derivative[start:end + 1]
    segment = np.asarray(segment, dtype=np.float32).ravel()
    if segment.size == 0:
        return 0.0, 0.0, 0.0

    derivative_std = float(np.std(segment))
    nonzero_values = segment[np.abs(segment) > eps]
    if nonzero_values.size < 2:
        zero_crossing_count = 0
    else:
        signs = np.sign(nonzero_values)
        zero_crossing_count = int(np.count_nonzero(signs[:-1] * signs[1:] < 0))
    zero_crossing_rate = zero_crossing_count / max(segment.size - 1, 1)
    return derivative_std, float(zero_crossing_count), float(zero_crossing_rate)


def _derivative_extrema_location(
    first_derivative: np.ndarray,
    start_idx: float,
    end_idx: float,
) -> tuple[float, float, float, float, float, float]:
    start, end = _segment_bounds(start_idx, end_idx, len(first_derivative))
    segment = first_derivative[start:end + 1]
    if segment.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    max_abs_pos = start + int(np.argmax(np.abs(segment)))
    max_positive_pos = start + int(np.argmax(segment))
    min_negative_pos = start + int(np.argmin(segment))
    denom = max(end - start, 1)

    return (
        float(max_abs_pos),
        float((max_abs_pos - start) / denom),
        float(max_positive_pos),
        float((max_positive_pos - start) / denom),
        float(min_negative_pos),
        float((min_negative_pos - start) / denom),
    )


def _add_first_derivative_doublet_shape_features(
    features: dict,
    prefix: str,
    segment_name: str,
    first_derivative: np.ndarray,
    start_idx: float,
    end_idx: float,
    eps: float,
) -> None:
    max_abs_d, mean_abs_d, energy_d = _abs_derivative_stats(
        first_derivative,
        start_idx,
        end_idx,
    )
    max_positive_d, min_negative_d = _signed_derivative_stats(
        first_derivative,
        start_idx,
        end_idx,
    )
    std_d, zc_count_d, zc_rate_d = _derivative_variability_stats(
        first_derivative,
        start_idx,
        end_idx,
        eps,
    )
    (
        max_abs_pos_d,
        max_abs_rel_pos_d,
        max_positive_pos_d,
        max_positive_rel_pos_d,
        min_negative_pos_d,
        min_negative_rel_pos_d,
    ) = _derivative_extrema_location(first_derivative, start_idx, end_idx)

    features[f"{prefix}_{segment_name}_max_abs_first_derivative"] = np.float32(max_abs_d)
    features[f"{prefix}_{segment_name}_mean_abs_first_derivative"] = np.float32(mean_abs_d)
    features[f"{prefix}_{segment_name}_std_first_derivative"] = np.float32(std_d)
    features[f"{prefix}_{segment_name}_zero_crossing_count"] = np.float32(zc_count_d)
    features[f"{prefix}_{segment_name}_zero_crossing_rate"] = np.float32(zc_rate_d)
    features[f"{prefix}_{segment_name}_max_abs_first_derivative_position"] = np.float32(max_abs_pos_d)
    features[f"{prefix}_{segment_name}_max_abs_first_derivative_relative_position"] = np.float32(max_abs_rel_pos_d)
    features[f"{prefix}_{segment_name}_first_derivative_energy"] = np.float32(energy_d)
    features[f"{prefix}_{segment_name}_max_positive_first_derivative"] = np.float32(max_positive_d)
    features[f"{prefix}_{segment_name}_max_positive_first_derivative_position"] = np.float32(max_positive_pos_d)
    features[f"{prefix}_{segment_name}_max_positive_first_derivative_relative_position"] = np.float32(max_positive_rel_pos_d)
    features[f"{prefix}_{segment_name}_min_negative_first_derivative"] = np.float32(min_negative_d)
    features[f"{prefix}_{segment_name}_min_negative_first_derivative_position"] = np.float32(min_negative_pos_d)
    features[f"{prefix}_{segment_name}_min_negative_first_derivative_relative_position"] = np.float32(min_negative_rel_pos_d)


def _add_empty_first_derivative_segment_features(features: dict, prefix: str, segment_name: str) -> None:
    for key in _FIRST_DERIVATIVE_SEGMENT_KEYS:
        features[f"{prefix}_{segment_name}_{key}"] = np.float32(0.0)


def _add_first_derivative_segment_features(
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
        first_derivative,
        start_freq,
        end_freq,
        eps,
    )
    (
        max_abs_pos,
        max_abs_rel_pos,
        max_positive_pos,
        max_positive_rel_pos,
        min_negative_pos,
        min_negative_rel_pos,
    ) = _derivative_extrema_location(first_derivative, start_freq, end_freq)

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


def _add_empty_second_derivative_segment_features(
    features: dict,
    prefix: str,
    segment_name: str,
) -> None:
    for key in _SECOND_DERIVATIVE_SEGMENT_KEYS:
        features[f"{prefix}_{segment_name}_{key}"] = np.float32(0.0)


def _add_second_derivative_segment_features(
    features: dict,
    prefix: str,
    segment_name: str,
    start_idx: float,
    end_idx: float,
    second_derivative: np.ndarray,
    eps: float,
) -> None:
    start, end = _segment_bounds(start_idx, end_idx, len(second_derivative))
    segment = second_derivative[start:end + 1]
    if segment.size == 0:
        return

    features[f"{prefix}_{segment_name}_second_derivative_exists"] = np.float32(1.0)
    features[f"{prefix}_{segment_name}_mean_second_derivative"] = np.float32(np.mean(segment))
    features[f"{prefix}_{segment_name}_mean_abs_second_derivative"] = np.float32(np.mean(np.abs(segment)))
    features[f"{prefix}_{segment_name}_max_abs_second_derivative"] = np.float32(np.max(np.abs(segment)))
    features[f"{prefix}_{segment_name}_second_derivative_energy"] = np.float32(np.sum(segment * segment))
    features[f"{prefix}_{segment_name}_max_positive_second_derivative"] = np.float32(np.max(segment))
    features[f"{prefix}_{segment_name}_min_negative_second_derivative"] = np.float32(np.min(segment))
    features[f"{prefix}_{segment_name}_positive_second_derivative_area"] = np.float32(np.sum(np.maximum(segment, 0.0)))
    features[f"{prefix}_{segment_name}_negative_second_derivative_area"] = np.float32(np.sum(np.maximum(-segment, 0.0)))


def calculate_within_band_first_derivative_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray] = None,
    window_radius: int = 5, # 5 corresponds to 55 original points
    eps: float = 1e-8,
) -> dict:
    """
    Calculate first-derivative shape features around selected band-wise peak/dip pairs.

    This covers doublet shape around the main peak and the two within-doublet dip-to-peak sides.

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
        Number of points on each side of the peak used for doublet shape derivative features.
    eps:
        Small value used to avoid division by zero.

    Returns
    -------
    features:
        Fixed-length dictionary of first-derivative features per band.
    """
    signal, first_derivative = _prepare_signal_and_derivative(signal, first_derivative)

    features = {}

    slope_keys = (
        [f"main_peak_{key}" for key in _MAIN_PEAK_SLOPE_KEYS]
        + _DIP_TO_PEAK_PAIR_KEYS
    )

    for pair in selected_pairs:
        band_id = pair["band_id"]
        prefix = f"band{band_id}"

        for key in slope_keys:
            features[f"{prefix}_{key}"] = np.float32(0.0)
        for segment_name in ["main_peak", "left_dip_to_peak", "right_dip_to_peak"]:
            for key in _FIRST_DERIVATIVE_DOUBLET_SHAPE_KEYS:
                features[f"{prefix}_{segment_name}_{key}"] = np.float32(0.0)

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

            features[f"{prefix}_main_peak_left_slope"] = np.float32(left_slope)
            features[f"{prefix}_main_peak_right_slope"] = np.float32(right_slope)
            features[f"{prefix}_main_peak_slope_balance"] = np.float32(slope_balance)
            features[f"{prefix}_main_peak_signed_slope_imbalance"] = np.float32(signed_slope_imbalance)
            _add_first_derivative_doublet_shape_features(
                features=features,
                prefix=prefix,
                segment_name="main_peak",
                first_derivative=first_derivative,
                start_idx=win_start,
                end_idx=win_end,
                eps=eps,
            )

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

        features[f"{prefix}_left_dip_to_peak_slope"] = np.float32(left_dip_to_peak_slope)
        features[f"{prefix}_right_dip_to_peak_slope"] = np.float32(right_dip_to_peak_slope)
        features[f"{prefix}_dip_to_peak_slope_balance"] = np.float32(dip_to_peak_slope_balance)
        features[f"{prefix}_dip_to_peak_signed_slope_imbalance"] = np.float32(dip_to_peak_signed_slope_imbalance)
        _add_first_derivative_doublet_shape_features(
            features=features,
            prefix=prefix,
            segment_name="left_dip_to_peak",
            first_derivative=first_derivative,
            start_idx=left_peak_freq,
            end_idx=dip_freq,
            eps=eps,
        )
        _add_first_derivative_doublet_shape_features(
            features=features,
            prefix=prefix,
            segment_name="right_dip_to_peak",
            first_derivative=first_derivative,
            start_idx=dip_freq,
            end_idx=right_peak_freq,
            eps=eps,
        )

    return features


def calculate_inter_band_first_derivative_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray] = None,
    detected_peak_dip: Optional[dict] = None,
    include_broad_transition: bool = True,
    eps: float = 1e-8,
) -> dict:
    """
    Calculate first-derivative transition features between adjacent bands.

    This covers the shape from the end peak of one selected band to the start peak of the next band.

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

    metadata_keys = [
        "num_inter_band_dips",
        "first_inter_band_dip_freq",
        "last_inter_band_dip_freq",
    ]
    segment_names = [
        "transition",
        "peak_to_first_dip",
        "first_dip_to_last_dip",
        "last_dip_to_peak",
    ]

    for left_pair, right_pair in zip(sorted_pairs[:-1], sorted_pairs[1:]): # i.e., zip([band1, band2], [band2, band3])
        left_band_id = left_pair["band_id"]
        right_band_id = right_pair["band_id"]
        prefix = f"band{left_band_id}_to_band{right_band_id}"

        for key in metadata_keys:
            features[f"{prefix}_{key}"] = np.float32(0.0)
        for segment_name in segment_names:
            _add_empty_first_derivative_segment_features(features, prefix, segment_name)

        start_endpoint = _inter_band_endpoint(left_pair, side="right")
        end_endpoint = _inter_band_endpoint(right_pair, side="left")
        if start_endpoint is None or end_endpoint is None:
            continue

        start_freq, start_amp = start_endpoint
        end_freq, end_amp = end_endpoint
        if include_broad_transition:
            _add_first_derivative_segment_features(
                features=features,
                prefix=prefix,
                segment_name="transition",
                start_freq=start_freq,
                start_amp=start_amp,
                end_freq=end_freq,
                end_amp=end_amp,
                first_derivative=first_derivative,
                eps=eps,
            )

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

        _add_first_derivative_segment_features(
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
            _add_first_derivative_segment_features(
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

        _add_first_derivative_segment_features(
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


def calculate_within_band_second_derivative_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray] = None,
    second_derivative: Optional[np.ndarray] = None,
    window_radius: int = 5,
    eps: float = 1e-8,
) -> dict:
    """
    Calculate second-derivative curvature features inside each selected band.

    This covers local curvature around the main peak, the left-peak-to-middle-dip
    side, and the middle-dip-to-right-peak side. It also records the exact
    second-derivative value at the main peak, left peak, right peak, and middle dip.

    Parameters
    ----------
    selected_pairs:
        Output from Peak_Dip_Features.select_band_peak_dip_pairs().
    signal:
        Original 1D signal used for peak/dip detection.
    first_derivative:
        Optional first-derivative signal aligned with `signal`. If not provided,
        np.gradient(signal) is used before calculating the second derivative.
    second_derivative:
        Optional second-derivative signal aligned with `signal`. If not provided,
        np.gradient(first_derivative) is used.
    window_radius:
        Number of points on each side of the main peak used for local curvature features.
    eps:
        Small value kept for API consistency with other derivative feature functions.

    Returns
    -------
    features:
        Fixed-length dictionary of within-band second-derivative features per band.
    """
    signal, first_derivative, second_derivative = _prepare_second_derivative(
        signal,
        first_derivative,
        second_derivative,
    )
    features = {}

    for pair in selected_pairs:
        band_id = pair["band_id"]
        prefix = f"band{band_id}"

        for segment_name in ["main_peak", "left_dip_to_peak", "right_dip_to_peak"]:
            _add_empty_second_derivative_segment_features(features, prefix, segment_name)

        features[f"{prefix}_main_peak_second_derivative_at_peak"] = np.float32(0.0)  # curvature at main peak
        features[f"{prefix}_left_peak_second_derivative_at_peak"] = np.float32(0.0)  # curvature at left peak
        features[f"{prefix}_right_peak_second_derivative_at_peak"] = np.float32(0.0)  # curvature at right peak
        features[f"{prefix}_middle_dip_second_derivative_at_dip"] = np.float32(0.0)  # curvature at middle dip

        if pair.get("main_peak_exists", False):
            peak_freq = float(pair["main_peak_freq"])
            peak_idx = _safe_index(peak_freq, len(second_derivative))
            win_start, win_end = _window_bounds(peak_freq, window_radius, len(second_derivative))
            _add_second_derivative_segment_features(
                features=features,
                prefix=prefix,
                segment_name="main_peak",
                start_idx=win_start,
                end_idx=win_end,
                second_derivative=second_derivative,
                eps=eps,
            )
            features[f"{prefix}_main_peak_second_derivative_at_peak"] = np.float32(second_derivative[peak_idx])  # main peak curvature

        if not pair.get("pair_exists", False) or not pair.get("middle_dip_exists", False):
            continue

        left_peak_freq = float(pair["left_peak_freq"])
        right_peak_freq = float(pair["right_peak_freq"])
        dip_freq = float(pair["middle_dip_freq"])
        left_peak_idx = _safe_index(left_peak_freq, len(second_derivative))
        right_peak_idx = _safe_index(right_peak_freq, len(second_derivative))
        dip_idx = _safe_index(dip_freq, len(second_derivative))

        _add_second_derivative_segment_features(
            features=features,
            prefix=prefix,
            segment_name="left_dip_to_peak",
            start_idx=left_peak_freq,
            end_idx=dip_freq,
            second_derivative=second_derivative,
            eps=eps,
        )
        _add_second_derivative_segment_features(
            features=features,
            prefix=prefix,
            segment_name="right_dip_to_peak",
            start_idx=dip_freq,
            end_idx=right_peak_freq,
            second_derivative=second_derivative,
            eps=eps,
        )
        features[f"{prefix}_left_peak_second_derivative_at_peak"] = np.float32(second_derivative[left_peak_idx])  # left peak curvature
        features[f"{prefix}_right_peak_second_derivative_at_peak"] = np.float32(second_derivative[right_peak_idx])  # right peak curvature
        features[f"{prefix}_middle_dip_second_derivative_at_dip"] = np.float32(second_derivative[dip_idx])  # middle dip curvature

    return features


def calculate_inter_band_second_derivative_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray] = None,
    second_derivative: Optional[np.ndarray] = None,
    detected_peak_dip: Optional[dict] = None,
    include_broad_transition: bool = True,
    eps: float = 1e-8,
) -> dict:
    """
    Calculate second-derivative curvature features between adjacent selected bands.

    This covers the curvature from the end peak of one selected band to the start peak of the next band.

    For each adjacent pair of bands, the transition is measured from:
        previous band right peak if a pair exists, otherwise previous band main peak
    to:
        next band left peak if a pair exists, otherwise next band main peak.

    If detected_peak_dip is provided, the broad transition is also split into:
        peak_to_first_dip
        first_dip_to_last_dip
        last_dip_to_peak
    """
    signal, first_derivative, second_derivative = _prepare_second_derivative(
        signal,
        first_derivative,
        second_derivative,
    )
    features = {}
    sorted_pairs = sorted(selected_pairs, key=lambda pair: pair["band_id"])
    dip_freq = np.asarray([], dtype=np.float32)
    if detected_peak_dip is not None:
        dip_freq = np.asarray(detected_peak_dip.get("dip_frequencies", []), dtype=np.float32)

    for left_pair, right_pair in zip(sorted_pairs[:-1], sorted_pairs[1:]):
        left_band_id = left_pair["band_id"]
        right_band_id = right_pair["band_id"]
        prefix = f"band{left_band_id}_to_band{right_band_id}"

        for segment_name in [
            "transition",
            "peak_to_first_dip",
            "first_dip_to_last_dip",
            "last_dip_to_peak",
        ]:
            _add_empty_second_derivative_segment_features(features, prefix, segment_name)

        start_endpoint = _inter_band_endpoint(left_pair, side="right")
        end_endpoint = _inter_band_endpoint(right_pair, side="left")
        if start_endpoint is None or end_endpoint is None:
            continue

        start_freq, _ = start_endpoint
        end_freq, _ = end_endpoint

        if include_broad_transition:
            _add_second_derivative_segment_features(
                features=features,
                prefix=prefix,
                segment_name="transition",
                start_idx=start_freq,
                end_idx=end_freq,
                second_derivative=second_derivative,
                eps=eps,
            )

        if dip_freq.size == 0:
            continue

        low_freq = min(start_freq, end_freq)
        high_freq = max(start_freq, end_freq)
        inter_band_dip_ids = np.where((dip_freq > low_freq) & (dip_freq < high_freq))[0]
        if inter_band_dip_ids.size == 0:
            continue

        inter_band_dip_ids = inter_band_dip_ids[np.argsort(dip_freq[inter_band_dip_ids])]
        first_dip_id = inter_band_dip_ids[0]
        last_dip_id = inter_band_dip_ids[-1]

        first_dip_freq = float(dip_freq[first_dip_id])
        last_dip_freq = float(dip_freq[last_dip_id])

        _add_second_derivative_segment_features(
            features=features,
            prefix=prefix,
            segment_name="peak_to_first_dip",
            start_idx=start_freq,
            end_idx=first_dip_freq,
            second_derivative=second_derivative,
            eps=eps,
        )
        if first_dip_id != last_dip_id:
            _add_second_derivative_segment_features(
                features=features,
                prefix=prefix,
                segment_name="first_dip_to_last_dip",
                start_idx=first_dip_freq,
                end_idx=last_dip_freq,
                second_derivative=second_derivative,
                eps=eps,
            )
        _add_second_derivative_segment_features(
            features=features,
            prefix=prefix,
            segment_name="last_dip_to_peak",
            start_idx=last_dip_freq,
            end_idx=end_freq,
            second_derivative=second_derivative,
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
    """Public first-derivative wrapper combining within-band and optional inter-band features."""
    features = calculate_within_band_first_derivative_features(
        selected_pairs=selected_pairs,
        signal=signal,
        first_derivative=first_derivative,
        window_radius=window_radius,
        eps=eps,
    )
    if include_inter_band:
        features.update(
            calculate_inter_band_first_derivative_features(
                selected_pairs=selected_pairs,
                signal=signal,
                first_derivative=first_derivative,
                detected_peak_dip=detected_peak_dip,
                include_broad_transition=include_broad_transition,
                eps=eps,
            )
        )
    return features


def calculate_second_derivative_features(
    selected_pairs: list[dict],
    signal: np.ndarray,
    first_derivative: Optional[np.ndarray] = None,
    second_derivative: Optional[np.ndarray] = None,
    detected_peak_dip: Optional[dict] = None,
    window_radius: int = 5,
    include_inter_band: bool = True,
    include_broad_transition: bool = True,
    eps: float = 1e-8,
) -> dict:
    """
    Calculate second-derivative curvature features within bands and, optionally, between bands.

    This is the public second-derivative wrapper combining the two split second-derivative functions.

    The second derivative describes curvature:
        negative values around peak tops indicate concave-down sharpness,
        positive values around dip bottoms indicate concave-up sharpness.
    """
    features = calculate_within_band_second_derivative_features(
        selected_pairs=selected_pairs,
        signal=signal,
        first_derivative=first_derivative,
        second_derivative=second_derivative,
        window_radius=window_radius,
        eps=eps,
    )
    if include_inter_band:
        features.update(
            calculate_inter_band_second_derivative_features(
                selected_pairs=selected_pairs,
                signal=signal,
                first_derivative=first_derivative,
                second_derivative=second_derivative,
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
    second_derivative: Optional[np.ndarray] = None,
    window_radius: int = 5,
    include_inter_band: bool = True,
    include_broad_transition: bool = True,
    include_second_derivative: bool = False,
    eps: float = 1e-8,
) -> dict:
    """Public wrapper for first-derivative features plus optional second-derivative features."""
    features = calculate_first_derivative_features(
        selected_pairs=selected_pairs,
        signal=signal,
        first_derivative=first_derivative,
        detected_peak_dip=detected_peak_dip,
        window_radius=window_radius,
        include_inter_band=include_inter_band,
        include_broad_transition=include_broad_transition,
        eps=eps,
    )
    if include_second_derivative:
        features.update(
            calculate_second_derivative_features(
                selected_pairs=selected_pairs,
                signal=signal,
                first_derivative=first_derivative,
                second_derivative=second_derivative,
                detected_peak_dip=detected_peak_dip,
                window_radius=window_radius,
                include_inter_band=include_inter_band,
                include_broad_transition=include_broad_transition,
                eps=eps,
            )
        )
    return features

