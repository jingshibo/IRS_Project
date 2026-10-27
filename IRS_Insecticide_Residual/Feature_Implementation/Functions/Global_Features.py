from typing import List, Tuple

import numpy as np


"""
Global feature design notes
---------------------------
Global features summarize the whole channel signal rather than peak/dip-specific regions.

Basic amplitude statistics:
    Mean, std, min/max, peak-to-peak, median, quartiles, IQR, abs mean, and RMS describe
    overall signal level, spread, range, and robust distribution shape.

Global shape statistics:
    Zero-crossing rate and linear slope describe broad oscillation/sign-change behavior and
    overall trend across the full channel.

Distribution-shape statistics:
    Skewness and excess kurtosis describe asymmetry and tail/peak heaviness of the full signal.

Global location statistics:
    Argmax and argmin record where the strongest high/low values occur in the full channel.

Spectral-style statistics:
    Dominant bin, spectral centroid, bandwidth, and entropy describe where the signal energy is
    concentrated and how broadly it is spread across bins.

Relation to derivative features:
    These features are intentionally global. Derivative_Features.py measures local doublet and
    inter-band shape, so similar ideas such as slope or zero crossing are measured at different scales.

Fail-fast behavior:
    Near-zero denominators intentionally raise errors so degenerate samples, such as constant or
    all-zero signals, are visible during data checks instead of being silently converted to zeros.
"""


_EPS = np.float32(1e-12)  # small float32 constant to prevent division by zero


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide float32 arrays and raise a detailed error for near-zero denominators."""
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)

    zero_mask = np.abs(denominator) <= _EPS
    if np.any(zero_mask):
        zero_positions = np.argwhere(zero_mask)
        raise ValueError(
            "_safe_divide encountered near-zero denominator: "
            f"min_abs={np.min(np.abs(denominator))}, "
            f"count={np.count_nonzero(zero_mask)}, "
            f"positions={zero_positions.tolist()}"
        )
    return numerator / denominator


def calculate_global_features(channel_values: np.ndarray, channel_name: str) -> Tuple[np.ndarray, List[str]]:
    """Extract global statistical and spectral features from one channel block [N, L]."""
    channel_values = np.asarray(channel_values, dtype=np.float32)
    if channel_values.ndim != 2:
        raise ValueError(f"channel_values must have shape [N, L], got {channel_values.shape}")

    signal_length = channel_values.shape[1]
    if signal_length < 2:
        raise ValueError(f"Signal length must be >= 2, got {signal_length}")

    centered = channel_values - channel_values.mean(axis=1, keepdims=True)
    std = channel_values.std(axis=1)
    q25 = np.percentile(channel_values, 25.0, axis=1)
    q75 = np.percentile(channel_values, 75.0, axis=1)

    positions = np.arange(signal_length, dtype=np.float32)
    centered_positions = positions - positions.mean()
    slope_denominator = np.sum(centered_positions ** 2, dtype=np.float32)
    slope = (centered @ centered_positions) / np.maximum(slope_denominator, _EPS)

    m3 = np.mean(centered ** 3, axis=1)
    m4 = np.mean(centered ** 4, axis=1)
    skewness = _safe_divide(m3, std ** 3)
    kurtosis_excess = _safe_divide(m4, std ** 4) - 3.0

    sign_changes = np.signbit(channel_values[:, 1:]) != np.signbit(channel_values[:, :-1])
    zero_crossing_rate = sign_changes.mean(axis=1)

    freq_amplitude = np.abs(channel_values).astype(np.float32, copy=False)
    n_freq_bins = freq_amplitude.shape[1]
    freq_idx = np.arange(n_freq_bins, dtype=np.float32)
    magnitude_sum = freq_amplitude.sum(axis=1)

    spectral_prob = _safe_divide(freq_amplitude, magnitude_sum[:, None])
    spectral_entropy = -np.sum(spectral_prob * np.log(spectral_prob + _EPS), axis=1)
    dominant_freq_bin = np.argmax(freq_amplitude[:, 1:], axis=1) + 1
    spectral_centroid = _safe_divide(np.sum(freq_amplitude * freq_idx[None, :], axis=1), magnitude_sum)
    spectral_bandwidth_var = _safe_divide(
        np.sum(freq_amplitude * (freq_idx[None, :] - spectral_centroid[:, None]) ** 2, axis=1),
        magnitude_sum,
    )
    spectral_bandwidth_std = np.sqrt(np.maximum(spectral_bandwidth_var, 0.0))
    argmax_frac = np.argmax(channel_values, axis=1) / np.float32(signal_length - 1)
    argmin_frac = np.argmin(channel_values, axis=1) / np.float32(signal_length - 1)

    feature_arrays = [
        channel_values.mean(axis=1),
        std,
        channel_values.min(axis=1),
        channel_values.max(axis=1),
        np.ptp(channel_values, axis=1),
        np.median(channel_values, axis=1),
        q25,
        q75,
        q75 - q25,
        np.mean(np.abs(channel_values), axis=1),
        np.sqrt(np.mean(channel_values ** 2, axis=1)),
        zero_crossing_rate,
        slope,
        skewness,
        kurtosis_excess,
        argmax_frac,
        argmin_frac,
        dominant_freq_bin,
        spectral_centroid,
        spectral_bandwidth_std,
        spectral_entropy,
    ]
    feature_names = [
        f"{channel_name}__mean",
        f"{channel_name}__std",
        f"{channel_name}__min",
        f"{channel_name}__max",
        f"{channel_name}__peak_to_peak",
        f"{channel_name}__median",
        f"{channel_name}__q25",
        f"{channel_name}__q75",
        f"{channel_name}__iqr",
        f"{channel_name}__abs_mean",
        f"{channel_name}__rms",
        f"{channel_name}__zero_crossing_rate",
        f"{channel_name}__slope",
        f"{channel_name}__skewness",
        f"{channel_name}__kurtosis_excess",
        f"{channel_name}__argmax_frac",
        f"{channel_name}__argmin_frac",
        f"{channel_name}__dominant_freq_bin",
        f"{channel_name}__spectral_centroid",
        f"{channel_name}__spectral_bandwidth",
        f"{channel_name}__spectral_entropy",
    ]

    feature_block = np.column_stack(feature_arrays).astype(np.float32, copy=False)
    return feature_block, feature_names
