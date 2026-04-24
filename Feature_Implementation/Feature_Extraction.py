from typing import List, Optional, Sequence, Tuple

import numpy as np


_EPS = 1e-12


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=np.abs(denominator) > _EPS,
    )


def _validate_signal_dataset(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"x must have shape [N, C, L], got {x.shape}")
    if x.shape[2] < 2:
        raise ValueError(f"Signal length must be >= 2, got {x.shape[2]}")
    return x


def _channel_feature_block(channel_values: np.ndarray, channel_name: str) -> Tuple[np.ndarray, List[str]]:
    channel_values = np.asarray(channel_values, dtype=np.float64)
    if channel_values.ndim != 2:
        raise ValueError(f"channel_values must have shape [N, L], got {channel_values.shape}")

    n_samples, signal_length = channel_values.shape
    centered = channel_values - channel_values.mean(axis=1, keepdims=True)
    std = channel_values.std(axis=1)
    diff = np.diff(channel_values, axis=1)
    q25 = np.percentile(channel_values, 25.0, axis=1)
    q75 = np.percentile(channel_values, 75.0, axis=1)

    positions = np.arange(signal_length, dtype=np.float64)
    centered_positions = positions - positions.mean()
    slope_denominator = np.sum(centered_positions ** 2)
    slope = centered @ centered_positions / max(slope_denominator, _EPS)

    m3 = np.mean(centered ** 3, axis=1)
    m4 = np.mean(centered ** 4, axis=1)
    skewness = _safe_divide(m3, std ** 3)
    kurtosis_excess = _safe_divide(m4, std ** 4) - 3.0

    sign_changes = np.signbit(channel_values[:, 1:]) != np.signbit(channel_values[:, :-1])
    zero_crossing_rate = sign_changes.mean(axis=1)

    fft_magnitude = np.abs(np.fft.rfft(channel_values, axis=1))
    n_freq_bins = fft_magnitude.shape[1]
    freq_idx = np.arange(n_freq_bins, dtype=np.float64)
    magnitude_sum = fft_magnitude.sum(axis=1)
    spectral_prob = _safe_divide(fft_magnitude, magnitude_sum[:, None])
    spectral_entropy = -np.sum(spectral_prob * np.log(spectral_prob + _EPS), axis=1)
    spectral_entropy = _safe_divide(spectral_entropy, np.log(max(n_freq_bins, 2)))

    if n_freq_bins > 1:
        dominant_freq_bin = np.argmax(fft_magnitude[:, 1:], axis=1) + 1
        dominant_freq_bin_frac = dominant_freq_bin / (n_freq_bins - 1)
    else:
        dominant_freq_bin_frac = np.zeros(n_samples, dtype=np.float64)

    spectral_centroid = _safe_divide(np.sum(fft_magnitude * freq_idx[None, :], axis=1), magnitude_sum)
    spectral_centroid_frac = _safe_divide(spectral_centroid, np.full_like(spectral_centroid, max(n_freq_bins - 1, 1)))

    spectral_bandwidth = _safe_divide(
        np.sum(fft_magnitude * (freq_idx[None, :] - spectral_centroid[:, None]) ** 2, axis=1),
        magnitude_sum,
    )
    spectral_bandwidth_frac = np.sqrt(np.maximum(spectral_bandwidth, 0.0)) / max(n_freq_bins - 1, 1)

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
        np.mean(np.abs(diff), axis=1),
        zero_crossing_rate,
        slope,
        skewness,
        kurtosis_excess,
        np.argmax(channel_values, axis=1) / max(signal_length - 1, 1),
        np.argmin(channel_values, axis=1) / max(signal_length - 1, 1),
        dominant_freq_bin_frac,
        spectral_centroid_frac,
        spectral_bandwidth_frac,
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
        f"{channel_name}__line_length",
        f"{channel_name}__zero_crossing_rate",
        f"{channel_name}__slope",
        f"{channel_name}__skewness",
        f"{channel_name}__kurtosis_excess",
        f"{channel_name}__argmax_frac",
        f"{channel_name}__argmin_frac",
        f"{channel_name}__dominant_freq_bin_frac",
        f"{channel_name}__spectral_centroid_frac",
        f"{channel_name}__spectral_bandwidth_frac",
        f"{channel_name}__spectral_entropy",
    ]

    feature_block = np.column_stack(feature_arrays).astype(np.float32, copy=False)
    return feature_block, feature_names


def extract_feature_matrix(
    x: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract handcrafted tabular features from a signal tensor."""
    x = _validate_signal_dataset(x)
    n_channels = x.shape[1]

    if channel_names is None:
        resolved_channel_names = [f"channel_{idx}" for idx in range(n_channels)]
    else:
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names length must match channel count {n_channels}, got {len(channel_names)}"
            )
        resolved_channel_names = [str(name) for name in channel_names]

    feature_blocks = []
    feature_names: List[str] = []
    for channel_idx, channel_name in enumerate(resolved_channel_names):
        channel_block, channel_feature_names = _channel_feature_block(x[:, channel_idx, :], channel_name=channel_name)
        feature_blocks.append(channel_block)
        feature_names.extend(channel_feature_names)

    feature_matrix = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    return feature_matrix, feature_names
