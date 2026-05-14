from typing import List, Optional, Sequence, Tuple
import numpy as np


_EPS = np.float32(1e-12) # small constant (float32) to prevent division by zero in skewness, kurtosis, and spectral features

##  safe division that returns 0 when denominator is close to zero
def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    # preserve float32 precision (avoid upcasting to float64)
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)

    # _safe_divide receives either [N] vs [N], or [N, L] vs [N, 1] as the input.
    zero_mask = np.abs(denominator) <= _EPS
    if np.any(zero_mask):
        raise ValueError(
            "_safe_divide encountered near-zero denominator: "
            f"min_abs={np.min(np.abs(denominator))}, "
            f"count={np.count_nonzero(zero_mask)}"
        )
    return numerator / denominator

##  validate that x has shape [N, C, L] and signal length >= 2
def _validate_signal_dataset(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"x must have shape [N, C, L], got {x.shape}")
    if x.shape[2] < 2:
        raise ValueError(f"Signal length must be >= 2, got {x.shape[2]}")
    return x

##  extract features from a single channel of shape [N, L], returning feature block of shape [N, F] and list of feature names
def _channel_feature_block(channel_values: np.ndarray, channel_name: str) -> Tuple[np.ndarray, List[str]]:
    # keep input in float32 (do not upcast to float64)
    channel_values = np.asarray(channel_values, dtype=np.float32)
    if channel_values.ndim != 2:
        raise ValueError(f"channel_values must have shape [N, L], got {channel_values.shape}")

    #// basic statistics
    n_samples, signal_length = channel_values.shape
    centered = channel_values - channel_values.mean(axis=1, keepdims=True)
    std = channel_values.std(axis=1)
    q25 = np.percentile(channel_values, 25.0, axis=1)
    q75 = np.percentile(channel_values, 75.0, axis=1)

    #// slope of linear fit to the signal values across time points
    positions = np.arange(signal_length, dtype=np.float32)
    centered_positions = positions - positions.mean()
    # compute denominator in float32 and use numpy maximum to avoid mixing python scalars
    slope_denominator = np.sum(centered_positions ** 2, dtype=np.float32)
    slope = (centered @ centered_positions) / np.maximum(slope_denominator, _EPS)

    #// skewness and excess kurtosis
    m3 = np.mean(centered ** 3, axis=1)
    m4 = np.mean(centered ** 4, axis=1)
    skewness = _safe_divide(m3, std ** 3)
    kurtosis_excess = _safe_divide(m4, std ** 4) - 3.0

    #// zero crossing rate
    sign_changes = np.signbit(channel_values[:, 1:]) != np.signbit(channel_values[:, :-1])
    zero_crossing_rate = sign_changes.mean(axis=1)

    # If channel_values contains complex FFT bins, take magnitudes. If it already contains magnitudes, abs() is a no-op.
    # treat input as frequency-domain amplitudes (keep float32)
    freq_amplitude = np.abs(channel_values).astype(np.float32, copy=False)  # shape (N, n_freq_bins)
    n_freq_bins = freq_amplitude.shape[1]
    freq_idx = np.arange(n_freq_bins, dtype=np.float32)
    # total magnitude per sample (used to normalize to a probability distribution)
    magnitude_sum = freq_amplitude.sum(axis=1)

    # spectral entropy: first convert to a probability distribution over frequency bins (safe division avoids NaNs)
    spectral_prob = _safe_divide(freq_amplitude, magnitude_sum[:, None]) # [:, None] adds a new axis to magnitude_sum
    # spectral entropy in nats (natural log)
    spectral_entropy = -np.sum(spectral_prob * np.log(spectral_prob + _EPS), axis=1)

    # dominant frequency bin index (skip DC at index 0)
    dominant_freq_bin = np.argmax(freq_amplitude[:, 1:], axis=1) + 1

    # the distribution of energy across FFT bins, using the same idea as mean and variance of a probability distribution.
    # spectral centroid (center of mass of bin indices)
    # Low centroid → energy concentrated at low bins / High centroid → energy at high bins → rapid oscillations
    spectral_centroid = _safe_divide(np.sum(freq_amplitude * freq_idx[None, :], axis=1), magnitude_sum)

    # spectral bandwidth (essentially variance around centroid), then normalized as fractional stddev
    # Bandwidth	Low	-> Energy concentrated near centroid (simple structure)
    # Bandwidth	High ->	Energy spread across bins (complex / noisy structure)
    spectral_bandwidth_var = _safe_divide(np.sum(freq_amplitude * (freq_idx[None, :] - spectral_centroid[:, None]) ** 2,
                                                 axis=1), magnitude_sum,)
    spectral_bandwidth_std = np.sqrt(np.maximum(spectral_bandwidth_var, 0.0))
    # fractional bandwidth is std divided by max index (optional; later StandardScaler will standardize)
    spectral_bandwidth_frac = spectral_bandwidth_std / max(n_freq_bins - 1, 1)

    feature_arrays = [
        channel_values.mean(axis=1),  # mean amplitude
        std,  # standard deviation (spread)
        channel_values.min(axis=1),  # minimum value
        channel_values.max(axis=1),  # maximum value
        np.ptp(channel_values, axis=1),  # peak-to-peak (max - min)
        np.median(channel_values, axis=1),  # median value
        q25,  # 25th percentile (first quartile)
        q75,  # 75th percentile (third quartile)
        q75 - q25,  # interquartile range (IQR)
        np.mean(np.abs(channel_values), axis=1),  # mean absolute amplitude
        np.sqrt(np.mean(channel_values ** 2, axis=1)),  # RMS (root mean square)
        zero_crossing_rate,  # fraction of sign changes (zero-crossing rate)
        slope,  # linear trend (slope) across bins
        skewness,  # skewness (asymmetry of distribution)
        kurtosis_excess,  # excess kurtosis (tailedness)
        np.argmax(channel_values, axis=1),  # argmax position
        np.argmin(channel_values, axis=1),  # argmin position
        dominant_freq_bin,  # dominant frequency bin (raw index)
        spectral_centroid,  # spectral centroid (in bin-index units)
        spectral_bandwidth_std,  # spectral bandwidth (std in bin-index units)
        spectral_entropy,  # spectral entropy (nats)
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

##  main function to extract feature matrix of shape [N, F] and list of feature names from signal tensor of shape [N, C, L]
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


