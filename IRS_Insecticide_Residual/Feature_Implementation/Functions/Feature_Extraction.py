from typing import List, Optional, Sequence, Tuple

import numpy as np

from IRS_Insecticide_Residual.Feature_Implementation.Functions.Global_Features import calculate_global_features


def _resolve_channel_names(n_channels: int, channel_names: Optional[Sequence[str]]) -> List[str]:
    if channel_names is None:
        return [f"channel_{idx}" for idx in range(n_channels)]

    if len(channel_names) != n_channels:
        raise ValueError(
            f"channel_names length must match channel count {n_channels}, got {len(channel_names)}"
        )
    return [str(name) for name in channel_names]


def _validate_signal_dataset(x: np.ndarray) -> np.ndarray:
    """Validate that x has shape [N, C, L] and signal length >= 2."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"x must have shape [N, C, L], got {x.shape}")
    if x.shape[2] < 2:
        raise ValueError(f"Signal length must be >= 2, got {x.shape[2]}")
    return x


def _channel_feature_block(channel_values: np.ndarray, channel_name: str) -> Tuple[np.ndarray, List[str]]:
    """
    Extract global features for one channel block.

    Inputs:
        channel_values: np.ndarray with shape [N, L], where N is sample count and L is signal length.
        channel_name: Prefix used in returned feature names.

    Outputs:
        feature_matrix: np.ndarray with shape [N, F_global], dtype float32.
        feature_names: list[str] with length F_global.
    """
    channel_values = np.asarray(channel_values, dtype=np.float32)
    if channel_values.ndim != 2:
        raise ValueError(f"channel_values must have shape [N, L], got {channel_values.shape}")

    return calculate_global_features(channel_values, channel_name=channel_name)


def _feature_dicts_to_matrix(feature_dicts: List[dict]) -> Tuple[np.ndarray, List[str]]:
    if not feature_dicts:
        raise ValueError("feature_dicts must contain at least one sample")

    feature_names = list(feature_dicts[0].keys())
    feature_matrix = np.empty((len(feature_dicts), len(feature_names)), dtype=np.float32)
    expected_names = set(feature_names)

    for sample_idx, sample_features in enumerate(feature_dicts):
        sample_names = set(sample_features.keys())
        if sample_names != expected_names: # Checks that every sample has exactly the same features.
            missing = sorted(expected_names - sample_names)
            extra = sorted(sample_names - expected_names)
            raise ValueError( # Raises a ValueError if feature names are inconsistent.
                "Feature names are inconsistent across samples: "
                f"sample_idx={sample_idx}, missing={missing}, extra={extra}"
            )
        feature_matrix[sample_idx, :] = np.asarray( # Fills one row of the matrix using the fixed feature_names order.
            [sample_features[name] for name in feature_names],
            dtype=np.float32,
        )

    return feature_matrix, feature_names


def _prefix_feature_dict(features: dict, prefix: str) -> dict:
    return {f"{prefix}__{name}": np.float32(value) for name, value in features.items()}


def _validate_optional_channel_matrix(
    values: Optional[np.ndarray],
    expected_shape: tuple[int, int],
    name: str,
) -> Optional[np.ndarray]:
    if values is None:
        return None

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [N, L], got {values.shape}")
    if values.shape != expected_shape:
        raise ValueError(f"{name} shape must match signals shape {expected_shape}, got {values.shape}")
    return values


def _validate_metadata_list(values: Optional[list], n_samples: int, name: str) -> Optional[list]:
    if values is None:
        return None
    if len(values) != n_samples:
        raise ValueError(f"{name} length must match sample count {n_samples}, got {len(values)}")
    return values


def _validate_channel_index(channel_index: Optional[int], n_channels: int, name: str) -> Optional[int]:
    if channel_index is None:
        return None

    channel_index = int(channel_index)
    if channel_index < 0 or channel_index >= n_channels:
        raise ValueError(f"{name} must be in [0, {n_channels - 1}], got {channel_index}")
    return channel_index


def _metadata_for_channel(
    metadata_by_channel,
    channel_idx: int,
    channel_name: str,
    metadata_name: str,
):
    """Return metadata for one channel, accepting either shared metadata or a dict keyed by index/name."""
    if isinstance(metadata_by_channel, dict):
        if channel_idx in metadata_by_channel:
            return metadata_by_channel[channel_idx]
        if channel_name in metadata_by_channel:
            return metadata_by_channel[channel_name]
        raise ValueError(
            f"{metadata_name} must contain key {channel_idx!r} or {channel_name!r} for channel {channel_name}"
        )
    return metadata_by_channel


def extract_global_features(
    x: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract global/statistical features from every channel in a signal tensor.

    Inputs:
        x: np.ndarray with shape [N, C, L], where C is channel count.
        channel_names: Optional sequence of C names. If None, uses channel_0, channel_1, ...

    Outputs:
        feature_matrix: np.ndarray with shape [N, C * F_global], dtype float32.
        feature_names: list[str] with one name per output feature.
    """
    x = _validate_signal_dataset(x)
    n_channels = x.shape[1]

    resolved_channel_names = _resolve_channel_names(n_channels, channel_names)

    feature_blocks = []
    feature_names: List[str] = []
    for channel_idx, channel_name in enumerate(resolved_channel_names):
        channel_block, channel_feature_names = _channel_feature_block(x[:, channel_idx, :], channel_name=channel_name)
        feature_blocks.append(channel_block)
        feature_names.extend(channel_feature_names)

    feature_matrix = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    return feature_matrix, feature_names


def detect_peak_dip_pairs_for_channel(
    channel_values: np.ndarray,
    band_edges: list[tuple[float, float]],
    peak_selection: str = "amplitude",
    dip_selection: str = "amplitude",
    min_prominence_frac: float = 0.20,
    min_distance: int = 1,
    min_width: int = 1,
    percentile_method: str = "histogram",
    histogram_bins: int = 50,
    general_peak_rel_height: float = 0.5,
    main_peak_rel_height: float = 0.9,
    dip_rel_height: float = 0.5,
) -> tuple[list[dict], list[list[dict]]]:
    """
    Detect peaks/dips and select fixed band-wise peak/dip pairs for one channel.

    Inputs:
        channel_values: np.ndarray with shape [N, L].
        band_edges: list of (start, end) index ranges in the downsampled signal.
        peak/dip parameters: Passed to Peak_Dip_Features.detect_peaks_and_dips()
            and select_band_peak_dip_pairs().

    Outputs:
        detected_peak_dip_list: list[dict] with length N. Each dict contains raw detected
            peak/dip arrays for one sample.
        peak_dip_pairs_list: list[list[dict]] with length N. Each inner list contains one
            selected peak/dip-pair dictionary per band for one sample.
    """
    from IRS_Insecticide_Residual.Feature_Implementation.Functions import Peak_Dip_Features

    channel_values = np.asarray(channel_values, dtype=np.float32)
    if channel_values.ndim != 2:
        raise ValueError(f"channel_values must have shape [N, L], got {channel_values.shape}")
    if channel_values.shape[0] < 1:
        raise ValueError("channel_values must contain at least one sample")
    if channel_values.shape[1] < 2:
        raise ValueError(f"Signal length must be >= 2, got {channel_values.shape[1]}")

    detected_peak_dip_list = []
    peak_dip_pairs_list = []

    for sample_signal in channel_values:
        detected_peak_dip = Peak_Dip_Features.detect_peaks_and_dips(
            sample_signal,
            min_prominence_frac=min_prominence_frac,
            min_distance=min_distance,
            min_width=min_width,
            percentile_method=percentile_method,
            histogram_bins=histogram_bins,
            general_peak_rel_height=general_peak_rel_height,
            main_peak_rel_height=main_peak_rel_height,
            dip_rel_height=dip_rel_height,
        )
        peak_dip_pairs = Peak_Dip_Features.select_band_peak_dip_pairs(
            detected_peak_dip,
            band_edges=band_edges,
            peak_selection=peak_selection,
            dip_selection=dip_selection,
        )
        detected_peak_dip_list.append(detected_peak_dip)
        peak_dip_pairs_list.append(peak_dip_pairs)

    return detected_peak_dip_list, peak_dip_pairs_list


def calculate_peak_dip_feature_matrix(
    channel_values: np.ndarray,
    peak_dip_pairs_list: list[list[dict]],
    channel_name: str,
    include_area_features: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract peak/dip morphology and optional area features from a single channel based on precomputed peak/dip pairs.

    Inputs:
        channel_values: np.ndarray with shape [N, L]. Used for area features.
        peak_dip_pairs_list: list[list[dict]] with length N, usually returned by
            detect_peak_dip_pairs_for_channel(...).
        channel_name: Prefix used in returned feature names.
        include_area_features: If True, append peak/dip area features.

    Outputs:
        feature_matrix: np.ndarray with shape [N_sample, N_feature], dtype float32.
        feature_names: list[str] with length N_feature.
    """
    from IRS_Insecticide_Residual.Feature_Implementation.Functions import Peak_Dip_Features

    channel_values = np.asarray(channel_values, dtype=np.float32)
    if channel_values.ndim != 2:
        raise ValueError(f"channel_values must have shape [N, L], got {channel_values.shape}")
    if channel_values.shape[0] < 1:
        raise ValueError("channel_values must contain at least one sample")
    if channel_values.shape[1] < 2:
        raise ValueError(f"Signal length must be >= 2, got {channel_values.shape[1]}")

    n_samples = channel_values.shape[0]
    peak_dip_pairs_list = _validate_metadata_list(peak_dip_pairs_list, n_samples, "peak_dip_pairs_list")

    feature_dicts: List[dict] = []
    for sample_signal, peak_dip_pairs in zip(channel_values, peak_dip_pairs_list):
        sample_features = Peak_Dip_Features.calculate_doublet_features(peak_dip_pairs)
        if include_area_features:
            sample_features.update( # add area features into the existing dictionary.
                Peak_Dip_Features.calculate_doublet_area_features(peak_dip_pairs, sample_signal)
            )
        feature_dicts.append(_prefix_feature_dict(sample_features, channel_name)) # adds a channel prefix to feature name.

    return _feature_dicts_to_matrix(feature_dicts) # converts that list of dictionaries into a feature matrix



def extract_peak_dip_features(
    x: np.ndarray,
    peak_dip_pairs_by_channel,
    channel_names: Optional[Sequence[str]] = None,
    channel_indices: Optional[Sequence[int]] = None,
    include_area_features: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract peak/dip features from multiple channels using precomputed peak/dip pairs.

    Inputs:
        x: np.ndarray with shape [N, C, L].
        peak_dip_pairs_by_channel: Either one list[list[dict]] applied to selected channels,
            or a dict keyed by channel index/name with each value shaped as list[list[dict]].
        channel_names: Optional sequence of C names.
        channel_indices: Optional sequence of channel indices to extract. If None, extracts all.
        include_area_features: If True, append peak/dip area features.

    Outputs:
        feature_matrix: np.ndarray with shape [N, F_peak_dip_total], dtype float32.
        feature_names: list[str] with length F_peak_dip_total.
    """
    x = _validate_signal_dataset(x)
    n_channels = x.shape[1]
    resolved_channel_names = _resolve_channel_names(n_channels, channel_names)

    if channel_indices is None:
        resolved_channel_indices = list(range(n_channels))
    else:
        resolved_channel_indices = [int(idx) for idx in channel_indices]
        if not resolved_channel_indices:
            raise ValueError("channel_indices must contain at least one channel")
        invalid_indices = [idx for idx in resolved_channel_indices if idx < 0 or idx >= n_channels]
        if invalid_indices:
            raise ValueError(
                f"channel_indices must be in [0, {n_channels - 1}], got {invalid_indices}"
            )

    feature_blocks = []
    feature_names: List[str] = []

    for channel_idx in resolved_channel_indices:
        channel_name = resolved_channel_names[channel_idx]
        channel_peak_dip_pairs = _metadata_for_channel(
            peak_dip_pairs_by_channel,
            channel_idx,
            channel_name,
            "peak_dip_pairs_by_channel",
        )
        channel_block, channel_feature_names = calculate_peak_dip_feature_matrix(
            channel_values=x[:, channel_idx, :],
            peak_dip_pairs_list=channel_peak_dip_pairs,
            channel_name=channel_name,
            include_area_features=include_area_features,
        )

        feature_blocks.append(channel_block)
        feature_names.extend(channel_feature_names)

    feature_matrix = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    return feature_matrix, feature_names


def calculate_derivative_feature_matrix(
    signals: np.ndarray,
    selected_pairs_list: list[list[dict]],
    feature_prefix: str = "derivative",
    first_derivatives: Optional[np.ndarray] = None,
    second_derivatives: Optional[np.ndarray] = None,
    detected_peak_dip_list: Optional[list[dict]] = None,
    window_radius: int = 5,
    include_inter_band: bool = True,
    include_broad_transition: bool = True,
    include_second_derivative: bool = True,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract first/second-derivative features from one original-signal channel.

    Inputs:
        signals: Original signal np.ndarray with shape [N, L].
        selected_pairs_list: list[list[dict]] with length N, one selected peak/dip-pair list
            per sample.
        feature_prefix: Prefix used in returned feature names.
        first_derivatives: Optional np.ndarray with shape [N, L]. If None, Derivative_Features
            uses its internal fallback.
        second_derivatives: Optional np.ndarray with shape [N, L]. Used when
            include_second_derivative=True.
        detected_peak_dip_list: Optional list[dict] with length N, used for inter-band split
            segments.

    Outputs:
        feature_matrix: np.ndarray with shape [N, F_derivative], dtype float32.
        feature_names: list[str] with length F_derivative.
    """
    from IRS_Insecticide_Residual.Feature_Implementation.Functions import Derivative_Features

    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape [N, L], got {signals.shape}")
    if signals.shape[0] < 1:
        raise ValueError("signals must contain at least one sample")
    if signals.shape[1] < 2:
        raise ValueError(f"Signal length must be >= 2, got {signals.shape[1]}")

    n_samples = signals.shape[0]
    if selected_pairs_list is None:
        raise ValueError("selected_pairs_list must contain one selected peak/dip pair list per sample")
    selected_pairs_list = _validate_metadata_list(selected_pairs_list, n_samples, "selected_pairs_list")
    detected_peak_dip_list = _validate_metadata_list(detected_peak_dip_list, n_samples, "detected_peak_dip_list")
    first_derivatives = _validate_optional_channel_matrix(first_derivatives, signals.shape, "first_derivatives")
    second_derivatives = _validate_optional_channel_matrix(second_derivatives, signals.shape, "second_derivatives")

    feature_dicts: List[dict] = []
    for sample_idx in range(n_samples):
        sample_features = Derivative_Features.calculate_derivative_features(
            selected_pairs=selected_pairs_list[sample_idx],
            signal=signals[sample_idx],
            first_derivative=None if first_derivatives is None else first_derivatives[sample_idx],
            detected_peak_dip=None if detected_peak_dip_list is None else detected_peak_dip_list[sample_idx],
            second_derivative=None if second_derivatives is None else second_derivatives[sample_idx],
            window_radius=window_radius,
            include_inter_band=include_inter_band,
            include_broad_transition=include_broad_transition,
            include_second_derivative=include_second_derivative,
            eps=eps,
        )
        feature_dicts.append(_prefix_feature_dict(sample_features, feature_prefix))

    return _feature_dicts_to_matrix(feature_dicts)


def extract_derivative_features(
    x: np.ndarray,
    selected_pairs_list: list[list[dict]],
    channel_names: Optional[Sequence[str]] = None,
    signal_channel_index: int = 0,
    first_derivative_channel_index: Optional[int] = None,
    second_derivative_channel_index: Optional[int] = None,
    feature_prefix: str = "derivative",
    detected_peak_dip_list: Optional[list[dict]] = None,
    window_radius: int = 5,
    include_inter_band: bool = True,
    include_broad_transition: bool = True,
    include_second_derivative: bool = False,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract derivative features from a multi-channel signal tensor.

    Inputs:
        x: np.ndarray with shape [N, C, L].
        selected_pairs_list: list[list[dict]] with length N, usually returned by
            detect_peak_dip_pairs_for_channel(...) on the original signal channel.
        channel_names: Optional sequence of C names.
        signal_channel_index: Channel containing the original signal.
        first_derivative_channel_index: Optional channel containing precomputed first derivative.
        second_derivative_channel_index: Optional channel containing precomputed second derivative.
        detected_peak_dip_list: Optional list[dict] with length N for inter-band split segments.

    Outputs:
        feature_matrix: np.ndarray with shape [N, F_derivative], dtype float32.
        feature_names: list[str] with length F_derivative.
    """
    x = _validate_signal_dataset(x)
    _, n_channels, _ = x.shape
    resolved_channel_names = _resolve_channel_names(n_channels, channel_names)

    signal_channel_index = _validate_channel_index(signal_channel_index, n_channels, "signal_channel_index")
    first_derivative_channel_index = _validate_channel_index(first_derivative_channel_index, n_channels,
        "first_derivative_channel_index",)
    second_derivative_channel_index = _validate_channel_index(second_derivative_channel_index, n_channels,
        "second_derivative_channel_index",)

    signals = x[:, signal_channel_index, :]
    first_derivatives = None
    second_derivatives = None
    if first_derivative_channel_index is not None:
        first_derivatives = x[:, first_derivative_channel_index, :]
    if second_derivative_channel_index is not None:
        second_derivatives = x[:, second_derivative_channel_index, :]

    if feature_prefix == "derivative":
        feature_prefix = f"{resolved_channel_names[signal_channel_index]}__derivative"

    feature_matrix, feature_names = calculate_derivative_feature_matrix(
        signals=signals,
        selected_pairs_list=selected_pairs_list,
        feature_prefix=feature_prefix,
        first_derivatives=first_derivatives,
        second_derivatives=second_derivatives,
        detected_peak_dip_list=detected_peak_dip_list,
        window_radius=window_radius,
        include_inter_band=include_inter_band,
        include_broad_transition=include_broad_transition,
        include_second_derivative=include_second_derivative,
        eps=eps,
    )

    return feature_matrix, feature_names


def extract_combined_features(
    x_signal: np.ndarray,  # input signals with shape [N samples, C channels, L points]
    band_edges: list[tuple[float, float]],  # frequency/index ranges used to select band-wise peaks and dips
    channel_names: Sequence[str],  # names for each input channel, used as feature-name prefixes
    signal_channel_index: int,  # channel index for original signal peak/dip detection
    first_derivative_channel_index: int,  # channel index containing the precomputed first derivative
    second_derivative_channel_index: int,  # channel index containing the precomputed second derivative
    peak_selection: str = "amplitude",  # rule for choosing peaks within each band
    dip_selection: str = "amplitude",  # rule for choosing the middle dip between selected peaks
    min_prominence_frac: float = 0.20,  # minimum prominence threshold as a fraction of signal range
    min_distance: int = 1,  # minimum point distance between detected peaks or dips
    min_width: int = 1,  # minimum accepted peak/dip width from detection
    percentile_method: str = "histogram",  # method used to estimate robust signal percentiles
    histogram_bins: int = 50,  # number of bins used when percentile_method is histogram
    general_peak_rel_height: float = 0.5,  # relative prominence level for normal peak width calculation
    main_peak_rel_height: float = 0.9,  # relative prominence level for main peak width calculation
    dip_rel_height: float = 0.5,  # relative prominence level for dip width calculation
    derivative_window_radius: int = 5,  # points on each side of a peak for local derivative features
    include_area_features: bool = True,  # whether to include peak/dip area features
    include_inter_band: bool = True,  # whether to include derivative features between adjacent bands
    include_broad_transition: bool = True,  # whether to include full peak-to-peak inter-band transitions
    include_second_derivative: bool = True,  # whether to include compact second-derivative curvature features
) -> tuple[np.ndarray, List[str], dict]:
    """
    Extract global, peak/dip, and derivative features for one already-split dataset.

    Inputs:
        x_signal: np.ndarray with shape [N, C, L].
        band_edges: list of (start, end) index ranges for peak/dip band selection.
        channel_names: Sequence of C names.
        signal_channel_index: Channel used for peak/dip detection and original-signal features.
        first_derivative_channel_index: Channel containing precomputed first derivative.
        second_derivative_channel_index: Channel containing precomputed second derivative.

    Outputs:
        feature_matrix: np.ndarray with shape [N, F_total], dtype float32.
        feature_names: list[str] with length F_total.
        metadata: dict containing feature-name groups plus detected_peak_dip and peak_dip_pairs.
    """
    x_signal = _validate_signal_dataset(x_signal)
    resolved_channel_names = _resolve_channel_names(x_signal.shape[1], channel_names)

    global_features, global_feature_names = extract_global_features(
        x_signal,
        channel_names=resolved_channel_names,
    )

    detected_peak_dip, peak_dip_pairs = detect_peak_dip_pairs_for_channel(
        x_signal[:, signal_channel_index, :],
        band_edges=band_edges,
        peak_selection=peak_selection,
        dip_selection=dip_selection,
        min_prominence_frac=min_prominence_frac,
        min_distance=min_distance,
        min_width=min_width,
        percentile_method=percentile_method,
        histogram_bins=histogram_bins,
        general_peak_rel_height=general_peak_rel_height,
        main_peak_rel_height=main_peak_rel_height,
        dip_rel_height=dip_rel_height,
    )

    peak_dip_features, peak_dip_feature_names = calculate_peak_dip_feature_matrix(
        channel_values=x_signal[:, signal_channel_index, :],
        peak_dip_pairs_list=peak_dip_pairs,
        channel_name=resolved_channel_names[signal_channel_index],
        include_area_features=include_area_features,
    )

    derivative_features, derivative_feature_names = extract_derivative_features(
        x_signal,
        selected_pairs_list=peak_dip_pairs,
        detected_peak_dip_list=detected_peak_dip,
        channel_names=resolved_channel_names,
        signal_channel_index=signal_channel_index,
        first_derivative_channel_index=first_derivative_channel_index,
        second_derivative_channel_index=second_derivative_channel_index,
        window_radius=derivative_window_radius,
        include_inter_band=include_inter_band,
        include_broad_transition=include_broad_transition,
        include_second_derivative=include_second_derivative,
    )

    feature_matrix = np.concatenate(
        [global_features, peak_dip_features, derivative_features],
        axis=1,
    ).astype(np.float32, copy=False)
    feature_names = global_feature_names + peak_dip_feature_names + derivative_feature_names
    metadata = {
        "global_feature_names": global_feature_names,
        "peak_dip_feature_names": peak_dip_feature_names,
        "derivative_feature_names": derivative_feature_names,
        "detected_peak_dip": detected_peak_dip,
        "peak_dip_pairs": peak_dip_pairs,
    }
    return feature_matrix, feature_names, metadata


def extract_combined_features_for_train_test(
    x_trainval_signal: np.ndarray,
    x_test_signal: np.ndarray,
    band_edges: list[tuple[float, float]],
    channel_names: Sequence[str],
    signal_channel_index: int,
    first_derivative_channel_index: int,
    second_derivative_channel_index: int,
    **feature_kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str], dict]:
    """
    Extract combined features for train/val and holdout test with matching feature names.

    Inputs:
        x_trainval_signal: np.ndarray with shape [N_trainval, C, L].
        x_test_signal: np.ndarray with shape [N_test, C, L].
        band_edges: list of (start, end) index ranges for peak/dip band selection.
        channel_names: Sequence of C names.
        signal_channel_index: Channel used for peak/dip detection and original-signal features.
        first_derivative_channel_index: Channel containing precomputed first derivative.
        second_derivative_channel_index: Channel containing precomputed second derivative.
        **feature_kwargs: Optional peak/dip and derivative extraction parameters.

    Outputs:
        x_trainval_features: np.ndarray with shape [N_trainval, F_total], dtype float32.
        x_test_features: np.ndarray with shape [N_test, F_total], dtype float32.
        feature_names: list[str] with length F_total.
        metadata: dict with "trainval" and "test" entries containing per-split extraction metadata.
    """
    x_trainval_features, trainval_feature_names, trainval_metadata = extract_combined_features(
        x_trainval_signal,
        band_edges=band_edges,
        channel_names=channel_names,
        signal_channel_index=signal_channel_index,
        first_derivative_channel_index=first_derivative_channel_index,
        second_derivative_channel_index=second_derivative_channel_index,
        **feature_kwargs,
    )
    x_test_features, test_feature_names, test_metadata = extract_combined_features(
        x_test_signal,
        band_edges=band_edges,
        channel_names=channel_names,
        signal_channel_index=signal_channel_index,
        first_derivative_channel_index=first_derivative_channel_index,
        second_derivative_channel_index=second_derivative_channel_index,
        **feature_kwargs,
    )

    if trainval_feature_names != test_feature_names:
        raise ValueError("Train/val and test combined feature names do not match")

    metadata = {
        "trainval": trainval_metadata,
        "test": test_metadata,
    }
    return x_trainval_features, x_test_features, trainval_feature_names, metadata
