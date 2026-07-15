"""
Core-view version of Feature_Extraction.py.

This file is for reading the extraction flow only. It intentionally removes most
input validation and consistency checks from Feature_Extraction.py so the main
global -> peak/dip -> derivative feature logic is easier to inspect.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from IRS_Insecticide_Residual.Feature_Implementation.Functions.Global_Features import calculate_global_features


def _resolve_channel_names(n_channels: int, channel_names: Optional[Sequence[str]]) -> List[str]:
    if channel_names is None:
        return [f"channel_{idx}" for idx in range(n_channels)]
    return [str(name) for name in channel_names]


def _feature_dicts_to_matrix(feature_dicts: List[dict]) -> Tuple[np.ndarray, List[str]]:
    feature_names = list(feature_dicts[0].keys())
    feature_matrix = np.asarray(
        [[sample_features[name] for name in feature_names] for sample_features in feature_dicts],
        dtype=np.float32,
    )
    return feature_matrix, feature_names


def _prefix_feature_dict(features: dict, prefix: str) -> dict:
    return {f"{prefix}__{name}": np.float32(value) for name, value in features.items()}


def _metadata_for_channel(metadata_by_channel, channel_idx: int, channel_name: str):
    if isinstance(metadata_by_channel, dict):
        return metadata_by_channel[channel_idx] if channel_idx in metadata_by_channel else metadata_by_channel[channel_name]
    return metadata_by_channel


def extract_global_features(
    x: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract global features from every channel and concatenate them by feature columns."""
    x = np.asarray(x, dtype=np.float32)
    resolved_channel_names = _resolve_channel_names(x.shape[1], channel_names)

    feature_blocks = []
    feature_names: List[str] = []
    for channel_idx, channel_name in enumerate(resolved_channel_names):
        channel_block, channel_feature_names = calculate_global_features(
            x[:, channel_idx, :],
            channel_name=channel_name,
        )
        feature_blocks.append(channel_block)
        feature_names.extend(channel_feature_names)

    return np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False), feature_names


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
    """Detect raw peaks/dips for each sample, then select one fixed peak/dip structure per band."""
    from IRS_Insecticide_Residual.Feature_Implementation.Functions import Peak_Dip_Features

    detected_peak_dip_list = []
    peak_dip_pairs_list = []

    for sample_signal in np.asarray(channel_values, dtype=np.float32):
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
    """Convert selected peak/dip pairs for one channel into a feature matrix."""
    from IRS_Insecticide_Residual.Feature_Implementation.Functions import Peak_Dip_Features

    feature_dicts: List[dict] = []
    for sample_signal, peak_dip_pairs in zip(np.asarray(channel_values, dtype=np.float32), peak_dip_pairs_list):
        sample_features = Peak_Dip_Features.calculate_doublet_features(peak_dip_pairs)
        if include_area_features:
            sample_features.update(
                Peak_Dip_Features.calculate_doublet_area_features(peak_dip_pairs, sample_signal)
            )
        feature_dicts.append(_prefix_feature_dict(sample_features, channel_name))

    return _feature_dicts_to_matrix(feature_dicts)


def extract_peak_dip_features(
    x: np.ndarray,
    peak_dip_pairs_by_channel,
    channel_names: Optional[Sequence[str]] = None,
    channel_indices: Optional[Sequence[int]] = None,
    include_area_features: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Extract peak/dip features for selected channels using precomputed peak/dip pairs."""
    x = np.asarray(x, dtype=np.float32)
    resolved_channel_names = _resolve_channel_names(x.shape[1], channel_names)
    resolved_channel_indices = list(range(x.shape[1])) if channel_indices is None else list(channel_indices)

    feature_blocks = []
    feature_names: List[str] = []
    for channel_idx in resolved_channel_indices:
        channel_name = resolved_channel_names[channel_idx]
        channel_peak_dip_pairs = _metadata_for_channel(peak_dip_pairs_by_channel, channel_idx, channel_name)
        channel_block, channel_feature_names = calculate_peak_dip_feature_matrix(
            channel_values=x[:, channel_idx, :],
            peak_dip_pairs_list=channel_peak_dip_pairs,
            channel_name=channel_name,
            include_area_features=include_area_features,
        )
        feature_blocks.append(channel_block)
        feature_names.extend(channel_feature_names)

    return np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False), feature_names


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
    """Convert peak/dip structures plus derivative signals into a derivative feature matrix."""
    from IRS_Insecticide_Residual.Feature_Implementation.Functions import Derivative_Features

    signals = np.asarray(signals, dtype=np.float32)
    first_derivatives = None if first_derivatives is None else np.asarray(first_derivatives, dtype=np.float32)
    second_derivatives = None if second_derivatives is None else np.asarray(second_derivatives, dtype=np.float32)

    feature_dicts: List[dict] = []
    for sample_idx, sample_signal in enumerate(signals):
        sample_features = Derivative_Features.calculate_derivative_features(
            selected_pairs=selected_pairs_list[sample_idx],
            signal=sample_signal,
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
    """Select original/derivative channels, then extract derivative features."""
    x = np.asarray(x, dtype=np.float32)
    resolved_channel_names = _resolve_channel_names(x.shape[1], channel_names)

    signals = x[:, signal_channel_index, :]
    first_derivatives = None if first_derivative_channel_index is None else x[:, first_derivative_channel_index, :]
    second_derivatives = None if second_derivative_channel_index is None else x[:, second_derivative_channel_index, :]

    if feature_prefix == "derivative":
        feature_prefix = f"{resolved_channel_names[signal_channel_index]}__derivative"

    return calculate_derivative_feature_matrix(
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


def extract_combined_features(
    x_signal: np.ndarray,
    band_edges: list[tuple[float, float]],
    channel_names: Sequence[str],
    signal_channel_index: int,
    first_derivative_channel_index: int,
    second_derivative_channel_index: int,
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
    derivative_window_radius: int = 5,
    include_area_features: bool = True,
    include_inter_band: bool = True,
    include_broad_transition: bool = True,
    include_second_derivative: bool = True,
) -> tuple[np.ndarray, List[str], dict]:
    """Run the full feature flow: global features, peak/dip features, then derivative features."""
    x_signal = np.asarray(x_signal, dtype=np.float32)
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
    """Apply the same combined feature flow to train/validation and holdout test sets."""
    x_trainval_features, trainval_feature_names, trainval_metadata = extract_combined_features(
        x_trainval_signal,
        band_edges=band_edges,
        channel_names=channel_names,
        signal_channel_index=signal_channel_index,
        first_derivative_channel_index=first_derivative_channel_index,
        second_derivative_channel_index=second_derivative_channel_index,
        **feature_kwargs,
    )
    x_test_features, _, test_metadata = extract_combined_features(
        x_test_signal,
        band_edges=band_edges,
        channel_names=channel_names,
        signal_channel_index=signal_channel_index,
        first_derivative_channel_index=first_derivative_channel_index,
        second_derivative_channel_index=second_derivative_channel_index,
        **feature_kwargs,
    )

    metadata = {
        "trainval": trainval_metadata,
        "test": test_metadata,
    }
    return x_trainval_features, x_test_features, trainval_feature_names, metadata
