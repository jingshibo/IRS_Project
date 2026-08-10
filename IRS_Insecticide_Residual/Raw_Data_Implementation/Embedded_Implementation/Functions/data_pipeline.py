from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Functions.config import (
    EmbeddedPipelineConfig,
)
from IRS_Insecticide_Residual.Utility_Functions import Preprocessing


def build_raw_multichannel_dataset(config: EmbeddedPipelineConfig) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Run the same raw-signal preprocessing used by Classify_Raw_Data.py."""
    df = pd.read_excel(config.excel_path, sheet_name=0)
    label_col = df.columns[0]
    df_clean, removed_zero_sample_indices = Preprocessing.remove_zero_samples(
        df,
        label_col=label_col,
        reset_index=True,
    )
    categorized_dict = {
        key: group.drop(columns=[label_col]).reset_index(drop=True)
        for key, group in df_clean.groupby(label_col)
    }

    sliced_dict = Preprocessing.slice_dict_signal_segments(
        categorized_dict,
        segments=config.signal_segments,
    )
    sliced_filtered_dict = Preprocessing.fast_spike_filter_dict(
        sliced_dict,
        radius=config.spike_radius,
        transform=config.spike_transform,
        method=config.spike_method,
        n_sigmas=config.spike_n_sigmas,
        k=config.spike_k,
        min_threshold=config.spike_min_threshold,
    )
    original_filtered_dict = Preprocessing.apply_savgol_filter_dict(
        sliced_filtered_dict,
        window_length=config.savgol_window_length,
        polyorder=config.savgol_polyorder,
        deriv=config.savgol_deriv,
        mode=config.savgol_mode,
    )
    original_filtered_dict = Preprocessing.downsample_dict_signals(
        original_filtered_dict,
        step=config.downsample_step,
        offset=config.downsample_offset,
    )

    central_diff_dict = Preprocessing.compute_central_diff_dict(original_filtered_dict)
    central_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(
        central_diff_dict,
        window_length=config.savgol_window_length,
        polyorder=config.savgol_polyorder,
        deriv=config.savgol_deriv,
        mode=config.savgol_mode,
    )
    second_diff_dict = Preprocessing.compute_second_central_diff_dict(original_filtered_dict)
    second_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(
        second_diff_dict,
        window_length=config.savgol_window_length,
        polyorder=config.savgol_polyorder,
        deriv=config.savgol_deriv,
        mode=config.savgol_mode,
    )

    rolling_variance_dict = Preprocessing.compute_rolling_variance_dict(
        original_filtered_dict,
        window_size=config.rolling_window_size,
    )
    derivative_energy_dict = Preprocessing.compute_derivative_energy_dict(
        central_diff_filtered_dict,
        window_size=config.rolling_window_size,
    )
    original_envelope_dict = Preprocessing.apply_savgol_filter_dict(
        original_filtered_dict,
        window_length=config.savgol_window_length,
        polyorder=config.savgol_polyorder,
        deriv=config.savgol_deriv,
    )
    original_residual_dict = Preprocessing.calculate_residual_dict(
        original_filtered_dict,
        original_envelope_dict,
    )

    value_type_dicts = {
        "original": original_filtered_dict,
        "first_diff_filtered": central_diff_filtered_dict,
        "second_diff_filtered": second_diff_filtered_dict,
        "rolling_variance": rolling_variance_dict,
        "derivative_energy": derivative_energy_dict,
        "residual": original_residual_dict,
    }
    x_all, y_all = Preprocessing.build_multi_channel_dataset(
        data_dict_map=value_type_dicts,
        selected_types=config.selected_value_types,
    )
    return x_all, y_all, removed_zero_sample_indices


def fit_transform_channel_scalers(
    x_train: np.ndarray,
    x_test: np.ndarray,
    clip_max_value: Optional[float],
) -> tuple[np.ndarray, np.ndarray, list[StandardScaler]]:
    """Fit one StandardScaler per channel on train data and transform both splits."""
    x_train_norm = np.asarray(x_train, dtype=np.float32).copy()
    x_test_norm = np.asarray(x_test, dtype=np.float32).copy()
    scalers: list[StandardScaler] = []

    for channel_idx in range(x_train_norm.shape[1]):
        scaler = StandardScaler()
        scaler.fit(x_train_norm[:, channel_idx, :])
        x_train_norm[:, channel_idx, :] = scaler.transform(x_train_norm[:, channel_idx, :]).astype(
            np.float32,
            copy=False,
        )
        x_test_norm[:, channel_idx, :] = scaler.transform(x_test_norm[:, channel_idx, :]).astype(
            np.float32,
            copy=False,
        )
        scalers.append(scaler)

    if clip_max_value is not None:
        x_train_norm = np.clip(x_train_norm, a_min=None, a_max=clip_max_value).astype(np.float32, copy=False)
        x_test_norm = np.clip(x_test_norm, a_min=None, a_max=clip_max_value).astype(np.float32, copy=False)

    return x_train_norm, x_test_norm, scalers


def encode_labels(labels: Sequence[str], class_order: Sequence[str]) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    label_to_idx = {label: idx for idx, label in enumerate(class_order)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    encoded = np.asarray([label_to_idx[label] for label in labels], dtype=np.int64)
    return encoded, label_to_idx, idx_to_label
