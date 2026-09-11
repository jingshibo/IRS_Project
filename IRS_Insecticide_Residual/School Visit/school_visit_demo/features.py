from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from IRS_Insecticide_Residual.Feature_Implementation.Functions.Feature_Extraction import (
    extract_combined_features,
)

from .config import DOWNSAMPLE_STEP


ORIGINAL_BAND_EDGES = ((0, 1000), (1000, 1750), (1750, 2700))
BAND_EDGES = [
    (start // DOWNSAMPLE_STEP, end // DOWNSAMPLE_STEP)
    for start, end in ORIGINAL_BAND_EDGES
]


@dataclass(frozen=True)
class DemoFeatures:
    x_features: np.ndarray
    feature_names: list[str]
    feature_table: pd.DataFrame


def extract_simple_demo_features(
    x_all: np.ndarray,
    y_all: np.ndarray,
    channel_names: Sequence[str],
) -> DemoFeatures:
    """Extract compact, easy-to-explain features from each signal channel."""
    x_all = np.asarray(x_all, dtype=np.float32)
    y_all = np.asarray(y_all)
    if x_all.ndim != 3:
        raise ValueError(f"x_all must have shape [N, C, L], got {x_all.shape}")
    if x_all.shape[0] != y_all.shape[0]:
        raise ValueError("x_all and y_all must contain the same number of samples")
    if len(channel_names) != x_all.shape[1]:
        raise ValueError("channel_names must match the number of signal channels")

    feature_blocks = []
    feature_names: list[str] = []
    for channel_idx, channel_name in enumerate(channel_names):
        block, names = _simple_channel_features(x_all[:, channel_idx, :], str(channel_name))
        feature_blocks.append(block)
        feature_names.extend(names)

    x_features = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    feature_table = pd.DataFrame(x_features, columns=feature_names)
    feature_table.insert(0, "label", y_all)
    return DemoFeatures(
        x_features=x_features,
        feature_names=feature_names,
        feature_table=feature_table,
    )


def extract_complex_demo_features(
    x_all: np.ndarray,
    y_all: np.ndarray,
    channel_names: Sequence[str],
) -> DemoFeatures:
    """Extract full manual features using the main Feature_Implementation pipeline."""
    x_all = np.asarray(x_all, dtype=np.float32)
    y_all = np.asarray(y_all)
    if x_all.ndim != 3:
        raise ValueError(f"x_all must have shape [N, C, L], got {x_all.shape}")
    if x_all.shape[0] != y_all.shape[0]:
        raise ValueError("x_all and y_all must contain the same number of samples")
    if len(channel_names) != x_all.shape[1]:
        raise ValueError("channel_names must match the number of signal channels")

    resolved_channel_names = tuple(str(name) for name in channel_names)
    signal_channel_index = resolved_channel_names.index("processed_signal")
    first_derivative_channel_index = resolved_channel_names.index("first_difference")
    second_derivative_channel_index = resolved_channel_names.index("second_difference")

    x_features, feature_names, _metadata = extract_combined_features(
        x_signal=x_all,
        band_edges=BAND_EDGES,
        channel_names=resolved_channel_names,
        signal_channel_index=signal_channel_index,
        first_derivative_channel_index=first_derivative_channel_index,
        second_derivative_channel_index=second_derivative_channel_index,
        peak_selection="amplitude",
        dip_selection="amplitude",
        min_prominence_frac=0.20,
        min_distance=1,
        min_width=1,
        percentile_method="histogram",
        general_peak_rel_height=0.5,
        main_peak_rel_height=0.9,
        dip_rel_height=0.5,
        derivative_window_radius=5,
        include_area_features=True,
        include_inter_band=True,
        include_broad_transition=True,
        include_second_derivative=True,
    )

    feature_table = pd.DataFrame(x_features, columns=feature_names)
    feature_table.insert(0, "label", y_all)
    return DemoFeatures(
        x_features=x_features,
        feature_names=feature_names,
        feature_table=feature_table,
    )


def extract_demo_features(
    x_all: np.ndarray,
    y_all: np.ndarray,
    channel_names: Sequence[str],
) -> DemoFeatures:
    """Backward-compatible name for the full manual feature extractor."""
    return extract_complex_demo_features(x_all, y_all, channel_names)


def feature_examples_table(
    feature_table: pd.DataFrame,
    class_order: Sequence[str],
    rows_per_class: int = 2,
) -> pd.DataFrame:
    """Return a small table suitable for showing during the visit."""
    compact_columns = [
        "label",
        "processed_signal__mean",
        "processed_signal__std",
        "processed_signal__peak_to_peak",
        "processed_signal__argmax_frac",
        "processed_signal__band1_main_peak_amp",
        "processed_signal__band2_main_peak_amp",
        "processed_signal__derivative__band1_main_peak_left_slope",
    ]
    available_columns = [name for name in compact_columns if name in feature_table.columns]

    rows = []
    for label in class_order:
        class_rows = feature_table.loc[feature_table["label"] == label, available_columns]
        rows.append(class_rows.head(rows_per_class))

    out = pd.concat(rows, axis=0, ignore_index=True)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(4)
    return out


def _simple_channel_features(
    channel_values: np.ndarray,
    channel_name: str,
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(channel_values, dtype=np.float32)
    signal_length = values.shape[1]
    x_axis = np.linspace(0.0, 1.0, signal_length, dtype=np.float32)

    mean = values.mean(axis=1)
    std = values.std(axis=1)
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)
    peak_to_peak = np.ptp(values, axis=1)
    median = np.median(values, axis=1)
    q25 = np.percentile(values, 25.0, axis=1)
    q75 = np.percentile(values, 75.0, axis=1)
    area = _trapezoid(values, x_axis)
    abs_area = _trapezoid(np.abs(values), x_axis)
    energy = np.mean(values**2, axis=1)
    peak_position = np.argmax(values, axis=1) / max(signal_length - 1, 1)
    valley_position = np.argmin(values, axis=1) / max(signal_length - 1, 1)

    block = np.column_stack(
        [
            mean,
            std,
            minimum,
            maximum,
            peak_to_peak,
            median,
            q25,
            q75,
            q75 - q25,
            area,
            abs_area,
            energy,
            peak_position,
            valley_position,
        ]
    )
    names = [
        f"{channel_name}__mean",
        f"{channel_name}__std",
        f"{channel_name}__min",
        f"{channel_name}__max",
        f"{channel_name}__peak_to_peak",
        f"{channel_name}__median",
        f"{channel_name}__q25",
        f"{channel_name}__q75",
        f"{channel_name}__iqr",
        f"{channel_name}__area",
        f"{channel_name}__abs_area",
        f"{channel_name}__energy",
        f"{channel_name}__peak_position",
        f"{channel_name}__valley_position",
    ]
    return block.astype(np.float32, copy=False), names


def _trapezoid(values: np.ndarray, x_axis: np.ndarray) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=x_axis, axis=1)
    return np.trapz(values, x=x_axis, axis=1)
