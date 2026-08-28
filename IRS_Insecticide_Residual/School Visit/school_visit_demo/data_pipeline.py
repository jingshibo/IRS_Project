from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from IRS_Insecticide_Residual.Utility_Functions import Preprocessing

from .config import CLASS_ORDER, DEFAULT_DATA_CANDIDATES, DOWNSAMPLE_STEP, SIGNAL_SEGMENTS


@dataclass(frozen=True)
class DemoSignalData:
    data_path: Path
    label_col: str
    class_order: tuple[str, ...]
    raw_by_class: dict[str, pd.DataFrame]
    full_smoothed_no_sqrt_by_class: dict[str, pd.DataFrame]
    full_smoothed_by_class: dict[str, pd.DataFrame]
    sliced_raw_by_class: dict[str, pd.DataFrame]
    despiked_by_class: dict[str, pd.DataFrame]
    smoothed_by_class: dict[str, pd.DataFrame]
    processed_by_class: dict[str, pd.DataFrame]
    first_diff_by_class: dict[str, pd.DataFrame]
    second_diff_by_class: dict[str, pd.DataFrame]
    x_all: np.ndarray
    y_all: np.ndarray
    selected_value_types: tuple[str, ...]
    removed_zero_sample_indices: list[int]


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    """Resolve an explicit path or one of the lab defaults."""
    candidates: list[Path] = []
    if data_path is not None:
        candidates.append(Path(data_path))
    candidates.extend(path for path in DEFAULT_DATA_CANDIDATES if path is not None)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find the insecticide residual Excel file.\n"
        "Set excel_path near the top of run_demo.py, or set IRS_SCHOOL_VISIT_DATA_PATH.\n"
        f"Checked:\n{candidate_text}"
    )


def load_grouped_insecticide_data(
    data_path: str | Path | None = None,
    sheet_name: int | str = 0,
    label_col: str | None = None,
    class_order: Sequence[str] = CLASS_ORDER,
) -> tuple[Path, str, tuple[str, ...], dict[str, pd.DataFrame], list[int]]:
    """Load the Excel file into the project convention: {class_label: signal dataframe}."""
    resolved_path = resolve_data_path(data_path)
    df = pd.read_excel(resolved_path, sheet_name=sheet_name)
    resolved_label_col = label_col if label_col is not None else str(df.columns[0])

    df_clean, removed_zero_sample_indices = Preprocessing.remove_zero_samples(
        df,
        label_col=resolved_label_col,
        reset_index=True,
    )

    grouped = {}
    for label, group in df_clean.groupby(resolved_label_col):
        signal_df = group.drop(columns=[resolved_label_col]).reset_index(drop=True)
        grouped[str(label)] = signal_df.apply(pd.to_numeric, errors="raise")

    ordered_labels = _resolve_class_order(grouped.keys(), class_order)
    grouped = {label: grouped[label] for label in ordered_labels}
    return resolved_path, resolved_label_col, ordered_labels, grouped, removed_zero_sample_indices


def build_demo_signal_data(
    data_path: str | Path | None = None,
    sheet_name: int | str = 0,
    label_col: str | None = None,
    class_order: Sequence[str] = CLASS_ORDER,
    signal_segments: Sequence[tuple[int, int]] = SIGNAL_SEGMENTS,
    downsample_step: int = DOWNSAMPLE_STEP,
) -> DemoSignalData:
    """Run the visitor-friendly version of the existing insecticide preprocessing pipeline."""
    (
        resolved_path,
        resolved_label_col,
        ordered_labels,
        raw_by_class,
        removed_zero_sample_indices,
    ) = load_grouped_insecticide_data(
        data_path=data_path,
        sheet_name=sheet_name,
        label_col=label_col,
        class_order=class_order,
    )

    return build_demo_signal_data_from_grouped(
        data_path=resolved_path,
        label_col=resolved_label_col,
        class_order=ordered_labels,
        raw_by_class=raw_by_class,
        removed_zero_sample_indices=removed_zero_sample_indices,
        signal_segments=signal_segments,
        downsample_step=downsample_step,
    )


def build_demo_signal_data_from_grouped(
    data_path: Path,
    label_col: str,
    class_order: Sequence[str],
    raw_by_class: dict[str, pd.DataFrame],
    removed_zero_sample_indices: list[int],
    signal_segments: Sequence[tuple[int, int]] = SIGNAL_SEGMENTS,
    downsample_step: int = DOWNSAMPLE_STEP,
) -> DemoSignalData:
    """Build processed demo signals from already-loaded grouped raw data."""
    full_despiked_no_sqrt_by_class = Preprocessing.fast_spike_filter_dict(
        raw_by_class,
        radius=3,
        transform="none",
        method="fast",
        n_sigmas=3.0,
        k=4.0,
        min_threshold=1000.0,
    )
    full_smoothed_no_sqrt_by_class = Preprocessing.apply_savgol_filter_dict(
        full_despiked_no_sqrt_by_class,
        window_length=31,
        polyorder=3,
        deriv=0,
        mode="mirror",
    )
    full_despiked_by_class = Preprocessing.fast_spike_filter_dict(
        raw_by_class,
        radius=3,
        transform="sqrt",
        method="fast",
        n_sigmas=3.0,
        k=4.0,
        min_threshold=1000.0,
    )
    full_smoothed_by_class = Preprocessing.apply_savgol_filter_dict(
        full_despiked_by_class,
        window_length=31,
        polyorder=3,
        deriv=0,
        mode="mirror",
    )

    sliced_raw_by_class = Preprocessing.slice_dict_signal_segments(
        raw_by_class,
        segments=signal_segments,
    )
    despiked_by_class = Preprocessing.fast_spike_filter_dict(
        sliced_raw_by_class,
        radius=3,
        transform="sqrt",
        method="fast",
        n_sigmas=3.0,
        k=4.0,
        min_threshold=1000.0,
    )
    smoothed_by_class = Preprocessing.apply_savgol_filter_dict(
        despiked_by_class,
        window_length=31,
        polyorder=3,
        deriv=0,
        mode="mirror",
    )
    processed_by_class = Preprocessing.downsample_dict_signals(
        smoothed_by_class,
        step=downsample_step,
        offset=0,
    )

    first_diff_by_class = Preprocessing.apply_savgol_filter_dict(
        Preprocessing.compute_central_diff_dict(processed_by_class),
        window_length=31,
        polyorder=3,
        deriv=0,
        mode="mirror",
    )
    second_diff_by_class = Preprocessing.apply_savgol_filter_dict(
        Preprocessing.compute_second_central_diff_dict(processed_by_class),
        window_length=31,
        polyorder=3,
        deriv=0,
        mode="mirror",
    )

    selected_value_types = ("processed_signal", "first_difference", "second_difference")
    x_all, y_all = Preprocessing.build_multi_channel_dataset(
        data_dict_map={
            selected_value_types[0]: processed_by_class,
            selected_value_types[1]: first_diff_by_class,
            selected_value_types[2]: second_diff_by_class,
        },
        selected_types=selected_value_types,
    )

    return DemoSignalData(
        data_path=data_path,
        label_col=label_col,
        class_order=tuple(class_order),
        raw_by_class=raw_by_class,
        full_smoothed_no_sqrt_by_class=full_smoothed_no_sqrt_by_class,
        full_smoothed_by_class=full_smoothed_by_class,
        sliced_raw_by_class=sliced_raw_by_class,
        despiked_by_class=despiked_by_class,
        smoothed_by_class=smoothed_by_class,
        processed_by_class=processed_by_class,
        first_diff_by_class=first_diff_by_class,
        second_diff_by_class=second_diff_by_class,
        x_all=x_all,
        y_all=y_all,
        selected_value_types=selected_value_types,
        removed_zero_sample_indices=removed_zero_sample_indices,
    )


def _resolve_class_order(
    observed_labels: Iterable[str],
    preferred_order: Sequence[str],
) -> tuple[str, ...]:
    observed = {str(label) for label in observed_labels}
    preferred = tuple(str(label) for label in preferred_order if str(label) in observed)
    extras = tuple(sorted(observed - set(preferred)))
    ordered = preferred + extras
    if not ordered:
        raise ValueError("No labelled samples were found in the data file.")
    return ordered
