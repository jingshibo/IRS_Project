from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import savgol_filter


def _resolve_safe_window_length(signal_length: int, window_length: int, polyorder: int) -> int:
    safe_window = min(int(window_length), int(signal_length))
    if safe_window % 2 == 0:
        safe_window -= 1
    if safe_window <= polyorder:
        raise ValueError(
            f"window_length must be greater than polyorder after adjustment, "
            f"got signal_length={signal_length}, window_length={window_length}, polyorder={polyorder}"
        )
    if safe_window < 3:
        raise ValueError(f"window_length must resolve to at least 3, got {safe_window}")
    return safe_window


def apply_savgol_filter_nested(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    window_length: int = 31,
    polyorder: int = 3,
    axis: int = 2,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Apply Savitzky-Golay smoothing to nested signal blocks.

    Expected block shape:
        [sample, rep, L, C]

    Smoothing is applied along the signal-length axis by default.
    Fully missing sample/rep slots (all-NaN) are preserved.
    """
    filtered_data: Dict[str, Dict[str, np.ndarray]] = {}

    for liquid, concentration_map in data.items():
        filtered_data[liquid] = {}
        for concentration, block in concentration_map.items():
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 4:
                raise ValueError(
                    f"Expected block with 4 dimensions [sample, rep, L, C], got shape {block.shape}"
                )

            safe_window = _resolve_safe_window_length(
                signal_length=block.shape[axis],
                window_length=window_length,
                polyorder=polyorder,
            )

            filtered_block = block.copy()
            valid_mask = ~np.isnan(block).all(axis=(2, 3))
            valid_traces = block[valid_mask]

            if valid_traces.size > 0:
                filtered_valid = savgol_filter(
                    valid_traces,
                    window_length=safe_window,
                    polyorder=polyorder,
                    axis=axis - 2,
                    mode="mirror",
                ).astype(np.float32, copy=False)
                filtered_block[valid_mask] = filtered_valid

            filtered_data[liquid][concentration] = filtered_block

    return filtered_data


def downsample_nested_data(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    ratio: int,
    axis: int = 2,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Downsample nested signal blocks by keeping every `ratio`-th point.

    Expected block shape:
        [sample, rep, L, C]

    Downsampling is applied along the signal-length axis by default.
    """
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")

    downsampled_data: Dict[str, Dict[str, np.ndarray]] = {}

    slicer = [slice(None)] * 4
    slicer[axis] = slice(None, None, ratio)
    slicer_tuple = tuple(slicer)

    for liquid, concentration_map in data.items():
        downsampled_data[liquid] = {}
        for concentration, block in concentration_map.items():
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 4:
                raise ValueError(
                    f"Expected block with 4 dimensions [sample, rep, L, C], got shape {block.shape}"
                )
            downsampled_data[liquid][concentration] = block[slicer_tuple].copy()

    return downsampled_data
