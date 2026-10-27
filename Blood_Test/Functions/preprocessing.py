from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import savgol_filter

from .data_loader import IM_IDX, RE_IDX


def _resolve_safe_window_length(
    signal_length: int,
    window_length: int,
    polyorder: int,
) -> int:
    safe_window = min(int(window_length), int(signal_length))
    if safe_window % 2 == 0:
        safe_window -= 1
    if safe_window <= polyorder:
        raise ValueError(
            "window_length must be greater than polyorder after adjustment, "
            f"got signal_length={signal_length}, window_length={window_length}, "
            f"polyorder={polyorder}"
        )
    if safe_window < 3:
        raise ValueError(f"window_length must resolve to at least 3, got {safe_window}")
    return safe_window


def apply_savgol_to_re_im_and_recompute(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    window_length: int = 31,
    polyorder: int = 3,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Smooth real/imaginary channels and recompute amplitude/phase.

    Expected blood-data block shape::

        [set, sample, repetition, frequency, channel]

    Filtering is applied only along the frequency axis. The output channel order
    remains [real, imaginary, amplitude, phase], and missing traces are preserved.
    """
    filtered_data: Dict[str, Dict[str, np.ndarray]] = {}

    for sample_type, concentration_map in data.items():
        filtered_data[sample_type] = {}
        for concentration, block in concentration_map.items():
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 5 or block.shape[-1] < 2:
                raise ValueError(
                    "Expected block with shape "
                    f"[set, sample, repetition, frequency, channel>=2], got {block.shape}"
                )

            safe_window = _resolve_safe_window_length(
                signal_length=block.shape[-2],
                window_length=window_length,
                polyorder=polyorder,
            )

            filtered_block = block.copy()
            valid_mask = ~np.isnan(block).all(axis=(-2, -1))
            valid_traces = block[valid_mask]

            if valid_traces.size > 0:
                filtered_re_im = savgol_filter(
                    valid_traces[..., [RE_IDX, IM_IDX]],
                    window_length=safe_window,
                    polyorder=polyorder,
                    axis=1,
                    mode="mirror",
                ).astype(np.float32, copy=False)

                filtered_re = filtered_re_im[..., 0]
                filtered_im = filtered_re_im[..., 1]
                filtered_amplitude = np.sqrt(
                    filtered_re**2 + filtered_im**2
                ).astype(np.float32, copy=False)
                filtered_phase = np.arctan2(
                    filtered_im, filtered_re
                ).astype(np.float32, copy=False)

                recomputed = np.stack(
                    (
                        filtered_re,
                        filtered_im,
                        filtered_amplitude,
                        filtered_phase,
                    ),
                    axis=-1,
                ).astype(np.float32, copy=False)
                filtered_block[valid_mask] = recomputed

            filtered_data[sample_type][concentration] = filtered_block

    return filtered_data


def apply_savgol_filter_nested(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    window_length: int = 31,
    polyorder: int = 3,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Smooth every channel along the frequency axis of each nested blood-data block.

    Expected block shape::

        [set, sample, repetition, frequency, channel]

    Fully missing traces are preserved.
    """
    filtered_data: Dict[str, Dict[str, np.ndarray]] = {}

    for sample_type, concentration_map in data.items():
        filtered_data[sample_type] = {}
        for concentration, block in concentration_map.items():
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 5:
                raise ValueError(
                    "Expected block with shape "
                    f"[set, sample, repetition, frequency, channel], got {block.shape}"
                )

            safe_window = _resolve_safe_window_length(
                signal_length=block.shape[-2],
                window_length=window_length,
                polyorder=polyorder,
            )

            filtered_block = block.copy()
            valid_mask = ~np.isnan(block).all(axis=(-2, -1))
            valid_traces = block[valid_mask]

            if valid_traces.size > 0:
                filtered_valid = savgol_filter(
                    valid_traces,
                    window_length=safe_window,
                    polyorder=polyorder,
                    axis=1,
                    mode="mirror",
                ).astype(np.float32, copy=False)
                filtered_block[valid_mask] = filtered_valid

            filtered_data[sample_type][concentration] = filtered_block

    return filtered_data


def downsample_nested_data(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    ratio: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Downsample each signal by keeping every ``ratio``-th frequency point.

    Expected block shape::

        [set, sample, repetition, frequency, channel]

    Only the frequency axis is downsampled. Sets, samples, repetitions, and
    channels are preserved.
    """
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")

    downsampled_data: Dict[str, Dict[str, np.ndarray]] = {}
    for sample_type, concentration_map in data.items():
        downsampled_data[sample_type] = {}
        for concentration, block in concentration_map.items():
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 5:
                raise ValueError(
                    "Expected block with shape "
                    f"[set, sample, repetition, frequency, channel], got {block.shape}"
                )

            downsampled_data[sample_type][concentration] = block[
                :, :, :, ::ratio, :
            ].copy()

    return downsampled_data
