from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def exclude_frequency_bands(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    excluded_bands: Sequence[tuple[int, int]],
    original_frequency_points: int,
    downsample_ratio: int = 1,
) -> tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    """Remove bands from an already downsampled nested dataset.

    Bands use zero-based indices of the ORIGINAL signal, with both endpoints
    included. For example, (8800, 9000) removes original indices 8800, 8900,
    and 9000 when downsample_ratio=100. An empty sequence keeps every point.
    The input must still contain the complete, unmasked downsampled grid.

    Apply after all smoothing and PBS correction, before dataset construction.
    Every block must have shape [set, sample, repetition, frequency, channel].
    The same mask is applied to all blocks and channels, without changing input
    arrays or the original downsampling positions. Missing traces stay missing.

    Returns the selected data and its retained ORIGINAL frequency indices, so
    frequency coordinates remain available despite gaps in the model input.
    Overlapping bands are combined. Invalid ranges and removing all retained
    points raise ValueError.
    """
    for name, value in (
        ("original_frequency_points", original_frequency_points),
        ("downsample_ratio", downsample_ratio),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")

    original_indices = np.arange(0, original_frequency_points, downsample_ratio)
    keep = np.ones(original_indices.size, dtype=bool)
    for band in excluded_bands:
        if len(band) != 2:
            raise ValueError(f"Each band must contain (start, end), got {band!r}")
        start, end = band
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in (start, end)
        ):
            raise ValueError(f"Band endpoints must be integer indices, got {band!r}")
        if not 0 <= start <= end < original_frequency_points:
            raise ValueError(
                f"Invalid band {band!r}: require 0 <= start <= end < "
                f"{original_frequency_points}"
            )
        keep &= ~((original_indices >= start) & (original_indices <= end))

    if not keep.any():
        raise ValueError("Frequency exclusions remove every retained frequency point")

    selected_data: Dict[str, Dict[str, np.ndarray]] = {}
    for sample_type, concentration_map in data.items():
        selected_data[sample_type] = {}
        for concentration, block in concentration_map.items():
            block = np.asarray(block)
            if block.ndim != 5 or block.shape[-2] != original_indices.size:
                raise ValueError(
                    f"{sample_type} {concentration}: expected shape "
                    "[set, sample, repetition, frequency, channel] with "
                    f"{original_indices.size} frequency points on the complete "
                    f"downsampled grid, got {block.shape}"
                )
            selected_data[sample_type][concentration] = block[..., keep, :].copy()

    return selected_data, original_indices[keep]


def exclude_data_blocks(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    excluded_blocks: Sequence[tuple[str, int, str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Exclude complete set/concentration blocks without changing the source files.

    Each exclusion is specified as::

        (sample_type, one_based_set_number, concentration)

    For example, ``("SAU", 2, "10-6")`` excludes every sample and repetition
    from SAU set 2 at concentration 10^-6. Excluded traces are represented by
    NaN so their original set indices remain intact and downstream dataset
    construction can skip them safely.
    """
    selected_data: Dict[str, Dict[str, np.ndarray]] = {
        sample_type: dict(concentration_map)
        for sample_type, concentration_map in data.items()
    }
    copied_blocks: set[tuple[str, str]] = set()

    for sample_type, set_number, concentration in excluded_blocks:
        if sample_type not in selected_data:
            raise KeyError(f"Sample type {sample_type!r} not found in the data")
        if concentration not in selected_data[sample_type]:
            raise KeyError(
                f"Concentration {concentration!r} not found for {sample_type!r}"
            )
        if set_number < 1:
            raise ValueError(f"Set numbers must start at 1, got {set_number}")

        block_key = (sample_type, concentration)
        block = np.asarray(selected_data[sample_type][concentration])
        set_index = set_number - 1
        if set_index >= block.shape[0]:
            raise IndexError(
                f"Set {set_number} is invalid for {sample_type} {concentration}; "
                f"available sets are 1-{block.shape[0]}"
            )

        if block_key not in copied_blocks:
            block = block.copy()
            selected_data[sample_type][concentration] = block
            copied_blocks.add(block_key)
        else:
            block = selected_data[sample_type][concentration]

        block[set_index] = np.nan

    return selected_data

