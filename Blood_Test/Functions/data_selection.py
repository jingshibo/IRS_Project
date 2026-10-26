from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


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

