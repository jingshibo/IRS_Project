from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict
import re

import numpy as np


FILENAME_PATTERN = re.compile(
    r"^(?P<liquid>[^_]+)_(?P<concentration>\d+-\d+)_(?P<sample>S\d+)_(?P<rep>REP\d+)\.csv$"
)
RE_IDX = 0
IM_IDX = 1
AMP_IDX = 2
PHASE_IDX = 3


def _parse_sample_index(sample_token: str) -> int:
    return int(sample_token[1:]) - 1


def _parse_rep_index(rep_token: str) -> int:
    return int(rep_token[3:]) - 1


def _compute_derived_channels(re_im_array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    re_values = re_im_array[:, RE_IDX]
    im_values = re_im_array[:, IM_IDX]
    amplitude = np.sqrt(re_values ** 2 + im_values ** 2, dtype=dtype)
    phase = np.arctan2(im_values, re_values).astype(dtype, copy=False)
    return np.column_stack((re_values, im_values, amplitude, phase)).astype(dtype, copy=False)


def _parse_liquid_file(file_path: Path, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    freq_values = []
    complex_rows = []

    with file_path.open("r", encoding="utf-8-sig") as handle:
        for line_idx, raw_line in enumerate(handle):
            if line_idx < 3:
                continue

            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(";") if part.strip()]
            if len(parts) < 3:
                raise ValueError(
                    f"{file_path.name}: expected at least 3 semicolon-separated values "
                    f"after the header, got {parts!r}"
                )

            freq_values.append(float(parts[0]))
            complex_rows.append((float(parts[1]), float(parts[2])))

    if not complex_rows:
        raise ValueError(f"{file_path.name}: no data rows found after the 3-line header")

    freq_array = np.asarray(freq_values, dtype=dtype)
    data_array = np.asarray(complex_rows, dtype=dtype)
    return freq_array, _compute_derived_channels(data_array, dtype=dtype)


def load_liquid_folder(
    folder_path: str | Path,
    *,
    dtype: np.dtype = np.float32,
    check_frequency_consistency: bool = True,
    require_complete_grid: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load Adam Wellcome CSV files into a nested dict:

        data[liquid][concentration] -> ndarray with shape [sample, rep, L, 4]

    The last axis stores [re, im, amplitude, phase]. Frequency values are
    validated for consistency but are not returned because they are shared
    across files.

    When a sample/rep combination is missing, the corresponding slice remains NaN
    unless require_complete_grid=True is requested.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    grouped_files: dict[str, dict[str, dict[tuple[int, int], Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for file_path in sorted(folder.glob("*.csv")):
        match = FILENAME_PATTERN.match(file_path.name)
        if not match:
            raise ValueError(f"Unexpected filename format: {file_path.name}")

        liquid = match.group("liquid")
        concentration = match.group("concentration")
        sample_idx = _parse_sample_index(match.group("sample"))
        rep_idx = _parse_rep_index(match.group("rep"))
        key = (sample_idx, rep_idx)

        if key in grouped_files[liquid][concentration]:
            raise ValueError(
                f"Duplicate sample/rep slot for {liquid}_{concentration}: {file_path.name}"
            )

        grouped_files[liquid][concentration][key] = file_path

    output: Dict[str, Dict[str, np.ndarray]] = {}

    for liquid, concentration_map in grouped_files.items():
        output[liquid] = {}
        for concentration, slot_map in concentration_map.items():
            max_sample = max(sample_idx for sample_idx, _ in slot_map) + 1
            max_rep = max(rep_idx for _, rep_idx in slot_map) + 1

            if require_complete_grid:
                expected_slots = {
                    (sample_idx, rep_idx)
                    for sample_idx in range(max_sample)
                    for rep_idx in range(max_rep)
                }
                missing_slots = sorted(expected_slots - set(slot_map))
                if missing_slots:
                    raise ValueError(
                        f"Missing files for {liquid}_{concentration} at sample/rep slots: "
                        f"{missing_slots}"
                    )

            first_slot = min(slot_map)
            reference_freq, reference_data = _parse_liquid_file(slot_map[first_slot], dtype=dtype)
            signal_length = reference_data.shape[0]

            stacked = np.full((max_sample, max_rep, signal_length, 4), np.nan, dtype=dtype)
            stacked[first_slot[0], first_slot[1]] = reference_data

            for slot, file_path in slot_map.items():
                if slot == first_slot:
                    continue

                freq_array, data_array = _parse_liquid_file(file_path, dtype=dtype)
                if data_array.shape != (signal_length, 4):
                    raise ValueError(
                        f"{file_path.name}: expected shape {(signal_length, 4)}, "
                        f"got {data_array.shape}"
                    )

                if check_frequency_consistency and not np.allclose(freq_array, reference_freq):
                    raise ValueError(
                        f"{file_path.name}: frequency axis does not match the reference "
                        f"for {liquid}_{concentration}"
                    )

                stacked[slot[0], slot[1]] = data_array

            output[liquid][concentration] = stacked

    return output
