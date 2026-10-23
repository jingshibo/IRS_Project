from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict
import re

import numpy as np


SAU_FILENAME_PATTERN = re.compile(
    r"^SAU-(?P<set>\d+)-10-(?P<exponent>\d+)-S(?P<sample>\d+)-REP(?P<rep>\d+)\.csv$"
)
PBS_FILENAME_PATTERN = re.compile(
    r"^PBS-(?P<set>\d+)-S(?P<sample>\d+)-REP(?P<rep>\d+)\.csv$"
)

RE_IDX = 0
IM_IDX = 1
AMP_IDX = 2
PHASE_IDX = 3


def _parse_one_based_index(value: str) -> int:
    """Convert the one-based indices in filenames to zero-based array indices."""
    index = int(value)
    if index < 1:
        raise ValueError(f"Filename indices must start at 1, got {value!r}")
    return index - 1


def _parse_blood_filename(file_path: Path) -> tuple[str, str, tuple[int, int, int]]:
    """Return (sample type, concentration, (set, sample, repetition))."""
    sau_match = SAU_FILENAME_PATTERN.match(file_path.name)
    if sau_match:
        concentration = f"10-{sau_match.group('exponent')}"
        slot = (
            _parse_one_based_index(sau_match.group("set")),
            _parse_one_based_index(sau_match.group("sample")),
            _parse_one_based_index(sau_match.group("rep")),
        )
        return "SAU", concentration, slot

    pbs_match = PBS_FILENAME_PATTERN.match(file_path.name)
    if pbs_match:
        slot = (
            _parse_one_based_index(pbs_match.group("set")),
            _parse_one_based_index(pbs_match.group("sample")),
            _parse_one_based_index(pbs_match.group("rep")),
        )
        return "PBS", "reference", slot

    raise ValueError(f"Unexpected filename format: {file_path.name}")


def _compute_derived_channels(re_im_array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    re_values = re_im_array[:, RE_IDX]
    im_values = re_im_array[:, IM_IDX]
    amplitude = np.sqrt(re_values**2 + im_values**2).astype(dtype, copy=False)
    phase = np.arctan2(im_values, re_values).astype(dtype, copy=False)
    return np.column_stack((re_values, im_values, amplitude, phase)).astype(dtype, copy=False)


def _parse_vna_file(file_path: Path, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    frequency_values = []
    complex_rows = []

    with file_path.open("r", encoding="utf-8-sig") as handle:
        for line_index, raw_line in enumerate(handle):
            if line_index < 3:
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

            frequency_values.append(float(parts[0]))
            complex_rows.append((float(parts[1]), float(parts[2])))

    if not complex_rows:
        raise ValueError(f"{file_path.name}: no data rows found after the 3-line header")

    frequency_array = np.asarray(frequency_values, dtype=dtype)
    re_im_array = np.asarray(complex_rows, dtype=dtype)
    return frequency_array, _compute_derived_channels(re_im_array, dtype=dtype)


def load_blood_folder(
    folder_path: str | Path,
    *,
    dtype: np.dtype = np.float32,
    check_frequency_consistency: bool = True,
    require_complete_grid: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load blood VNA files into nested arrays.

    The output layout is::

        data["SAU"][concentration] -> [set, sample, repetition, frequency, channel]
        data["PBS"]["reference"]  -> [set, sample, repetition, frequency, channel]

    The channel axis contains [real, imaginary, amplitude, phase]. Missing slots
    remain NaN unless ``require_complete_grid=True``. Frequency values are checked
    across all files but are not returned because they are shared by every signal.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    grouped_files: dict[
        str, dict[str, dict[tuple[int, int, int], Path]]
    ] = defaultdict(lambda: defaultdict(dict))

    for file_path in csv_files:
        sample_type, concentration, slot = _parse_blood_filename(file_path)
        if slot in grouped_files[sample_type][concentration]:
            raise ValueError(
                f"Duplicate set/sample/repetition slot for {sample_type} "
                f"{concentration}: {file_path.name}"
            )
        grouped_files[sample_type][concentration][slot] = file_path

    output: Dict[str, Dict[str, np.ndarray]] = {}
    global_reference_frequency: np.ndarray | None = None

    for sample_type in sorted(grouped_files):
        output[sample_type] = {}
        for concentration in sorted(grouped_files[sample_type]):
            slot_map = grouped_files[sample_type][concentration]
            number_of_sets = max(slot[0] for slot in slot_map) + 1
            number_of_samples = max(slot[1] for slot in slot_map) + 1
            number_of_repetitions = max(slot[2] for slot in slot_map) + 1

            if require_complete_grid:
                expected_slots = {
                    (set_index, sample_index, repetition_index)
                    for set_index in range(number_of_sets)
                    for sample_index in range(number_of_samples)
                    for repetition_index in range(number_of_repetitions)
                }
                missing_slots = sorted(expected_slots - set(slot_map))
                if missing_slots:
                    readable_slots = [
                        (set_index + 1, sample_index + 1, repetition_index + 1)
                        for set_index, sample_index, repetition_index in missing_slots
                    ]
                    raise ValueError(
                        f"Missing files for {sample_type} {concentration} at "
                        f"(set, sample, repetition) slots: {readable_slots}"
                    )

            first_slot = min(slot_map)
            first_frequency, first_data = _parse_vna_file(
                slot_map[first_slot], dtype=dtype
            )
            signal_length = first_data.shape[0]

            if global_reference_frequency is None:
                global_reference_frequency = first_frequency
            elif check_frequency_consistency and not np.allclose(
                first_frequency, global_reference_frequency
            ):
                raise ValueError(
                    f"{slot_map[first_slot].name}: frequency axis does not match "
                    "the global reference axis"
                )

            stacked = np.full(
                (
                    number_of_sets,
                    number_of_samples,
                    number_of_repetitions,
                    signal_length,
                    4,
                ),
                np.nan,
                dtype=dtype,
            )
            stacked[first_slot] = first_data

            for slot, file_path in slot_map.items():
                if slot == first_slot:
                    continue

                frequency_array, data_array = _parse_vna_file(file_path, dtype=dtype)
                if data_array.shape != (signal_length, 4):
                    raise ValueError(
                        f"{file_path.name}: expected shape {(signal_length, 4)}, "
                        f"got {data_array.shape}"
                    )

                if check_frequency_consistency and not np.allclose(
                    frequency_array, global_reference_frequency
                ):
                    raise ValueError(
                        f"{file_path.name}: frequency axis does not match the "
                        "global reference axis"
                    )

                stacked[slot] = data_array

            output[sample_type][concentration] = stacked

    return output


def print_blood_data_summary(data: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Print the loaded array shapes and measurement counts."""
    total_measurements = 0
    print("Loaded blood VNA data:")
    for sample_type in sorted(data):
        for concentration in sorted(data[sample_type]):
            values = data[sample_type][concentration]
            measurement_count = int(np.sum(~np.isnan(values[..., RE_IDX]).all(axis=-1)))
            total_measurements += measurement_count
            print(
                f"  {sample_type:3s} {concentration:9s}: "
                f"shape={values.shape}, measurements={measurement_count}"
            )
    print(f"Total measurements: {total_measurements}")

