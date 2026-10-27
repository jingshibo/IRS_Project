from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


WSL_DATA_ROOT = Path("/home/shibojing/data/Endpoint_Old")
WINDOWS_WSL_DATA_ROOT = Path(r"\\wsl.localhost\Ubuntu\home\shibojing\data\Endpoint_Old")
DATA_ROOT = WSL_DATA_ROOT if WSL_DATA_ROOT.exists() else WINDOWS_WSL_DATA_ROOT
DATA_SUFFIXES = {".csv", ".xls", ".xlsx"}
SENSOR_PATTERN = re.compile(r"lf(?P<sensor_number>\d)", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"no(?P<subject_number>\d+)", re.IGNORECASE)
NIGHT_PATTERN = re.compile(r"night\s*(?P<night_number>\d+)", re.IGNORECASE)


def iter_data_files(data_root: Path = DATA_ROOT):
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.name.endswith(":Zone.Identifier"):
            continue
        if path.suffix.lower() in DATA_SUFFIXES:
            yield path


def infer_label(path: Path, data_root: Path = DATA_ROOT) -> str:
    folder_name = path.parent.name.lower().replace("-", "_").replace(" ", "_")
    file_name = path.stem.lower()

    if "all_positive" in folder_name:
        return "POSITIVE"
    if "all_negative" in folder_name or "all__negative" in folder_name:
        return "NEGATIVE"
    if "pos" in file_name:
        return "POSITIVE"
    if "neg" in file_name or "neb" in file_name:
        return "NEGATIVE"

    relative_path = path.relative_to(data_root)
    raise ValueError(f"Cannot infer label from {relative_path}")


def parse_file_metadata(path: Path) -> dict[str, int]:
    sensor_match = SENSOR_PATTERN.search(path.name)
    subject_match = SUBJECT_PATTERN.search(path.name)

    if sensor_match is None:
        raise ValueError(f"Cannot infer sensor number from {path.name}")
    if subject_match is None:
        raise ValueError(f"Cannot infer subject number from {path.name}")

    return {
        "sensor_number": int(sensor_match.group("sensor_number")),
        "subject_number": int(subject_match.group("subject_number")),
    }


def parse_night_number(path: Path) -> int:
    night_match = NIGHT_PATTERN.search(path.parent.name)

    if night_match is None:
        raise ValueError(f"Cannot infer night number from {path.parent.name}")

    return int(night_match.group("night_number"))


def get_subject_info(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = ["label", "source_file", "subject_number", "night", "sensor_number"]
    subject_metadata = (
        dataset[metadata_columns]
        .drop_duplicates()
        .sort_values(["label", "subject_number", "night", "sensor_number"], ignore_index=True)
    )
    subject_metadata["subject_file_count"] = subject_metadata.groupby("subject_number")[
        "source_file"
    ].transform("count")
    sample_counts = (
        dataset.groupby(["subject_number", "night"])
        .size()
        .rename("subject_night_sample_count")
        .reset_index()
    )
    subject_metadata = subject_metadata.merge(
        sample_counts,
        on=["subject_number", "night"],
        how="left",
    )
    repeated_subject_metadata = subject_metadata[
        subject_metadata["subject_file_count"] > 1
    ].reset_index(drop=True)
    return subject_metadata, repeated_subject_metadata


def read_endpoint_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=None, names=["frequency", "amplitude"])
    else:
        df = pd.read_excel(path, header=None, names=["frequency", "amplitude"])

    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")
    df["amplitude"] = pd.to_numeric(df["amplitude"], errors="coerce")
    df = df.dropna(subset=["frequency", "amplitude"])
    return build_sample_rows(df, path)


def build_sample_rows(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    first_frequency = df.loc[0, "frequency"]
    period_starts = df.index[df["frequency"] == first_frequency].tolist()
    period_length = period_starts[1] if len(period_starts) > 1 else len(df)

    if len(df) % period_length != 0:
        raise ValueError(f"Frequency rows do not split evenly into samples for {path.name}")

    frequency_columns = df["frequency"].iloc[:period_length].reset_index(drop=True)
    if not frequency_columns.is_unique:
        raise ValueError(f"Frequency values are not unique within one sample for {path.name}")

    sample_count = len(df) // period_length
    expected_frequencies = np.tile(frequency_columns.to_numpy(), sample_count)
    if not np.array_equal(df["frequency"].to_numpy(), expected_frequencies):
        raise ValueError(f"Frequency periods are not consistent for {path.name}")

    amplitude_rows = df["amplitude"].to_numpy().reshape(sample_count, period_length)
    sample_rows = pd.DataFrame(amplitude_rows, columns=frequency_columns.to_numpy())
    sample_rows.insert(0, "sample_time", range(1, sample_count + 1))
    return sample_rows


def load_endpoint_dataset(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    rows = []
    for path in iter_data_files(data_root):
        file_data = read_endpoint_data(path)
        file_metadata = parse_file_metadata(path)
        file_data["label"] = infer_label(path, data_root)
        file_data["sensor_number"] = file_metadata["sensor_number"]
        file_data["subject_number"] = file_metadata["subject_number"]
        file_data["night"] = parse_night_number(path)
        file_data["source_file"] = path.name
        rows.append(file_data)

    if not rows:
        raise FileNotFoundError(f"No endpoint data files found under {data_root}")

    dataset = pd.concat(rows, ignore_index=True)
    metadata_columns = ["label", "source_file", "night", "sensor_number", "subject_number", "sample_time"]
    frequency_columns = [column for column in dataset.columns if column not in metadata_columns]
    return dataset[metadata_columns + frequency_columns]
