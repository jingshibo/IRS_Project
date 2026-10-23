from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class BloodConcentrationDataset:
    x: np.ndarray
    y: np.ndarray
    group_ids: np.ndarray
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    measurement_keys: List[dict]
    sample_groups: List[dict]


@dataclass
class BloodConcentrationSplit:
    fold: int
    validation_sample: int
    x_train: np.ndarray
    y_train: np.ndarray
    train_group_ids: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    validation_group_ids: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    measurement_keys: List[dict]
    sample_groups: List[dict]
    train_mean: np.ndarray
    train_std: np.ndarray


def _concentration_sort_key(value: str) -> int:
    try:
        base, exponent = value.split("-", maxsplit=1)
        if base != "10":
            raise ValueError
        return int(exponent)
    except ValueError as error:
        raise ValueError(
            f"Expected concentration label in the form '10-N', got {value!r}"
        ) from error


def build_concentration_dataset(
    difference_data: Dict[str, Dict[str, np.ndarray]],
    *,
    channel_indices: Sequence[int] = (9,),
    sample_type: str = "SAU",
) -> BloodConcentrationDataset:
    """
    Build the three-class SAU concentration dataset.

    Input blocks have shape::

        [set, sample, repetition, frequency, channel]

    Output measurements have shape::

        [measurement, selected_channel, frequency]

    A sample group is uniquely identified by ``(set, concentration, sample)``.
    Consequently, all repeated measurements from one sample share one group ID.
    PBS is excluded because only ``sample_type`` is read from the input data.
    """
    if len(channel_indices) == 0:
        raise ValueError("channel_indices must not be empty")
    if sample_type not in difference_data:
        raise KeyError(f"Sample type {sample_type!r} not found in the data")

    x_list: List[np.ndarray] = []
    label_list: List[str] = []
    group_ids: List[int] = []
    measurement_keys: List[dict] = []
    sample_groups: List[dict] = []
    group_lookup: Dict[tuple[int, str, int], int] = {}

    concentrations = sorted(
        difference_data[sample_type],
        key=_concentration_sort_key,
    )
    for concentration in concentrations:
        block = np.asarray(
            difference_data[sample_type][concentration],
            dtype=np.float32,
        )
        if block.ndim != 5:
            raise ValueError(
                "Expected block with shape "
                f"[set, sample, repetition, frequency, channel], got {block.shape}"
            )

        invalid_channels = [
            channel_index
            for channel_index in channel_indices
            if channel_index < 0 or channel_index >= block.shape[-1]
        ]
        if invalid_channels:
            raise IndexError(
                f"Channel indices {invalid_channels} are invalid for {sample_type} "
                f"{concentration} with {block.shape[-1]} channels"
            )

        selected = block[..., list(channel_indices)]
        selected = np.moveaxis(
            selected,
            -1,
            -2,
        )  # [set, sample, repetition, selected_channel, frequency]

        for set_index in range(selected.shape[0]):
            for sample_index in range(selected.shape[1]):
                group_key = (set_index + 1, concentration, sample_index + 1)

                for repetition_index in range(selected.shape[2]):
                    trace = selected[set_index, sample_index, repetition_index]
                    if np.isnan(trace).all():
                        continue
                    if np.isnan(trace).any():
                        raise ValueError(
                            f"Partial NaN trace found for {sample_type}-{set_index + 1}-"
                            f"{concentration}-S{sample_index + 1}-REP{repetition_index + 1}"
                        )

                    if group_key not in group_lookup:
                        group_lookup[group_key] = len(sample_groups)
                        sample_groups.append(
                            {
                                "sample_type": sample_type,
                                "set": set_index + 1,
                                "concentration": concentration,
                                "sample": sample_index + 1,
                                "label": concentration,
                            }
                        )
                    group_id = group_lookup[group_key]

                    x_list.append(trace.astype(np.float32, copy=False))
                    label_list.append(concentration)
                    group_ids.append(group_id)
                    measurement_keys.append(
                        {
                            "sample_type": sample_type,
                            "set": set_index + 1,
                            "concentration": concentration,
                            "sample": sample_index + 1,
                            "repetition": repetition_index + 1,
                            "group_id": group_id,
                        }
                    )

    if not x_list:
        raise ValueError(f"No valid {sample_type} measurements were found")

    unique_labels = sorted(set(label_list), key=_concentration_sort_key)
    label_to_idx = {label: index for index, label in enumerate(unique_labels)}
    idx_to_label = {index: label for label, index in label_to_idx.items()}
    y = np.asarray([label_to_idx[label] for label in label_list], dtype=np.int64)

    for group in sample_groups:
        group["label_idx"] = label_to_idx[group["label"]]

    return BloodConcentrationDataset(
        x=np.stack(x_list, axis=0).astype(np.float32, copy=False),
        y=y,
        group_ids=np.asarray(group_ids, dtype=np.int64),
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        measurement_keys=measurement_keys,
        sample_groups=sample_groups,
    )


def _compute_channel_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        mean.astype(np.float32, copy=False),
        std.astype(np.float32, copy=False),
    )


def _normalize_signals(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32, copy=False)


def _measurement_indices_for_groups(
    dataset: BloodConcentrationDataset,
    selected_group_ids: set[int],
) -> np.ndarray:
    mask = np.asarray(
        [group_id in selected_group_ids for group_id in dataset.group_ids],
        dtype=bool,
    )
    return np.flatnonzero(mask)


def build_leave_one_sample_out_splits(
    dataset: BloodConcentrationDataset,
) -> List[BloodConcentrationSplit]:
    """
    Build balanced folds using one sample index for validation.

    With eight samples per set/concentration:

    - Fold 1 validates on S1 and trains on S2-S8.
    - Fold 2 validates on S2 and trains on all remaining samples.
    - This continues until every sample index has been used for validation.

    Every set and concentration appears in both partitions, while all repetitions
    belonging to a sample group remain in the same partition.
    """
    sample_indices = sorted(
        {int(group["sample"]) for group in dataset.sample_groups}
    )
    if len(sample_indices) < 2:
        raise ValueError(
            "At least two sample indices are required for train/validation, "
            f"got {sample_indices}"
        )

    block_to_samples: Dict[tuple[int, str], set[int]] = {}
    for group in dataset.sample_groups:
        block_key = (int(group["set"]), str(group["concentration"]))
        block_to_samples.setdefault(block_key, set()).add(int(group["sample"]))

    expected_samples = set(sample_indices)
    incomplete_blocks = {
        block_key: sorted(samples)
        for block_key, samples in block_to_samples.items()
        if samples != expected_samples
    }
    if incomplete_blocks:
        raise ValueError(
            "Every set/concentration block must contain the same sample indices; "
            f"found incomplete blocks={incomplete_blocks}"
        )

    splits: List[BloodConcentrationSplit] = []
    all_group_ids = set(range(len(dataset.sample_groups)))

    for fold, validation_sample in enumerate(sample_indices):
        validation_group_ids = {
            group_id
            for group_id, group in enumerate(dataset.sample_groups)
            if int(group["sample"]) == validation_sample
        }
        train_group_ids = all_group_ids - validation_group_ids

        if train_group_ids & validation_group_ids:
            raise ValueError(f"Fold {fold + 1}: train/validation groups overlap")

        train_indices = _measurement_indices_for_groups(dataset, train_group_ids)
        validation_indices = _measurement_indices_for_groups(
            dataset,
            validation_group_ids,
        )
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(f"Fold {fold + 1}: one or more partitions are empty")

        train_mean, train_std = _compute_channel_stats(dataset.x[train_indices])
        splits.append(
            BloodConcentrationSplit(
                fold=fold,
                validation_sample=validation_sample,
                x_train=_normalize_signals(
                    dataset.x[train_indices], train_mean, train_std
                ),
                y_train=dataset.y[train_indices],
                train_group_ids=dataset.group_ids[train_indices],
                x_validation=_normalize_signals(
                    dataset.x[validation_indices], train_mean, train_std
                ),
                y_validation=dataset.y[validation_indices],
                validation_group_ids=dataset.group_ids[validation_indices],
                train_indices=train_indices,
                validation_indices=validation_indices,
                label_to_idx=dataset.label_to_idx,
                idx_to_label=dataset.idx_to_label,
                measurement_keys=dataset.measurement_keys,
                sample_groups=dataset.sample_groups,
                train_mean=train_mean,
                train_std=train_std,
            )
        )

    return splits
