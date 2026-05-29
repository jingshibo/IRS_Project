from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class RepeatedMeasurementDataset:
    x: np.ndarray
    y: np.ndarray
    group_ids: np.ndarray
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    measurement_keys: List[dict]
    sample_groups: List[dict]


@dataclass
class RepeatedMeasurementSplit:
    fold: int
    x_train: np.ndarray
    y_train: np.ndarray
    train_group_ids: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_group_ids: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    measurement_keys: List[dict]
    sample_groups: List[dict]
    train_mean: np.ndarray
    train_std: np.ndarray


def _make_label(liquid: str, concentration: str, label_mode: str) -> str:
    if label_mode == "joint":
        return f"{liquid}__{concentration}"
    if label_mode == "liquid":
        return liquid
    if label_mode == "concentration":
        return concentration
    raise ValueError(
        f"Unsupported label_mode='{label_mode}'. Use 'joint', 'liquid', or 'concentration'."
    )


def _concentration_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("-"))


def build_repeated_measurement_dataset(
    difference_data: Dict[str, Dict[str, np.ndarray]],
    *,
    channel_indices: Sequence[int] = (4, 5, 6, 7),
    label_mode: str = "joint",
    include_reference: bool = False,
    reference_liquid: str = "PBS",
    reference_concentration: str = "10-0",
) -> RepeatedMeasurementDataset:
    if len(channel_indices) == 0:
        raise ValueError("channel_indices must not be empty")

    x_list: List[np.ndarray] = []
    label_list: List[str] = []
    group_ids: List[int] = []
    measurement_keys: List[dict] = []
    sample_groups: List[dict] = []
    group_lookup: Dict[tuple[str, str, int], int] = {}

    for liquid in sorted(difference_data):
        for concentration in sorted(difference_data[liquid], key=_concentration_sort_key):
            if not include_reference and liquid == reference_liquid and concentration == reference_concentration:
                continue

            block = np.asarray(difference_data[liquid][concentration], dtype=np.float32)
            selected = block[..., channel_indices]
            selected = np.moveaxis(selected, -1, -2)  # [sample, rep, C, L]
            label = _make_label(liquid, concentration, label_mode=label_mode)

            for sample_idx in range(selected.shape[0]):
                group_key = (liquid, concentration, sample_idx + 1)

                for rep_idx in range(selected.shape[1]):
                    trace = selected[sample_idx, rep_idx]
                    if np.isnan(trace).all():
                        continue
                    if np.isnan(trace).any():
                        raise ValueError(
                            f"Partial NaN trace found for {liquid}_{concentration} "
                            f"S{sample_idx + 1}_REP{rep_idx + 1}"
                        )

                    if group_key not in group_lookup:
                        group_lookup[group_key] = len(sample_groups)
                        sample_groups.append(
                            {
                                "liquid": liquid,
                                "concentration": concentration,
                                "sample": sample_idx + 1,
                                "label": label,
                            }
                        )
                    group_id = group_lookup[group_key]

                    x_list.append(trace.astype(np.float32, copy=False))
                    label_list.append(label)
                    group_ids.append(group_id)
                    measurement_keys.append(
                        {
                            "liquid": liquid,
                            "concentration": concentration,
                            "sample": sample_idx + 1,
                            "rep": rep_idx + 1,
                            "group_id": group_id,
                        }
                    )

    if not x_list:
        raise ValueError("No valid traces found for repeated-measurement dataset")

    unique_labels = sorted(set(label_list))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    y = np.asarray([label_to_idx[label] for label in label_list], dtype=np.int64)
    for group in sample_groups:
        group["label_idx"] = label_to_idx[group["label"]]

    return RepeatedMeasurementDataset(
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
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)


def _normalize_signals(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32, copy=False)


def _build_split(
    dataset: RepeatedMeasurementDataset,
    *,
    fold: int,
    train_group_set: set[int],
    test_group_set: set[int],
) -> RepeatedMeasurementSplit:
    train_mask = np.asarray([group_id in train_group_set for group_id in dataset.group_ids])
    test_mask = np.asarray([group_id in test_group_set for group_id in dataset.group_ids])
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)

    if train_indices.size == 0:
        raise ValueError(f"Fold {fold}: training split is empty")
    if test_indices.size == 0:
        raise ValueError(f"Fold {fold}: test split is empty")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError(f"Fold {fold}: train/test measurement indices overlap")

    x_train = dataset.x[train_indices]
    train_mean, train_std = _compute_channel_stats(x_train)

    return RepeatedMeasurementSplit(
        fold=fold,
        x_train=_normalize_signals(x_train, train_mean, train_std),
        y_train=dataset.y[train_indices],
        train_group_ids=dataset.group_ids[train_indices],
        x_test=_normalize_signals(dataset.x[test_indices], train_mean, train_std),
        y_test=dataset.y[test_indices],
        test_group_ids=dataset.group_ids[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        label_to_idx=dataset.label_to_idx,
        idx_to_label=dataset.idx_to_label,
        measurement_keys=dataset.measurement_keys,
        sample_groups=dataset.sample_groups,
        train_mean=train_mean,
        train_std=train_std,
    )


def build_leave_one_sample_out_splits(
    dataset: RepeatedMeasurementDataset,
) -> List[RepeatedMeasurementSplit]:
    """
    Build one fold per independent sample index inside each liquid-concentration block.

    For the Adam Wellcome layout this creates 3 folds. In fold 0, all S1 groups
    are tested; in fold 1, all S2 groups are tested; and in fold 2, all S3
    groups are tested. All repeated measurements from a sample group stay in the
    same split.
    """
    block_to_groups: Dict[tuple[str, str], List[int]] = {}
    for group_id, group in enumerate(dataset.sample_groups):
        block_key = (group["liquid"], group["concentration"])
        block_to_groups.setdefault(block_key, []).append(group_id)

    for group_ids in block_to_groups.values():
        group_ids.sort(key=lambda group_id: int(dataset.sample_groups[group_id]["sample"]))

    fold_counts = {len(group_ids) for group_ids in block_to_groups.values()}
    if len(fold_counts) != 1:
        raise ValueError(
            "All liquid-concentration blocks must have the same number of independent samples; "
            f"found counts={sorted(fold_counts)}"
        )

    n_folds = fold_counts.pop()
    if n_folds < 2:
        raise ValueError(f"Need at least 2 independent samples per block, got {n_folds}")

    all_group_ids = set(range(len(dataset.sample_groups)))
    splits: List[RepeatedMeasurementSplit] = []
    for fold in range(n_folds):
        test_group_set = {group_ids[fold] for group_ids in block_to_groups.values()}
        train_group_set = all_group_ids - test_group_set
        splits.append(
            _build_split(
                dataset,
                fold=fold,
                train_group_set=train_group_set,
                test_group_set=test_group_set,
            )
        )

    return splits
