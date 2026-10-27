from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class DatasetBundle:
    x: np.ndarray
    y: np.ndarray
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    sample_keys: List[dict]


@dataclass
class SplitBundle:
    fold: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    sample_keys: List[dict]
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


def build_classification_dataset(
    difference_data: Dict[str, Dict[str, np.ndarray]],
    *,
    channel_indices: Sequence[int] = (4, 5, 6, 7),
    label_mode: str = "joint",
    include_reference: bool = False,
    reference_liquid: str = "PBS",
    reference_concentration: str = "10-0",
) -> DatasetBundle:
    if len(channel_indices) == 0:
        raise ValueError("channel_indices must not be empty")

    x_list: List[np.ndarray] = []
    label_list: List[str] = []
    sample_keys: List[dict] = []

    for liquid in sorted(difference_data):
        for concentration in sorted(
            difference_data[liquid],
            key=lambda value: tuple(int(part) for part in value.split("-")),
        ):
            if not include_reference and liquid == reference_liquid and concentration == reference_concentration:
                continue

            block = np.asarray(difference_data[liquid][concentration], dtype=np.float32)
            selected = block[..., channel_indices]
            selected = np.moveaxis(selected, -1, -2)  # [sample, rep, C, L]

            for sample_idx in range(selected.shape[0]):
                for rep_idx in range(selected.shape[1]):
                    trace = selected[sample_idx, rep_idx]
                    if np.isnan(trace).all():
                        continue
                    if np.isnan(trace).any():
                        raise ValueError(
                            f"Partial NaN trace found for {liquid}_{concentration} "
                            f"S{sample_idx + 1}_REP{rep_idx + 1}"
                        )

                    x_list.append(trace.astype(np.float32, copy=False))
                    label_list.append(_make_label(liquid, concentration, label_mode=label_mode))
                    sample_keys.append(
                        {
                            "liquid": liquid,
                            "concentration": concentration,
                            "sample": sample_idx + 1,
                            "rep": rep_idx + 1,
                        }
                    )

    if not x_list:
        raise ValueError("No valid traces found for classification dataset")

    unique_labels = sorted(set(label_list))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y = np.asarray([label_to_idx[label] for label in label_list], dtype=np.int64)
    x = np.stack(x_list, axis=0).astype(np.float32, copy=False)

    return DatasetBundle(
        x=x,
        y=y,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        sample_keys=sample_keys,
    )


def _compute_channel_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)


def _normalize_signals(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32, copy=False)


def _build_split_bundle(
    dataset: DatasetBundle,
    *,
    fold: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> SplitBundle:
    x_train = dataset.x[train_indices]
    y_train = dataset.y[train_indices]
    x_val = dataset.x[val_indices]
    y_val = dataset.y[val_indices]
    x_test = dataset.x[test_indices]
    y_test = dataset.y[test_indices]

    train_mean, train_std = _compute_channel_stats(x_train)
    x_train = _normalize_signals(x_train, train_mean, train_std)
    x_val = _normalize_signals(x_val, train_mean, train_std)
    x_test = _normalize_signals(x_test, train_mean, train_std)

    return SplitBundle(
        fold=fold,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        label_to_idx=dataset.label_to_idx,
        idx_to_label=dataset.idx_to_label,
        sample_keys=dataset.sample_keys,
        train_mean=train_mean,
        train_std=train_std,
    )


def build_stratified_cv_splits(
    dataset: DatasetBundle,
    *,
    test_ratio: float = 0.15,
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[SplitBundle]:
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio must be > 0 and < 1")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    indices = np.arange(dataset.y.shape[0])
    ## if every class has 15 samples and you use test_ratio=0.15, then each class ideally contributes 2.25 samples.
    # Since that is impossible, the split will assign: some classes 2; some classes 3.
    trainval_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=dataset.y,
    )

    trainval_y = dataset.y[trainval_indices]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    split_bundles: List[SplitBundle] = []
    for fold, (train_rel_idx, val_rel_idx) in enumerate(skf.split(trainval_indices, trainval_y)):
        train_indices = trainval_indices[train_rel_idx]
        val_indices = trainval_indices[val_rel_idx]
        split_bundles.append(
            _build_split_bundle(
                dataset,
                fold=fold,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )
        )

    return split_bundles
