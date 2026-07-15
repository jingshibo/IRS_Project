from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class ResidualPreprocessingResult:
    x_train: np.ndarray
    x_test: np.ndarray
    liquid_means: Dict[str, np.ndarray]


def _validate_mode(mode: str) -> str:
    mode = mode.lower()
    if mode not in {"residual_only", "concat"}:
        raise ValueError("residual_mode must be 'residual_only' or 'concat'")
    return mode


def build_liquid_reference_residuals(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    train_liquids: Iterable[str],
    test_liquids: Iterable[str],
    residual_mode: str = "concat",
) -> ResidualPreprocessingResult:
    residual_mode = _validate_mode(residual_mode)
    train_liquids = list(train_liquids)
    test_liquids = list(test_liquids)

    if len(train_liquids) != len(x_train):
        raise ValueError("train_liquids length must match x_train")
    if len(test_liquids) != len(x_test):
        raise ValueError("test_liquids length must match x_test")

    liquid_means: Dict[str, np.ndarray] = {}
    for liquid in sorted(set(train_liquids)):
        mask = np.asarray([value == liquid for value in train_liquids], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"No training rows found for liquid '{liquid}'")
        liquid_means[liquid] = x_train[mask].mean(axis=0).astype(np.float32, copy=False)

    train_residuals: List[np.ndarray] = []
    for row, liquid in zip(x_train, train_liquids):
        train_residuals.append((row - liquid_means[liquid]).astype(np.float32, copy=False))

    test_residuals: List[np.ndarray] = []
    for row, liquid in zip(x_test, test_liquids):
        if liquid not in liquid_means:
            raise ValueError(f"Liquid '{liquid}' missing from training fold reference")
        test_residuals.append((row - liquid_means[liquid]).astype(np.float32, copy=False))

    x_train_residual = np.stack(train_residuals, axis=0).astype(np.float32, copy=False)
    x_test_residual = np.stack(test_residuals, axis=0).astype(np.float32, copy=False)

    if residual_mode == "residual_only":
        transformed_train = x_train_residual
        transformed_test = x_test_residual
    else:
        transformed_train = np.concatenate([x_train, x_train_residual], axis=1).astype(np.float32, copy=False)
        transformed_test = np.concatenate([x_test, x_test_residual], axis=1).astype(np.float32, copy=False)

    return ResidualPreprocessingResult(
        x_train=transformed_train,
        x_test=transformed_test,
        liquid_means=liquid_means,
    )

