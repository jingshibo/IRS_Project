from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualStage2Data:
    x_train: np.ndarray
    x_test: np.ndarray
    liquid_mean: np.ndarray


def _validate_mode(mode: str) -> str:
    mode = mode.lower()
    if mode not in {"residual_only", "concat"}:
        raise ValueError("residual_mode must be 'residual_only' or 'concat'")
    return mode


def _apply_mode(
    *,
    x: np.ndarray,
    residual: np.ndarray,
    residual_mode: str,
) -> np.ndarray:
    residual_mode = _validate_mode(residual_mode)
    if residual_mode == "residual_only":
        return residual.astype(np.float32, copy=False)
    return np.concatenate([x, residual], axis=1).astype(np.float32, copy=False)


def build_stage2_residual_data(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    residual_mode: str = "residual_only",
) -> ResidualStage2Data:
    liquid_mean = x_train.mean(axis=0).astype(np.float32, copy=False)
    x_train_residual = (x_train - liquid_mean).astype(np.float32, copy=False)
    x_test_residual = (x_test - liquid_mean).astype(np.float32, copy=False)
    return ResidualStage2Data(
        x_train=_apply_mode(x=x_train, residual=x_train_residual, residual_mode=residual_mode),
        x_test=_apply_mode(x=x_test, residual=x_test_residual, residual_mode=residual_mode),
        liquid_mean=liquid_mean,
    )


def transform_with_liquid_mean(
    *,
    x: np.ndarray,
    liquid_mean: np.ndarray,
    residual_mode: str = "residual_only",
) -> np.ndarray:
    residual = (x - liquid_mean).astype(np.float32, copy=False)
    return _apply_mode(x=x, residual=residual, residual_mode=residual_mode)

