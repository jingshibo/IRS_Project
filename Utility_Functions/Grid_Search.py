import itertools
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypedDict

from Utility_Functions import Model_Training


@dataclass
class GridSearchTrial:
    """One grid-search trial result."""

    trial_id: int
    params: Dict[str, Any]
    mean_best_val_acc: float
    fold_best_val_accs: List[float]
    fold_best_epochs: List[int]
    duration_sec: float
    train_out: Optional[Model_Training.TrainOutput] = None


class GridSearchOutput(TypedDict):
    """Return object for grid search runs."""

    trials: List[GridSearchTrial]
    best_trial: GridSearchTrial
    sorted_trials: List[GridSearchTrial]


def _expand_param_grid(param_grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Expand a dict of parameter lists into all combinations."""
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values_product = itertools.product(*(param_grid[k] for k in keys))
    return [dict(zip(keys, values)) for values in values_product]


def _validate_param_grid(param_grid: Dict[str, Sequence[Any]]) -> None:
    """Validate grid-search parameter lists."""
    for key, values in param_grid.items():
        if isinstance(values, (str, bytes)):
            raise ValueError(f"Grid parameter '{key}' must be a sequence of values, not a string.")
        try:
            n_values = len(values)
        except TypeError as exc:
            raise ValueError(f"Grid parameter '{key}' must be a sized sequence.") from exc
        if n_values == 0:
            raise ValueError(f"Grid parameter '{key}' has an empty value list.")


def run_training_grid_search(
    cv_folds,
    param_grid: Dict[str, Sequence[Any]],
    fixed_params: Optional[Dict[str, Any]] = None,
    class_order: Optional[Sequence[str]] = ("LOW", "TARGET", "HIGH"),
    keep_train_outputs: bool = False,
    maximize_metric: bool = True,
    sort_metric: str = "mean_best_val_acc",
    print_progress: bool = True,
) -> GridSearchOutput:
    """Grid-search important training parameters on existing normalized cv_folds.

    Args:
        cv_folds: output of `Preprocessing.build_normalized_cv_folds`
        param_grid: e.g. {"lr": [1e-3, 3e-4], "batch_size": [32, 64]}
        fixed_params: constant kwargs passed to `Model_Training.train_1d_cnn_cv`
        class_order: label order for training
        keep_train_outputs: if True, store full `TrainOutput` for each trial (uses more memory)
        maximize_metric: True for accuracy-like metrics
        sort_metric: currently supports "mean_best_val_acc"
        print_progress: print each trial summary while running
    """
    if len(cv_folds) == 0:
        raise ValueError("cv_folds is empty.")
    if sort_metric != "mean_best_val_acc":
        raise ValueError("Only sort_metric='mean_best_val_acc' is currently supported.")

    _validate_param_grid(param_grid)
    fixed_params = dict(fixed_params or {})
    combinations = _expand_param_grid(param_grid)

    trials: List[GridSearchTrial] = []
    total_trials = len(combinations)

    for trial_idx, combo_params in enumerate(combinations, start=1):
        train_kwargs = dict(fixed_params)
        train_kwargs.update(combo_params)
        train_kwargs["cv_folds"] = cv_folds
        train_kwargs["class_order"] = class_order

        start_time = time.perf_counter()
        train_out = Model_Training.train_1d_cnn_cv(**train_kwargs)
        duration_sec = time.perf_counter() - start_time

        fold_results = train_out["fold_results"]
        trial = GridSearchTrial(
            trial_id=trial_idx,
            params=combo_params,
            mean_best_val_acc=float(train_out["mean_best_val_acc"]),
            fold_best_val_accs=[float(res.best_val_acc) for res in fold_results],
            fold_best_epochs=[int(res.best_epoch) for res in fold_results],
            duration_sec=duration_sec,
            train_out=train_out if keep_train_outputs else None,
        )
        trials.append(trial)

        if print_progress:
            print(
                f"[GridSearch] trial {trial_idx}/{total_trials} "
                f"mean_best_val_acc={trial.mean_best_val_acc:.4f} "
                f"time={trial.duration_sec:.1f}s params={trial.params}"
            )

    sorted_trials = sorted(
        trials,
        key=lambda t: getattr(t, sort_metric),
        reverse=maximize_metric,
    )
    if not sorted_trials:
        raise RuntimeError("No grid-search trials were executed.")

    return {
        "trials": trials,
        "best_trial": sorted_trials[0],
        "sorted_trials": sorted_trials,
    }


def print_top_trials(grid_out: GridSearchOutput, top_k: int = 10) -> None:
    """Print a compact summary of the top grid-search trials."""
    top_k = max(1, top_k)
    print(f"Top {min(top_k, len(grid_out['sorted_trials']))} trials:")
    for rank, trial in enumerate(grid_out["sorted_trials"][:top_k], start=1):
        print(
            f"#{rank} trial_id={trial.trial_id} "
            f"mean_best_val_acc={trial.mean_best_val_acc:.4f} "
            f"time={trial.duration_sec:.1f}s params={trial.params}"
        )
