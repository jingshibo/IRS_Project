from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def build_stratified_representative_indices(
    y: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Select reproducible, class-balanced calibration sample indices."""
    labels = np.asarray(y)
    n_samples = len(labels)
    if n_samples == 0:
        raise ValueError("Cannot select representative samples from an empty training set.")
    count = min(max(int(count), 1), n_samples)

    indices = np.arange(n_samples, dtype=np.int64)
    if count == n_samples:
        return indices

    _, representative_indices = train_test_split(
        indices,
        test_size=count,
        stratify=labels,
        random_state=seed,
    )
    return np.asarray(representative_indices, dtype=np.int64)
