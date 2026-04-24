import numpy as np
from sklearn.preprocessing import StandardScaler

from Utility_Functions.Preprocessing import build_stratified_cv_indices


def normalize_fold_feature_matrix(x_train, x_val):
    """Standardize a tabular feature matrix [N, F] using train-only statistics."""
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    if x_train.ndim != 2 or x_val.ndim != 2:
        raise ValueError(
            f"x_train and x_val must have shape [N, F], got {x_train.shape} and {x_val.shape}"
        )

    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_val_norm = scaler.transform(x_val)
    return x_train_norm.astype(np.float32, copy=False), x_val_norm.astype(np.float32, copy=False), scaler


def build_normalized_feature_cv_folds(
    x_trainval,
    y_trainval,
    n_splits=5,
    random_seed=42,
    cv_indices=None,
):
    """Build leakage-safe normalized CV folds for a feature matrix [N, F]."""
    x_trainval = np.asarray(x_trainval, dtype=np.float32)
    if x_trainval.ndim != 2:
        raise ValueError(f"x_trainval must have shape [N, F], got {x_trainval.shape}")

    folds = []
    cv_indices = (
        build_stratified_cv_indices(y_trainval, n_splits=n_splits, random_seed=random_seed)
        if cv_indices is None
        else list(cv_indices)
    )

    for fold_id, (train_idx, val_idx) in enumerate(cv_indices):
        x_train = x_trainval[train_idx]
        y_train = y_trainval[train_idx]
        x_val = x_trainval[val_idx]
        y_val = y_trainval[val_idx]

        x_train_norm, x_val_norm, feature_scaler = normalize_fold_feature_matrix(x_train, x_val)
        folds.append(
            {
                "fold": fold_id,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "X_train": x_train_norm,
                "y_train": y_train,
                "X_val": x_val_norm,
                "y_val": y_val,
                "feature_scaler": feature_scaler,
            }
        )

    return folds
