import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from IRS_Insecticide_Residual.Utility_Functions.Preprocessing import build_stratified_cv_indices

##  standardize a tabular feature matrix [N, F] using train-only statistics
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


def reduce_fold_feature_matrix(
    x_train,
    x_val,
    n_components=None,
    random_seed=42,
):
    """Fit PCA on one training fold and transform both train and validation features."""
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)

    reducer = PCA(n_components=n_components, random_state=random_seed)
    x_train_reduced = reducer.fit_transform(x_train)
    x_val_reduced = reducer.transform(x_val)
    return x_train_reduced.astype(np.float32, copy=False), x_val_reduced.astype(np.float32, copy=False), reducer


def get_pca_details(pca: PCA) -> dict:
    """Return useful PCA diagnostics from a fitted PCA object."""
    explained_variance_ratio = pca.explained_variance_ratio_.astype(np.float32, copy=False)
    explained_variance = pca.explained_variance_.astype(np.float32, copy=False)
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio).astype(np.float32, copy=False)

    return {
        "n_components": int(pca.n_components_),
        "requested_n_components": pca.n_components,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance_ratio": cumulative_explained_variance_ratio,
        "explained_variance": explained_variance,
        "singular_values": pca.singular_values_.astype(np.float32, copy=False),
        "total_explained_variance_ratio": np.float32(cumulative_explained_variance_ratio[-1]),
    }


def normalize_reduce_fold_feature_matrix(
    x_train,
    x_val,
    n_components=None,
    random_seed=42,
):
    """Standardize one feature fold, then fit PCA on training data and transform validation data."""
    x_train_norm, x_val_norm, feature_scaler = normalize_fold_feature_matrix(x_train, x_val)
    x_train_reduced, x_val_reduced, feature_reducer = reduce_fold_feature_matrix(
        x_train_norm,
        x_val_norm,
        n_components=n_components,
        random_seed=random_seed,
    )
    return x_train_reduced, x_val_reduced, feature_scaler, feature_reducer


def normalize_reduce_train_test_feature_matrix(
    x_train,
    x_test,
    n_components=None,
    random_seed=42,
):
    """Fit StandardScaler plus PCA on training features and transform holdout test features."""
    return normalize_reduce_fold_feature_matrix(
        x_train,
        x_test,
        n_components=n_components,
        random_seed=random_seed,
    )

##  build leakage-safe normalized CV folds for a feature matrix [N, F], with optional PCA
def build_feature_cv_folds(
    x_trainval,
    y_trainval,
    n_splits=5,
    random_seed=42,
    use_pca=False,
    pca_n_components=None,
):
    """Build leakage-safe feature CV folds with train-only StandardScaler and optional PCA."""
    x_trainval = np.asarray(x_trainval, dtype=np.float32)
    if x_trainval.ndim != 2:
        raise ValueError(f"x_trainval must have shape [N, F], got {x_trainval.shape}")

    folds = []
    cv_indices = build_stratified_cv_indices(y_trainval, n_splits=n_splits, random_seed=random_seed)

    for fold_id, (train_idx, val_idx) in enumerate(cv_indices):
        x_train = x_trainval[train_idx]
        y_train = y_trainval[train_idx]
        x_val = x_trainval[val_idx]
        y_val = y_trainval[val_idx]

        x_train_norm, x_val_norm, feature_scaler = normalize_fold_feature_matrix(x_train, x_val)
        feature_reducer = None
        pca_details = None
        reduction_method = "none"
        if use_pca:
            x_train_out, x_val_out, feature_reducer = reduce_fold_feature_matrix(
                x_train_norm,
                x_val_norm,
                n_components=pca_n_components,
                random_seed=random_seed,
            )
            pca_details = get_pca_details(feature_reducer)
            reduction_method = "pca"
        else:
            x_train_out = x_train_norm
            x_val_out = x_val_norm

        fold_payload = {
            "fold": fold_id,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "X_train": x_train_out,
            "y_train": y_train,
            "X_val": x_val_out,
            "y_val": y_val,
            "feature_scaler": feature_scaler,
            "feature_reducer": feature_reducer,
            "reduction_method": reduction_method,
            "n_components": x_train_out.shape[1],
        }
        if pca_details is not None:
            fold_payload["pca_details"] = pca_details

        folds.append(fold_payload)

    return folds
