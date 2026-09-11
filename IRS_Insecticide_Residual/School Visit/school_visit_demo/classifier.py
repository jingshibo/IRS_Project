from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DemoClassificationResult:
    train_indices: np.ndarray
    test_indices: np.ndarray
    x_train_map: np.ndarray
    x_test_map: np.ndarray
    x_train_map_3d: np.ndarray
    x_test_map_3d: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    class_order: tuple[str, ...]
    scaler: StandardScaler
    reducer: PCA
    reducer_3d: PCA
    classifier: Any
    method_name: str
    input_description: str
    test_accuracy: float
    confusion_count: np.ndarray
    unknown_global_index: int
    unknown_map_point: np.ndarray
    unknown_map_point_3d: np.ndarray
    unknown_true_label: str
    unknown_pred_label: str
    unknown_prob: np.ndarray
    pre_cnn_x_train_map: np.ndarray | None = None
    pre_cnn_x_test_map: np.ndarray | None = None
    pre_cnn_unknown_map_point: np.ndarray | None = None
    pre_cnn_x_train_map_3d: np.ndarray | None = None
    pre_cnn_x_test_map_3d: np.ndarray | None = None
    pre_cnn_unknown_map_point_3d: np.ndarray | None = None


def train_demo_classifier(
    x_features: np.ndarray,
    y_all: np.ndarray,
    class_order: Sequence[str],
    random_seed: int = 42,
    test_size: float = 0.2,
    unknown_test_position: int | None = None,
    method_name: str = "KNN",
    input_description: str = "simple signal features",
) -> DemoClassificationResult:
    """Train a simple feature-similarity classifier for an explainable visitor demo."""
    x_features = np.asarray(x_features, dtype=np.float32)
    y_all = np.asarray(y_all)
    indices = np.arange(len(y_all))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=y_all,
        random_state=random_seed,
    )

    x_train = x_features[train_indices]
    x_test = x_features[test_indices]
    y_train = y_all[train_indices]
    y_test = y_all[test_indices]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_map, x_test_map, reducer, x_train_map_3d, x_test_map_3d, reducer_3d = build_pca_feature_maps(
        x_train_scaled,
        x_test_scaled,
        random_seed=random_seed,
        values_are_scaled=True,
    )

    n_neighbors = min(5, len(x_train_scaled))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    classifier.fit(x_train_scaled, y_train)

    y_pred = classifier.predict(x_test_scaled)
    y_prob = _predict_prob_in_class_order(classifier, x_test_scaled, class_order)
    test_accuracy = float(np.mean(y_pred == y_test))
    confusion_count = confusion_matrix(y_test, y_pred, labels=list(class_order))

    unknown_position = _choose_unknown_position(
        y_prob=y_prob,
        y_pred=y_pred,
        y_true=y_test,
        requested_position=unknown_test_position,
    )
    return DemoClassificationResult(
        train_indices=train_indices,
        test_indices=test_indices,
        x_train_map=x_train_map,
        x_test_map=x_test_map,
        x_train_map_3d=x_train_map_3d,
        x_test_map_3d=x_test_map_3d,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        class_order=tuple(class_order),
        scaler=scaler,
        reducer=reducer,
        reducer_3d=reducer_3d,
        classifier=classifier,
        method_name=method_name,
        input_description=input_description,
        test_accuracy=test_accuracy,
        confusion_count=confusion_count,
        unknown_global_index=int(test_indices[unknown_position]),
        unknown_map_point=x_test_map[unknown_position],
        unknown_map_point_3d=x_test_map_3d[unknown_position],
        unknown_true_label=str(y_test[unknown_position]),
        unknown_pred_label=str(y_pred[unknown_position]),
        unknown_prob=y_prob[unknown_position],
    )


def train_pca_demo_classifier(
    x_features: np.ndarray,
    y_all: np.ndarray,
    class_order: Sequence[str],
    random_seed: int = 42,
    test_size: float = 0.2,
    unknown_test_position: int | None = None,
) -> DemoClassificationResult:
    """Train a simple classifier after reducing processed signals to PCA coordinates."""
    x_features = np.asarray(x_features, dtype=np.float32)
    y_all = np.asarray(y_all)
    indices = np.arange(len(y_all))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=y_all,
        random_state=random_seed,
    )

    x_train = x_features[train_indices]
    x_test = x_features[test_indices]
    y_train = y_all[train_indices]
    y_test = y_all[test_indices]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    reducer = PCA(n_components=2, random_state=random_seed)
    x_train_map = reducer.fit_transform(x_train_scaled).astype(np.float32, copy=False)
    x_test_map = reducer.transform(x_test_scaled).astype(np.float32, copy=False)

    reducer_3d = PCA(n_components=3, random_state=random_seed)
    x_train_map_3d = reducer_3d.fit_transform(x_train_scaled).astype(np.float32, copy=False)
    x_test_map_3d = reducer_3d.transform(x_test_scaled).astype(np.float32, copy=False)

    n_neighbors = min(5, len(x_train_map_3d))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    classifier.fit(x_train_map_3d, y_train)

    y_pred = classifier.predict(x_test_map_3d)
    y_prob = _predict_prob_in_class_order(classifier, x_test_map_3d, class_order)
    test_accuracy = float(np.mean(y_pred == y_test))
    confusion_count = confusion_matrix(y_test, y_pred, labels=list(class_order))

    unknown_position = _choose_unknown_position(
        y_prob=y_prob,
        y_pred=y_pred,
        y_true=y_test,
        requested_position=unknown_test_position,
    )
    return DemoClassificationResult(
        train_indices=train_indices,
        test_indices=test_indices,
        x_train_map=x_train_map,
        x_test_map=x_test_map,
        x_train_map_3d=x_train_map_3d,
        x_test_map_3d=x_test_map_3d,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        class_order=tuple(class_order),
        scaler=scaler,
        reducer=reducer,
        reducer_3d=reducer_3d,
        classifier=classifier,
        method_name="PCA",
        input_description="3D PCA coordinates from processed signal",
        test_accuracy=test_accuracy,
        confusion_count=confusion_count,
        unknown_global_index=int(test_indices[unknown_position]),
        unknown_map_point=x_test_map[unknown_position],
        unknown_map_point_3d=x_test_map_3d[unknown_position],
        unknown_true_label=str(y_test[unknown_position]),
        unknown_pred_label=str(y_pred[unknown_position]),
        unknown_prob=y_prob[unknown_position],
    )


def build_pca_feature_maps(
    x_train_features: np.ndarray,
    x_test_features: np.ndarray,
    random_seed: int = 42,
    values_are_scaled: bool = False,
) -> tuple[np.ndarray, np.ndarray, PCA, np.ndarray, np.ndarray, PCA]:
    """Build 2D and 3D PCA maps for display from tabular feature inputs."""
    x_train_features = np.asarray(x_train_features, dtype=np.float32)
    x_test_features = np.asarray(x_test_features, dtype=np.float32)

    if values_are_scaled:
        x_train_scaled = x_train_features
        x_test_scaled = x_test_features
    else:
        map_scaler = StandardScaler()
        x_train_scaled = map_scaler.fit_transform(x_train_features)
        x_test_scaled = map_scaler.transform(x_test_features)

    reducer = PCA(n_components=2, random_state=random_seed)
    x_train_map = reducer.fit_transform(x_train_scaled).astype(np.float32, copy=False)
    x_test_map = reducer.transform(x_test_scaled).astype(np.float32, copy=False)

    reducer_3d = PCA(n_components=3, random_state=random_seed)
    x_train_map_3d = reducer_3d.fit_transform(x_train_scaled).astype(np.float32, copy=False)
    x_test_map_3d = reducer_3d.transform(x_test_scaled).astype(np.float32, copy=False)

    return x_train_map, x_test_map, reducer, x_train_map_3d, x_test_map_3d, reducer_3d


def _predict_prob_in_class_order(
    classifier: KNeighborsClassifier,
    x_map: np.ndarray,
    class_order: Sequence[str],
) -> np.ndarray:
    raw_prob = classifier.predict_proba(x_map)
    ordered_prob = np.zeros((raw_prob.shape[0], len(class_order)), dtype=np.float32)
    class_to_idx = {str(label): idx for idx, label in enumerate(class_order)}
    for source_col, label in enumerate(classifier.classes_):
        ordered_prob[:, class_to_idx[str(label)]] = raw_prob[:, source_col]
    return ordered_prob


def _choose_unknown_position(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    requested_position: int | None,
) -> int:
    if requested_position is not None:
        if not 0 <= requested_position < len(y_true):
            raise IndexError(
                f"unknown_test_position must be in [0, {len(y_true) - 1}], got {requested_position}"
            )
        return int(requested_position)

    correct_mask = y_pred == y_true
    if np.any(correct_mask):
        correct_positions = np.flatnonzero(correct_mask)
        confidence = y_prob[correct_positions].max(axis=1)
        return int(correct_positions[np.argmax(confidence)])
    return int(np.argmax(y_prob.max(axis=1)))


def choose_unknown_position(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    requested_position: int | None,
) -> int:
    return _choose_unknown_position(
        y_prob=y_prob,
        y_pred=y_pred,
        y_true=y_true,
        requested_position=requested_position,
    )
