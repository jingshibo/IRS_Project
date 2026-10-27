from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TypedDict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


@dataclass
class SklearnFeatureFoldResult:
    fold: int
    model: object
    val_acc: float
    y_true_label: List[str]
    y_pred_label: List[str]
    y_prob: np.ndarray
    confusion_count: np.ndarray
    confusion_recall: np.ndarray


class SklearnFeatureTrainOutput(TypedDict):
    model_name: str
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    fold_results: List[SklearnFeatureFoldResult]
    mean_best_val_acc: float
    overall_confusion_count: np.ndarray
    overall_confusion_recall: np.ndarray


def _build_label_mapping(cv_folds: Sequence[dict], class_order: Optional[Sequence[str]]) -> Dict[str, int]:
    if class_order is not None:
        return {label: idx for idx, label in enumerate(class_order)}

    labels = []
    for fold_data in cv_folds:
        labels.extend(fold_data["y_train"].tolist())
        labels.extend(fold_data["y_val"].tolist())
    return {label: idx for idx, label in enumerate(sorted(set(labels)))}


def _create_model(model_name: str, random_seed: int):
    model_name = model_name.lower()
    if model_name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if model_name == "svm":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_seed,
        )
    if model_name in ("random_forest", "rf"):
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
    if model_name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    raise ValueError("model_name must be one of: 'lda', 'svm', 'random_forest', 'rf', 'knn'")


def _predict_prob_in_label_order(model, x: np.ndarray, label_to_idx: Dict[str, int]) -> np.ndarray:
    probs = model.predict_proba(np.asarray(x, dtype=np.float32))
    ordered_probs = np.zeros((probs.shape[0], len(label_to_idx)), dtype=np.float32)
    for source_col, label in enumerate(model.classes_):
        ordered_probs[:, label_to_idx[str(label)]] = probs[:, source_col]
    return ordered_probs


def _confusion_outputs(y_true_label, y_pred_label, class_order: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    count = confusion_matrix(y_true_label, y_pred_label, labels=list(class_order))
    row_sum = count.sum(axis=1, keepdims=True)
    recall = np.divide(
        count.astype(np.float64),
        row_sum,
        out=np.zeros_like(count, dtype=np.float64),
        where=row_sum != 0,
    )
    return count, recall


def train_sklearn_feature_cv(
    cv_folds: Sequence[dict],
    model_name: str,
    class_order: Optional[Sequence[str]] = ("LOW", "TARGET", "HIGH"),
    random_seed: int = 42,
) -> SklearnFeatureTrainOutput:
    """Train a classical sklearn classifier on each feature CV fold."""
    if len(cv_folds) == 0:
        raise ValueError("cv_folds is empty.")

    label_to_idx = _build_label_mapping(cv_folds, class_order)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    ordered_labels = [idx_to_label[idx] for idx in range(len(idx_to_label))]

    fold_results: List[SklearnFeatureFoldResult] = []
    all_true = []
    all_pred = []

    for fold_data in cv_folds:
        fold_id = int(fold_data["fold"])
        x_train = np.asarray(fold_data["X_train"], dtype=np.float32)
        x_val = np.asarray(fold_data["X_val"], dtype=np.float32)
        y_train = np.asarray(fold_data["y_train"])
        y_val = np.asarray(fold_data["y_val"])

        model = _create_model(model_name, random_seed=random_seed)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_val)
        y_prob = _predict_prob_in_label_order(model, x_val, label_to_idx)
        val_acc = float(np.mean(y_pred == y_val))
        confusion_count, confusion_recall = _confusion_outputs(y_val, y_pred, ordered_labels)

        fold_results.append(
            SklearnFeatureFoldResult(
                fold=fold_id,
                model=model,
                val_acc=val_acc,
                y_true_label=y_val.tolist(),
                y_pred_label=y_pred.tolist(),
                y_prob=y_prob,
                confusion_count=confusion_count,
                confusion_recall=confusion_recall,
            )
        )
        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())

    overall_confusion_count, overall_confusion_recall = _confusion_outputs(all_true, all_pred, ordered_labels)
    return {
        "model_name": model_name,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "fold_results": fold_results,
        "mean_best_val_acc": float(np.mean([fold.val_acc for fold in fold_results])),
        "overall_confusion_count": overall_confusion_count,
        "overall_confusion_recall": overall_confusion_recall,
    }


def predict_prob(model, x: np.ndarray, label_to_idx: Dict[str, int]) -> np.ndarray:
    """Predict sklearn class probabilities in the shared label index order."""
    return _predict_prob_in_label_order(model, x, label_to_idx)

