from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SampleLevelResult:
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray
    group_ids: np.ndarray
    probabilities: np.ndarray
    confusion_matrix: np.ndarray


def _extract_liquid_name(label: str) -> str:
    return label.split("__", maxsplit=1)[0]


def _build_liquid_label_indices(idx_to_label: Dict[int, str]) -> Dict[str, List[int]]:
    liquid_to_indices: Dict[str, List[int]] = {}
    for label_idx, label in idx_to_label.items():
        liquid = _extract_liquid_name(label)
        liquid_to_indices.setdefault(liquid, []).append(int(label_idx))
    for liquid in liquid_to_indices:
        liquid_to_indices[liquid].sort()
    return liquid_to_indices


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        confusion[int(true_idx), int(pred_idx)] += 1
    return confusion


def aggregate_repeated_measurements(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    group_ids: np.ndarray,
    *,
    num_classes: int,
    method: str = "probability_average",
) -> SampleLevelResult:
    method = method.lower()
    if method not in {"probability_average", "majority_vote"}:
        raise ValueError("method must be 'probability_average' or 'majority_vote'")
    if not (len(y_true) == len(y_pred) == len(probabilities) == len(group_ids)):
        raise ValueError("y_true, y_pred, probabilities, and group_ids must have the same length")

    sample_group_ids: List[int] = []
    sample_true: List[int] = []
    sample_pred: List[int] = []
    sample_probabilities: List[np.ndarray] = []

    for group_id in sorted(set(int(value) for value in group_ids)):
        mask = group_ids == group_id
        group_true = y_true[mask]
        if np.unique(group_true).size != 1:
            raise ValueError(f"Group {group_id} contains multiple true labels: {np.unique(group_true)}")

        mean_probability = probabilities[mask].mean(axis=0)
        if method == "probability_average":
            pred_idx = int(np.argmax(mean_probability))
        else:
            vote_counts = np.bincount(y_pred[mask], minlength=num_classes)
            pred_idx = int(np.argmax(vote_counts))

        sample_group_ids.append(group_id)
        sample_true.append(int(group_true[0]))
        sample_pred.append(pred_idx)
        sample_probabilities.append(mean_probability.astype(np.float64, copy=False))

    sample_true_array = np.asarray(sample_true, dtype=np.int64)
    sample_pred_array = np.asarray(sample_pred, dtype=np.int64)
    confusion = compute_confusion_matrix(
        sample_true_array,
        sample_pred_array,
        num_classes=num_classes,
    )

    return SampleLevelResult(
        accuracy=float((sample_true_array == sample_pred_array).mean()) if sample_true_array.size else 0.0,
        y_true=sample_true_array,
        y_pred=sample_pred_array,
        group_ids=np.asarray(sample_group_ids, dtype=np.int64),
        probabilities=np.stack(sample_probabilities, axis=0) if sample_probabilities else np.empty((0, num_classes)),
        confusion_matrix=confusion,
    )


def recall_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    return np.divide(
        confusion_matrix,
        confusion_matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(confusion_matrix, dtype=np.float64),
        where=confusion_matrix.sum(axis=1, keepdims=True) != 0,
    )


def summarize_repeated_measurement_results(fold_results: List[object]) -> Dict[str, object]:
    measurement_accs = np.asarray([result.measurement_test_acc for result in fold_results], dtype=np.float64)
    probability_sample_accs = np.asarray(
        [result.sample_probability_average.accuracy for result in fold_results],
        dtype=np.float64,
    )
    majority_sample_accs = np.asarray(
        [result.sample_majority_vote.accuracy for result in fold_results],
        dtype=np.float64,
    )

    average_measurement_confusion = np.mean(
        [result.measurement_confusion_matrix.astype(np.float64) for result in fold_results],
        axis=0,
    )
    average_probability_sample_confusion = np.mean(
        [result.sample_probability_average.confusion_matrix.astype(np.float64) for result in fold_results],
        axis=0,
    )
    average_majority_sample_confusion = np.mean(
        [result.sample_majority_vote.confusion_matrix.astype(np.float64) for result in fold_results],
        axis=0,
    )
    idx_to_label = fold_results[0].idx_to_label
    liquid_to_indices = _build_liquid_label_indices(idx_to_label)
    liquids = sorted(liquid_to_indices)

    return {
        "mean_measurement_test_acc": float(np.mean(measurement_accs)),
        "std_measurement_test_acc": float(np.std(measurement_accs)),
        "mean_sample_probability_acc": float(np.mean(probability_sample_accs)),
        "std_sample_probability_acc": float(np.std(probability_sample_accs)),
        "mean_sample_majority_acc": float(np.mean(majority_sample_accs)),
        "std_sample_majority_acc": float(np.std(majority_sample_accs)),
        "average_measurement_confusion_matrix": average_measurement_confusion,
        "average_measurement_recall_matrix": recall_matrix(average_measurement_confusion),
        "average_sample_probability_confusion_matrix": average_probability_sample_confusion,
        "average_sample_probability_recall_matrix": recall_matrix(average_probability_sample_confusion),
        "average_sample_majority_confusion_matrix": average_majority_sample_confusion,
        "average_sample_majority_recall_matrix": recall_matrix(average_majority_sample_confusion),
        "idx_to_label": idx_to_label,
        "liquid_to_indices": liquid_to_indices,
        "liquids": liquids,
    }


def print_repeated_measurement_summary(summary: Dict[str, object]) -> None:
    print(
        f"Measurement test acc={summary['mean_measurement_test_acc']:.4f} "
        f"+/- {summary['std_measurement_test_acc']:.4f}"
    )
    print(
        f"Sample test acc, probability average={summary['mean_sample_probability_acc']:.4f} "
        f"+/- {summary['std_sample_probability_acc']:.4f}"
    )
    print(
        f"Sample test acc, majority vote={summary['mean_sample_majority_acc']:.4f} "
        f"+/- {summary['std_sample_majority_acc']:.4f}"
    )


def plot_grouped_recall_matrix(
    summary: Dict[str, object],
    *,
    matrix_name: str = "sample_probability",
    cmap: str = "Blues",
    title: str | None = None,
    show_class_ticks: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    matrix_name = matrix_name.lower()
    matrix_map = {
        "measurement": summary["average_measurement_recall_matrix"],
        "sample_probability": summary["average_sample_probability_recall_matrix"],
        "sample_majority": summary["average_sample_majority_recall_matrix"],
    }
    if matrix_name not in matrix_map:
        raise ValueError("matrix_name must be 'measurement', 'sample_probability', or 'sample_majority'")

    matrix = matrix_map[matrix_name]
    liquids = summary["liquids"]
    liquid_to_indices = summary["liquid_to_indices"]

    fig_size = max(10.0, matrix.shape[0] * 0.22)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, fraction=0.04, pad=0.03, aspect=30)
    colorbar.set_label("Recall", rotation=270, labelpad=14, fontsize=12)

    boundaries = []
    centers = []
    for liquid in liquids:
        indices = liquid_to_indices[liquid]
        start_idx = indices[0]
        end_idx = indices[-1]
        boundaries.append(end_idx + 0.5)
        centers.append((start_idx + end_idx) / 2.0)

    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="black", linewidth=1.2, alpha=0.8)
        ax.axvline(boundary, color="black", linewidth=1.2, alpha=0.8)

    default_title = {
        "measurement": "Measurement-Level Recall Matrix",
        "sample_probability": "Sample-Level Recall Matrix (Probability Average)",
        "sample_majority": "Sample-Level Recall Matrix (Majority Vote)",
    }[matrix_name]
    ax.set_title(title or default_title)
    ax.set_xlabel("Predicted Label Group")
    ax.set_ylabel("True Label Group")
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(liquids, rotation=45, ha="right")
    ax.set_yticklabels(liquids)

    if show_class_ticks:
        class_labels = [summary["idx_to_label"][idx] for idx in sorted(summary["idx_to_label"])]
        ax.set_xticks(np.arange(len(class_labels)), minor=True)
        ax.set_yticks(np.arange(len(class_labels)), minor=True)
        ax.set_xticklabels(class_labels, minor=True, rotation=90, fontsize=12)
        ax.set_yticklabels(class_labels, minor=True, fontsize=12)
        ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_grouped_recall_matrix_with_numbers(
    summary: Dict[str, object],
    *,
    matrix_name: str = "sample_probability",
    cmap: str = "Blues",
    title: str | None = None,
    show_class_ticks: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    matrix_name = matrix_name.lower()
    matrix_map = {
        "measurement": summary["average_measurement_recall_matrix"],
        "sample_probability": summary["average_sample_probability_recall_matrix"],
        "sample_majority": summary["average_sample_majority_recall_matrix"],
    }
    if matrix_name not in matrix_map:
        raise ValueError("matrix_name must be 'measurement', 'sample_probability', or 'sample_majority'")

    matrix = matrix_map[matrix_name]
    liquids = summary["liquids"]
    liquid_to_indices = summary["liquid_to_indices"]

    fig_size = max(10.0, matrix.shape[0] * 0.22)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, fraction=0.04, pad=0.03, aspect=30)
    colorbar.set_label("Recall", rotation=270, labelpad=14)
    annotation_fontsize = max(8, min(12, int(fig_size * 2)))

    boundaries = []
    centers = []
    for liquid in liquids:
        indices = liquid_to_indices[liquid]
        start_idx = indices[0]
        end_idx = indices[-1]
        boundaries.append(end_idx + 0.5)
        centers.append((start_idx + end_idx) / 2.0)

    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="black", linewidth=1.2, alpha=0.8)
        ax.axvline(boundary, color="black", linewidth=1.2, alpha=0.8)

    default_title = {
        "measurement": "Measurement-Level Recall Matrix",
        "sample_probability": "Overall Test Recall Matrix",
        "sample_majority": "Sample-Level Recall Matrix (Majority Vote)",
    }[matrix_name]
    ax.set_title(title or default_title)
    ax.set_xlabel("Predicted Label Group")
    ax.set_ylabel("True Label Group")
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(liquids, rotation=45, ha="right")
    ax.set_yticklabels(liquids)

    threshold = float(matrix.max()) * 0.5 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                color="white" if value > threshold else "black",
            )

    if show_class_ticks:
        class_labels = [summary["idx_to_label"][idx] for idx in sorted(summary["idx_to_label"])]
        ax.set_xticks(np.arange(len(class_labels)), minor=True)
        ax.set_yticks(np.arange(len(class_labels)), minor=True)
        ax.set_xticklabels(class_labels, minor=True, rotation=90, fontsize=6)
        ax.set_yticklabels(class_labels, minor=True, fontsize=6)
        ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    plt.show()
    return fig, ax
