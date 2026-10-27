from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .training import TrainResult


def _extract_liquid_name(label: str) -> str:
    return label.split("__", maxsplit=1)[0]


def _compute_per_liquid_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_label: Dict[int, str],
) -> Dict[str, float]:
    per_liquid_correct: Dict[str, int] = {}
    per_liquid_total: Dict[str, int] = {}

    for true_idx, pred_idx in zip(y_true, y_pred):
        liquid = _extract_liquid_name(idx_to_label[int(true_idx)])
        per_liquid_total[liquid] = per_liquid_total.get(liquid, 0) + 1
        if int(true_idx) == int(pred_idx):
            per_liquid_correct[liquid] = per_liquid_correct.get(liquid, 0) + 1

    return {
        liquid: per_liquid_correct.get(liquid, 0) / total
        for liquid, total in per_liquid_total.items()
        if total > 0
    }


def _build_liquid_label_indices(idx_to_label: Dict[int, str]) -> Dict[str, List[int]]:
    liquid_to_indices: Dict[str, List[int]] = {}
    for label_idx, label in idx_to_label.items():
        liquid = _extract_liquid_name(label)
        liquid_to_indices.setdefault(liquid, []).append(int(label_idx))
    for liquid in liquid_to_indices:
        liquid_to_indices[liquid].sort()
    return liquid_to_indices


def _compute_recall_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    return np.divide(
        confusion_matrix,
        confusion_matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(confusion_matrix, dtype=np.float64),
        where=confusion_matrix.sum(axis=1, keepdims=True) != 0,
    )


def summarize_fold_results(fold_results: List[TrainResult]) -> Dict[str, object]:
    best_val_accs = np.asarray([result.best_val_acc for result in fold_results], dtype=np.float64)
    val_accs = np.asarray([result.val_acc for result in fold_results], dtype=np.float64)
    test_accs = np.asarray([result.test_acc for result in fold_results], dtype=np.float64)

    average_val_confusion_matrix = np.mean(
        [result.val_confusion_matrix.astype(np.float64) for result in fold_results],
        axis=0,
    )
    average_test_confusion_matrix = np.mean(
        [result.confusion_matrix.astype(np.float64) for result in fold_results],
        axis=0,
    )
    average_val_recall_matrix = _compute_recall_matrix(average_val_confusion_matrix)
    average_test_recall_matrix = _compute_recall_matrix(average_test_confusion_matrix)

    liquids = sorted({_extract_liquid_name(label) for label in fold_results[0].idx_to_label.values()})
    liquid_to_indices = _build_liquid_label_indices(fold_results[0].idx_to_label)
    val_per_liquid_by_fold = [
        _compute_per_liquid_accuracy(result.y_val_true, result.y_val_pred, result.idx_to_label)
        for result in fold_results
    ]
    test_per_liquid_by_fold = [
        _compute_per_liquid_accuracy(result.y_test_true, result.y_test_pred, result.idx_to_label)
        for result in fold_results
    ]

    per_liquid_accuracy: Dict[str, Dict[str, float]] = {}
    per_liquid_recall_matrices: Dict[str, Dict[str, np.ndarray]] = {}
    for liquid in liquids:
        liquid_val_accs = np.asarray([fold_metrics[liquid] for fold_metrics in val_per_liquid_by_fold], dtype=np.float64)
        liquid_test_accs = np.asarray([fold_metrics[liquid] for fold_metrics in test_per_liquid_by_fold], dtype=np.float64)
        liquid_indices = liquid_to_indices[liquid]
        per_liquid_accuracy[liquid] = {
            "val_mean": float(np.mean(liquid_val_accs)),
            "val_std": float(np.std(liquid_val_accs)),
            "test_mean": float(np.mean(liquid_test_accs)),
            "test_std": float(np.std(liquid_test_accs)),
        }
        per_liquid_recall_matrices[liquid] = {
            "val": average_val_recall_matrix[np.ix_(liquid_indices, liquid_indices)],
            "test": average_test_recall_matrix[np.ix_(liquid_indices, liquid_indices)],
        }

    return {
        "mean_best_val_acc": float(np.mean(best_val_accs)),
        "std_best_val_acc": float(np.std(best_val_accs)),
        "mean_val_acc": float(np.mean(val_accs)),
        "std_val_acc": float(np.std(val_accs)),
        "mean_test_acc": float(np.mean(test_accs)),
        "std_test_acc": float(np.std(test_accs)),
        "average_val_confusion_matrix": average_val_confusion_matrix,
        "average_test_confusion_matrix": average_test_confusion_matrix,
        "average_val_recall_matrix": average_val_recall_matrix,
        "average_test_recall_matrix": average_test_recall_matrix,
        "per_liquid_accuracy": per_liquid_accuracy,
        "per_liquid_recall_matrices": per_liquid_recall_matrices,
        "idx_to_label": fold_results[0].idx_to_label,
        "liquid_to_indices": liquid_to_indices,
        "liquids": liquids,
    }


def print_result_summary(summary: Dict[str, object]) -> None:
    print(
        f"Mean best val acc={summary['mean_best_val_acc']:.4f} +/- {summary['std_best_val_acc']:.4f}, "
        f"mean val acc={summary['mean_val_acc']:.4f} +/- {summary['std_val_acc']:.4f}, "
        f"mean test acc={summary['mean_test_acc']:.4f} +/- {summary['std_test_acc']:.4f}"
    )

    print("Per-liquid concentration accuracy across 5 folds:")
    for liquid in summary["liquids"]:
        liquid_metrics = summary["per_liquid_accuracy"][liquid]
        print(
            f"{liquid}: "
            f"val={liquid_metrics['val_mean']:.4f} +/- {liquid_metrics['val_std']:.4f}, "
            f"test={liquid_metrics['test_mean']:.4f} +/- {liquid_metrics['test_std']:.4f}"
        )

    print("Per-liquid validation recall submatrices:")
    for liquid in summary["liquids"]:
        print(f"{liquid} val recall:")
        print(summary["per_liquid_recall_matrices"][liquid]["val"])

    print("Per-liquid test recall submatrices:")
    for liquid in summary["liquids"]:
        print(f"{liquid} test recall:")
        print(summary["per_liquid_recall_matrices"][liquid]["test"])


def plot_selected_confusion_matrix(
    summary: Dict[str, object],
    matrix_name: str,
    *,
    liquid: str | None = None,
    cmap: str = "Blues",
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    matrix_name = matrix_name.lower()

    if liquid is None:
        matrix_map = {
            "val_confusion": summary["average_val_confusion_matrix"],
            "test_confusion": summary["average_test_confusion_matrix"],
            "val_recall": summary["average_val_recall_matrix"],
            "test_recall": summary["average_test_recall_matrix"],
        }
        if matrix_name not in matrix_map:
            raise ValueError("matrix_name must be one of: val_confusion, test_confusion, val_recall, test_recall")
        matrix = matrix_map[matrix_name]
        class_labels = [summary["idx_to_label"][idx] for idx in sorted(summary["idx_to_label"])]
        plot_title = title or matrix_name.replace("_", " ").title()
    else:
        if liquid not in summary["liquids"]:
            raise ValueError(f"Unknown liquid '{liquid}'")
        if matrix_name not in {"val_recall", "test_recall"}:
            raise ValueError("Per-liquid plotting currently supports val_recall or test_recall only")
        matrix = summary["per_liquid_recall_matrices"][liquid]["val" if matrix_name == "val_recall" else "test"]
        liquid_indices = summary["liquid_to_indices"][liquid]
        class_labels = [summary["idx_to_label"][idx].split("__", maxsplit=1)[1] for idx in liquid_indices]
        plot_title = title or f"{liquid} {matrix_name.replace('_', ' ').title()}"

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap=cmap)
    fig.colorbar(image, ax=ax)
    ax.set_title(plot_title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)

    threshold = float(matrix.max()) * 0.5 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = f"{value:.2f}" if "recall" in matrix_name else f"{value:.1f}"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_grouped_recall_matrix(
    summary: Dict[str, object],
    *,
    split: str = "test",
    cmap: str = "Blues",
    title: str | None = None,
    show_class_ticks: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    split = split.lower()
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    matrix = summary["average_val_recall_matrix"] if split == "val" else summary["average_test_recall_matrix"]
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

    ax.set_title(title or f"Overall {split.title()} Recall Matrix")
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
    split: str = "test",
    cmap: str = "Blues",
    title: str | None = None,
    show_class_ticks: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    split = split.lower()
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    matrix = summary["average_val_recall_matrix"] if split == "val" else summary["average_test_recall_matrix"]
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

    ax.set_title(title or f"Overall {split.title()} Recall Matrix")
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
