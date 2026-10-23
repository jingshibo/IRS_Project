from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _recall_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    return np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=np.float64),
        where=row_sums != 0,
    )


def _macro_f1(confusion_matrix: np.ndarray) -> float:
    scores = []
    for class_index in range(confusion_matrix.shape[0]):
        true_positive = float(confusion_matrix[class_index, class_index])
        false_positive = float(confusion_matrix[:, class_index].sum() - true_positive)
        false_negative = float(confusion_matrix[class_index, :].sum() - true_positive)
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0.0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def summarize_results(fold_results: List[object]) -> Dict[str, object]:
    if not fold_results:
        raise ValueError("fold_results must not be empty")

    measurement_accuracies = np.asarray(
        [result.measurement_accuracy for result in fold_results],
        dtype=np.float64,
    )
    sample_accuracies = np.asarray(
        [result.sample_accuracy for result in fold_results],
        dtype=np.float64,
    )
    measurement_confusion = np.sum(
        [result.measurement_confusion_matrix for result in fold_results],
        axis=0,
        dtype=np.int64,
    )
    sample_confusion = np.sum(
        [result.sample_confusion_matrix for result in fold_results],
        axis=0,
        dtype=np.int64,
    )

    return {
        "mean_measurement_accuracy": float(measurement_accuracies.mean()),
        "std_measurement_accuracy": float(measurement_accuracies.std()),
        "mean_sample_accuracy": float(sample_accuracies.mean()),
        "std_sample_accuracy": float(sample_accuracies.std()),
        "pooled_measurement_accuracy": float(
            np.trace(measurement_confusion) / measurement_confusion.sum()
        ),
        "pooled_sample_accuracy": float(
            np.trace(sample_confusion) / sample_confusion.sum()
        ),
        "measurement_macro_f1": _macro_f1(measurement_confusion),
        "sample_macro_f1": _macro_f1(sample_confusion),
        "measurement_confusion_matrix": measurement_confusion,
        "sample_confusion_matrix": sample_confusion,
        "measurement_recall_matrix": _recall_matrix(measurement_confusion),
        "sample_recall_matrix": _recall_matrix(sample_confusion),
        "idx_to_label": fold_results[0].idx_to_label,
    }


def print_result_summary(summary: Dict[str, object]) -> None:
    print(
        "Validation measurement accuracy: "
        f"{summary['mean_measurement_accuracy']:.4f} +/- "
        f"{summary['std_measurement_accuracy']:.4f} "
        f"(pooled={summary['pooled_measurement_accuracy']:.4f}, "
        f"macro F1={summary['measurement_macro_f1']:.4f})"
    )
    print(
        "Validation sample accuracy:      "
        f"{summary['mean_sample_accuracy']:.4f} +/- "
        f"{summary['std_sample_accuracy']:.4f} "
        f"(pooled={summary['pooled_sample_accuracy']:.4f}, "
        f"macro F1={summary['sample_macro_f1']:.4f})"
    )


def plot_recall_matrix_with_numbers(
    summary: Dict[str, object],
    *,
    level: str = "sample",
    cmap: str = "Blues",
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    level = level.lower()
    matrix_map = {
        "measurement": summary["measurement_recall_matrix"],
        "sample": summary["sample_recall_matrix"],
    }
    if level not in matrix_map:
        raise ValueError("level must be 'measurement' or 'sample'")

    matrix = np.asarray(matrix_map[level], dtype=np.float64)
    idx_to_label = summary["idx_to_label"]
    labels = [idx_to_label[index].replace("10-", "10^-") for index in range(len(idx_to_label))]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Recall", rotation=270, labelpad=14)

    ax.set_title(title or f"Concentration {level.title()} Validation Recall")
    ax.set_xlabel("Predicted concentration")
    ax.set_ylabel("True concentration")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )

    fig.tight_layout()
    plt.show()
    return fig, ax

