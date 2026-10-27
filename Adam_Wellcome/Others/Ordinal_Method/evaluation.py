from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from Repeated_Measurements.evaluation import recall_matrix


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


def summarize_ordinal_results(fold_results: List[object]) -> Dict[str, object]:
    measurement_joint_accs = np.asarray([result.measurement_joint_acc for result in fold_results], dtype=np.float64)
    sample_joint_accs = np.asarray([result.sample_joint_acc for result in fold_results], dtype=np.float64)
    measurement_liquid_accs = np.asarray([result.measurement_liquid_acc for result in fold_results], dtype=np.float64)
    sample_liquid_accs = np.asarray([result.sample_liquid_acc for result in fold_results], dtype=np.float64)
    measurement_concentration_accs = np.asarray(
        [result.measurement_concentration_acc for result in fold_results],
        dtype=np.float64,
    )
    sample_concentration_accs = np.asarray(
        [result.sample_concentration_acc for result in fold_results],
        dtype=np.float64,
    )

    average_measurement_confusion = np.mean(
        [result.measurement_confusion_matrix.astype(np.float64) for result in fold_results],
        axis=0,
    )
    average_sample_confusion = np.mean(
        [result.sample_confusion_matrix.astype(np.float64) for result in fold_results],
        axis=0,
    )
    idx_to_label = fold_results[0].idx_to_joint_label
    liquid_to_indices = _build_liquid_label_indices(idx_to_label)
    liquids = sorted(liquid_to_indices)

    return {
        "mean_measurement_joint_acc": float(np.mean(measurement_joint_accs)),
        "std_measurement_joint_acc": float(np.std(measurement_joint_accs)),
        "mean_sample_joint_acc": float(np.mean(sample_joint_accs)),
        "std_sample_joint_acc": float(np.std(sample_joint_accs)),
        "mean_measurement_liquid_acc": float(np.mean(measurement_liquid_accs)),
        "std_measurement_liquid_acc": float(np.std(measurement_liquid_accs)),
        "mean_sample_liquid_acc": float(np.mean(sample_liquid_accs)),
        "std_sample_liquid_acc": float(np.std(sample_liquid_accs)),
        "mean_measurement_concentration_acc": float(np.mean(measurement_concentration_accs)),
        "std_measurement_concentration_acc": float(np.std(measurement_concentration_accs)),
        "mean_sample_concentration_acc": float(np.mean(sample_concentration_accs)),
        "std_sample_concentration_acc": float(np.std(sample_concentration_accs)),
        "average_measurement_confusion_matrix": average_measurement_confusion,
        "average_measurement_recall_matrix": recall_matrix(average_measurement_confusion),
        "average_sample_confusion_matrix": average_sample_confusion,
        "average_sample_recall_matrix": recall_matrix(average_sample_confusion),
        "idx_to_label": idx_to_label,
        "liquid_to_indices": liquid_to_indices,
        "liquids": liquids,
    }


def print_ordinal_summary(summary: Dict[str, object]) -> None:
    print(
        f"Ordinal measurement joint acc={summary['mean_measurement_joint_acc']:.4f} "
        f"+/- {summary['std_measurement_joint_acc']:.4f}"
    )
    print(
        f"Ordinal sample joint acc={summary['mean_sample_joint_acc']:.4f} "
        f"+/- {summary['std_sample_joint_acc']:.4f}"
    )
    print(
        f"Ordinal measurement liquid acc={summary['mean_measurement_liquid_acc']:.4f} "
        f"+/- {summary['std_measurement_liquid_acc']:.4f}"
    )
    print(
        f"Ordinal sample liquid acc={summary['mean_sample_liquid_acc']:.4f} "
        f"+/- {summary['std_sample_liquid_acc']:.4f}"
    )
    print(
        f"Ordinal measurement concentration acc={summary['mean_measurement_concentration_acc']:.4f} "
        f"+/- {summary['std_measurement_concentration_acc']:.4f}"
    )
    print(
        f"Ordinal sample concentration acc={summary['mean_sample_concentration_acc']:.4f} "
        f"+/- {summary['std_sample_concentration_acc']:.4f}"
    )


def plot_grouped_recall_matrix(
    summary: Dict[str, object],
    *,
    matrix_name: str = "sample",
    cmap: str = "Blues",
    title: str | None = None,
    show_class_ticks: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    matrix_name = matrix_name.lower()
    matrix_map = {
        "measurement": summary["average_measurement_recall_matrix"],
        "sample": summary["average_sample_recall_matrix"],
    }
    if matrix_name not in matrix_map:
        raise ValueError("matrix_name must be 'measurement' or 'sample'")

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
        boundaries.append(indices[-1] + 0.5)
        centers.append((indices[0] + indices[-1]) / 2.0)

    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="black", linewidth=1.2, alpha=0.8)
        ax.axvline(boundary, color="black", linewidth=1.2, alpha=0.8)

    ax.set_title(title or f"Ordinal {matrix_name.title()} Recall Matrix")
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
    matrix_name: str = "sample",
    cmap: str = "Blues",
    title: str | None = None,
    show_class_ticks: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    matrix_name = matrix_name.lower()
    matrix_map = {
        "measurement": summary["average_measurement_recall_matrix"],
        "sample": summary["average_sample_recall_matrix"],
    }
    if matrix_name not in matrix_map:
        raise ValueError("matrix_name must be 'measurement' or 'sample'")

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
        boundaries.append(indices[-1] + 0.5)
        centers.append((indices[0] + indices[-1]) / 2.0)

    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="black", linewidth=1.2, alpha=0.8)
        ax.axvline(boundary, color="black", linewidth=1.2, alpha=0.8)

    ax.set_title(title or f"Ordinal {matrix_name.title()} Recall Matrix")
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
