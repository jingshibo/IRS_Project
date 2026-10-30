from __future__ import annotations

import csv
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


CHANNEL_NAMES = {
    0: "real",
    1: "imaginary",
    2: "amplitude",
    3: "phase",
    4: "delta_real",
    5: "delta_imaginary",
    6: "delta_amplitude",
    7: "delta_phase",
    8: "complex_ratio_phase",
    9: "complex_difference_magnitude",
    10: "conjugate_product_phase_difference",
    11: "cosine_phase_difference",
    12: "sine_phase_difference",
}


def build_original_difference_combinations(
    *,
    original_channels: Sequence[int] = (0, 1, 2, 3),
    difference_channels: Sequence[int] = (4, 5, 6, 7, 8, 9),
) -> List[tuple[int, int]]:
    """Return every pair containing one original and one difference channel."""
    channels = (*original_channels, *difference_channels)
    unknown_channels = [channel for channel in channels if channel not in CHANNEL_NAMES]
    if unknown_channels:
        raise ValueError(f"Unknown channel indices: {unknown_channels}")
    if set(original_channels) & set(difference_channels):
        raise ValueError("Original and difference channel groups must not overlap")
    return list(product(original_channels, difference_channels))


def build_difference_pair_combinations(
    *,
    difference_channels: Sequence[int] = (4, 5, 6, 7, 8, 9, 10),
) -> List[tuple[int, int]]:
    """Return every unique unordered pair of selected difference channels."""
    unknown_channels = [
        channel for channel in difference_channels if channel not in CHANNEL_NAMES
    ]
    if unknown_channels:
        raise ValueError(f"Unknown channel indices: {unknown_channels}")
    if len(set(difference_channels)) != len(difference_channels):
        raise ValueError("difference_channels must not contain duplicates")
    return list(combinations(difference_channels, 2))


def _summary_fields(
    summary: Dict[str, object],
    elapsed_seconds: float,
) -> Dict[str, object]:
    return {
        "mean_measurement_accuracy": float(summary["mean_measurement_accuracy"]),
        "std_measurement_accuracy": float(summary["std_measurement_accuracy"]),
        "pooled_measurement_accuracy": float(summary["pooled_measurement_accuracy"]),
        "measurement_macro_f1": float(summary["measurement_macro_f1"]),
        "mean_sample_accuracy": float(summary["mean_sample_accuracy"]),
        "std_sample_accuracy": float(summary["std_sample_accuracy"]),
        "pooled_sample_accuracy": float(summary["pooled_sample_accuracy"]),
        "sample_macro_f1": float(summary["sample_macro_f1"]),
        "measurement_confusion_matrix": np.asarray(
            summary["measurement_confusion_matrix"], dtype=np.int64
        ).copy(),
        "sample_confusion_matrix": np.asarray(
            summary["sample_confusion_matrix"], dtype=np.int64
        ).copy(),
        "measurement_recall_matrix": np.asarray(
            summary["measurement_recall_matrix"], dtype=np.float64
        ).copy(),
        "sample_recall_matrix": np.asarray(
            summary["sample_recall_matrix"], dtype=np.float64
        ).copy(),
        "idx_to_label": dict(summary["idx_to_label"]),
        "elapsed_seconds": float(elapsed_seconds),
    }


def build_channel_comparison_result(
    *,
    original_channel: int,
    difference_channel: int,
    summary: Dict[str, object],
    elapsed_seconds: float,
) -> Dict[str, object]:
    """Create the compact result retained after one channel pair is trained."""
    return {
        "channels": (original_channel, difference_channel),
        "original_channel": original_channel,
        "original_name": CHANNEL_NAMES[original_channel],
        "difference_channel": difference_channel,
        "difference_name": CHANNEL_NAMES[difference_channel],
        **_summary_fields(summary, elapsed_seconds),
    }


def build_difference_pair_result(
    *,
    first_channel: int,
    second_channel: int,
    summary: Dict[str, object],
    elapsed_seconds: float,
) -> Dict[str, object]:
    """Create the retained result for one unordered difference-channel pair."""
    if first_channel not in CHANNEL_NAMES or second_channel not in CHANNEL_NAMES:
        raise ValueError(
            f"Unknown channel pair: ({first_channel}, {second_channel})"
        )
    if first_channel == second_channel:
        raise ValueError("A channel cannot be paired with itself")
    return {
        "channels": (first_channel, second_channel),
        "first_channel": first_channel,
        "first_name": CHANNEL_NAMES[first_channel],
        "second_channel": second_channel,
        "second_name": CHANNEL_NAMES[second_channel],
        **_summary_fields(summary, elapsed_seconds),
    }


def build_single_channel_result(
    *,
    channel: int,
    summary: Dict[str, object],
    elapsed_seconds: float,
) -> Dict[str, object]:
    """Create the compact result retained after one channel is trained."""
    if channel not in CHANNEL_NAMES:
        raise ValueError(f"Unknown channel index: {channel}")
    return {
        "channel": channel,
        "channel_name": CHANNEL_NAMES[channel],
        **_summary_fields(summary, elapsed_seconds),
    }


def _sort_results(
    results: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    return sorted(
        results,
        key=lambda result: (
            float(result["pooled_sample_accuracy"]),
            float(result["sample_macro_f1"]),
            float(result["pooled_measurement_accuracy"]),
        ),
        reverse=True,
    )


def sort_channel_comparison_results(
    results: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Rank channel pairs primarily by pooled sample accuracy."""
    return _sort_results(results)


def sort_difference_pair_results(
    results: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Rank difference-channel pairs primarily by pooled sample accuracy."""
    return _sort_results(results)


def sort_single_channel_results(
    results: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Rank individual channels primarily by pooled sample accuracy."""
    return _sort_results(results)


def print_ranked_channel_results(results: Sequence[Dict[str, object]]) -> None:
    ranked_results = sort_channel_comparison_results(results)
    print("\nRanked two-channel combinations (primary metric: pooled sample accuracy)")
    print(
        f"{'Rank':>4}  {'Channels':>9}  {'Original':<10}  {'Difference':<29}  "
        f"{'Sample acc':>10}  {'Sample F1':>9}  {'Meas. acc':>10}"
    )
    print("-" * 100)
    for rank, result in enumerate(ranked_results, start=1):
        print(
            f"{rank:>4}  {str(result['channels']):>9}  "
            f"{str(result['original_name']):<10}  "
            f"{str(result['difference_name']):<29}  "
            f"{float(result['pooled_sample_accuracy']):>10.4f}  "
            f"{float(result['sample_macro_f1']):>9.4f}  "
            f"{float(result['pooled_measurement_accuracy']):>10.4f}"
        )


def print_ranked_difference_pair_results(
    results: Sequence[Dict[str, object]],
) -> None:
    ranked_results = sort_difference_pair_results(results)
    print("\nRanked difference-channel pairs (primary metric: pooled sample accuracy)")
    print(
        f"{'Rank':>4}  {'Channels':>9}  {'First difference':<36}  "
        f"{'Second difference':<36}  {'Sample acc':>10}  "
        f"{'Sample F1':>9}  {'Meas. acc':>10}"
    )
    print("-" * 136)
    for rank, result in enumerate(ranked_results, start=1):
        print(
            f"{rank:>4}  {str(result['channels']):>9}  "
            f"{str(result['first_name']):<36}  "
            f"{str(result['second_name']):<36}  "
            f"{float(result['pooled_sample_accuracy']):>10.4f}  "
            f"{float(result['sample_macro_f1']):>9.4f}  "
            f"{float(result['pooled_measurement_accuracy']):>10.4f}"
        )


def print_ranked_single_channel_results(
    results: Sequence[Dict[str, object]],
) -> None:
    ranked_results = sort_single_channel_results(results)
    print("\nRanked individual channels (primary metric: pooled sample accuracy)")
    print(
        f"{'Rank':>4}  {'Channel':>7}  {'Name':<29}  "
        f"{'Sample acc':>10}  {'Sample F1':>9}  {'Meas. acc':>10}"
    )
    print("-" * 80)
    for rank, result in enumerate(ranked_results, start=1):
        print(
            f"{rank:>4}  {int(result['channel']):>7}  "
            f"{str(result['channel_name']):<29}  "
            f"{float(result['pooled_sample_accuracy']):>10.4f}  "
            f"{float(result['sample_macro_f1']):>9.4f}  "
            f"{float(result['pooled_measurement_accuracy']):>10.4f}"
        )


def _matrix_columns(
    prefix: str,
    matrix: np.ndarray,
    idx_to_label: Dict[int, str],
) -> Dict[str, float | int]:
    columns: Dict[str, float | int] = {}
    for true_index in range(matrix.shape[0]):
        for predicted_index in range(matrix.shape[1]):
            true_label = idx_to_label[true_index]
            predicted_label = idx_to_label[predicted_index]
            columns[
                f"{prefix}_true_{true_label}_pred_{predicted_label}"
            ] = matrix[true_index, predicted_index].item()
    return columns


def _metric_columns(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "mean_sample_accuracy": result["mean_sample_accuracy"],
        "std_sample_accuracy": result["std_sample_accuracy"],
        "pooled_sample_accuracy": result["pooled_sample_accuracy"],
        "sample_macro_f1": result["sample_macro_f1"],
        "mean_measurement_accuracy": result["mean_measurement_accuracy"],
        "std_measurement_accuracy": result["std_measurement_accuracy"],
        "pooled_measurement_accuracy": result["pooled_measurement_accuracy"],
        "measurement_macro_f1": result["measurement_macro_f1"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _add_matrix_columns(
    row: Dict[str, object],
    result: Dict[str, object],
) -> None:
    idx_to_label = result["idx_to_label"]
    row.update(
        _matrix_columns(
            "sample_confusion",
            np.asarray(result["sample_confusion_matrix"]),
            idx_to_label,
        )
    )
    row.update(
        _matrix_columns(
            "sample_recall",
            np.asarray(result["sample_recall_matrix"]),
            idx_to_label,
        )
    )


def _write_rows(rows: List[Dict[str, object]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def save_channel_comparison_csv(
    results: Sequence[Dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Save ranked pair metrics plus all sample confusion/recall cells."""
    ranked_results = sort_channel_comparison_results(results)
    if not ranked_results:
        raise ValueError("results must not be empty")
    rows = []
    for rank, result in enumerate(ranked_results, start=1):
        row = {
            "rank": rank,
            "original_channel": result["original_channel"],
            "original_name": result["original_name"],
            "difference_channel": result["difference_channel"],
            "difference_name": result["difference_name"],
            **_metric_columns(result),
        }
        _add_matrix_columns(row, result)
        rows.append(row)
    return _write_rows(rows, output_path)


def save_difference_pair_comparison_csv(
    results: Sequence[Dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Save ranked difference-pair metrics and sample matrix cells."""
    ranked_results = sort_difference_pair_results(results)
    if not ranked_results:
        raise ValueError("results must not be empty")
    rows = []
    for rank, result in enumerate(ranked_results, start=1):
        row = {
            "rank": rank,
            "first_channel": result["first_channel"],
            "first_name": result["first_name"],
            "second_channel": result["second_channel"],
            "second_name": result["second_name"],
            **_metric_columns(result),
        }
        _add_matrix_columns(row, result)
        rows.append(row)
    return _write_rows(rows, output_path)


def save_single_channel_comparison_csv(
    results: Sequence[Dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Save ranked individual-channel metrics and sample matrix cells."""
    ranked_results = sort_single_channel_results(results)
    if not ranked_results:
        raise ValueError("results must not be empty")
    rows = []
    for rank, result in enumerate(ranked_results, start=1):
        row = {
            "rank": rank,
            "channel": result["channel"],
            "channel_name": result["channel_name"],
            **_metric_columns(result),
        }
        _add_matrix_columns(row, result)
        rows.append(row)
    return _write_rows(rows, output_path)


def plot_channel_metric_heatmap(
    results: Sequence[Dict[str, object]],
    *,
    metric: str = "pooled_sample_accuracy",
    output_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an original-channel by difference-channel metric heatmap."""
    if not results:
        raise ValueError("results must not be empty")
    original_channels = sorted({int(result["original_channel"]) for result in results})
    difference_channels = sorted(
        {int(result["difference_channel"]) for result in results}
    )
    matrix = np.full(
        (len(original_channels), len(difference_channels)), np.nan, dtype=np.float64
    )
    original_lookup = {value: index for index, value in enumerate(original_channels)}
    difference_lookup = {
        value: index for index, value in enumerate(difference_channels)
    }
    for result in results:
        matrix[
            original_lookup[int(result["original_channel"])],
            difference_lookup[int(result["difference_channel"])],
        ] = float(result[metric])

    title = _metric_title(metric)
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03).set_label(title)
    ax.set_title(f"Two-Channel Comparison: {title}")
    ax.set_xlabel("Difference channel")
    ax.set_ylabel("Original channel")
    ax.set_xticks(np.arange(len(difference_channels)))
    ax.set_yticks(np.arange(len(original_channels)))
    ax.set_xticklabels(
        [f"{value}: {CHANNEL_NAMES[value]}" for value in difference_channels],
        rotation=35,
        ha="right",
    )
    ax.set_yticklabels(
        [f"{value}: {CHANNEL_NAMES[value]}" for value in original_channels]
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    _finish_plot(fig, output_path, show)
    return fig, ax


def plot_difference_pair_metric_heatmap(
    results: Sequence[Dict[str, object]],
    *,
    metric: str = "pooled_sample_accuracy",
    output_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a symmetric metric heatmap for unordered difference-channel pairs."""
    if not results:
        raise ValueError("results must not be empty")
    channels = sorted(
        {
            int(channel)
            for result in results
            for channel in (
                result["first_channel"],
                result["second_channel"],
            )
        }
    )
    matrix = np.full((len(channels), len(channels)), np.nan, dtype=np.float64)
    channel_lookup = {value: index for index, value in enumerate(channels)}
    for result in results:
        first = channel_lookup[int(result["first_channel"])]
        second = channel_lookup[int(result["second_channel"])]
        value = float(result[metric])
        matrix[first, second] = value
        matrix[second, first] = value

    title = _metric_title(metric)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03).set_label(title)
    ax.set_title(f"Difference-Channel Pair Comparison: {title}")
    ax.set_xlabel("Difference channel")
    ax.set_ylabel("Difference channel")
    ax.set_xticks(np.arange(len(channels)))
    ax.set_yticks(np.arange(len(channels)))
    labels = [f"{value}: {CHANNEL_NAMES[value]}" for value in channels]
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}",
                    ha="center",
                    va="center",
                )
    _finish_plot(fig, output_path, show)
    return fig, ax


def _metric_title(metric: str) -> str:
    return {
        "pooled_sample_accuracy": "Pooled Sample Accuracy",
        "sample_macro_f1": "Sample Macro F1",
        "pooled_measurement_accuracy": "Pooled Measurement Accuracy",
    }.get(metric, metric.replace("_", " ").title())


def _finish_plot(
    fig: plt.Figure,
    output_path: str | Path | None,
    show: bool,
) -> None:
    fig.tight_layout()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single_channel_metric(
    results: Sequence[Dict[str, object]],
    *,
    metric: str = "pooled_sample_accuracy",
    output_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one metric for individual channels in descending order."""
    if not results:
        raise ValueError("results must not be empty")
    ranked_results = sorted(
        results, key=lambda result: float(result[metric]), reverse=True
    )
    values = np.asarray([float(result[metric]) for result in ranked_results])
    labels = [
        f"{int(result['channel'])}: {result['channel_name']}"
        for result in ranked_results
    ]
    title = _metric_title(metric)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(np.arange(len(values)), values, color="steelblue")
    ax.set_title(f"Individual-Channel Comparison: {title}")
    ax.set_xlabel("Channel")
    ax.set_ylabel(title)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _finish_plot(fig, output_path, show)
    return fig, ax
