from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IRS_Insecticide_Residual.Utility_Functions import Preprocessing

from .classifier import DemoClassificationResult
from .config import RAW_EXAMPLES_PER_CLASS, RAW_SUBPLOTS_PER_CLASS, SIGNAL_SEGMENTS


CLASS_COLORS = {
    "LOW": "#2F80ED",
    "TARGET": "#27AE60",
    "HIGH": "#EB5757",
}


def select_raw_sample_indices(
    raw_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
    examples_per_class: int = RAW_SUBPLOTS_PER_CLASS,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    sample_indices_by_class = {}
    for label in class_order:
        values = raw_by_class[label]
        sample_count = min(examples_per_class, len(values))
        sample_indices_by_class[label] = rng.choice(
            len(values),
            size=sample_count,
            replace=False,
        )
    return sample_indices_by_class


def find_abnormal_spike_examples(
    raw_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
    n_examples: int = 3,
    radius: int = 3,
    k: float = 4.0,
    min_threshold: float = 1000.0,
) -> list[dict[str, object]]:
    """Find raw signal samples most changed by the same spike filter used in preprocessing."""
    candidates = []

    for label in class_order:
        values = raw_by_class[label].to_numpy(dtype=np.float32, copy=False)
        despiked_values = Preprocessing.fast_spike_filter(
            values,
            radius=radius,
            k=k,
            min_threshold=min_threshold,
        )
        changes = np.abs(values - despiked_values)
        if changes.size == 0:
            continue

        point_indices = np.argmax(changes, axis=1)
        sample_changes = changes[np.arange(changes.shape[0]), point_indices]
        changed_sample_indices = np.flatnonzero(sample_changes > 0.0)

        for sample_idx in changed_sample_indices:
            point_idx = int(point_indices[int(sample_idx)])
            candidates.append(
                {
                    "label": label,
                    "sample_idx": int(sample_idx),
                    "point_idx": point_idx,
                    "raw_signal": values[int(sample_idx)].copy(),
                    "despiked_signal": despiked_values[int(sample_idx)].copy(),
                    "raw_value": float(values[int(sample_idx), point_idx]),
                    "despiked_value": float(despiked_values[int(sample_idx), point_idx]),
                    "change": float(sample_changes[int(sample_idx)]),
                    "radius": radius,
                    "k": k,
                    "min_threshold": min_threshold,
                }
            )

    candidates.sort(key=lambda item: float(item["change"]), reverse=True)
    examples = candidates[:n_examples]

    if not examples:
        raise ValueError(
            "No abnormal spike was changed by the configured spike filter. "
            "Try lowering min_threshold for this teaching example."
        )

    return examples


def find_abnormal_spike_example(
    raw_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
    radius: int = 3,
    k: float = 4.0,
    min_threshold: float = 1000.0,
) -> dict[str, object]:
    """Backward-compatible wrapper returning the single strongest spike example."""
    return find_abnormal_spike_examples(
        raw_by_class=raw_by_class,
        class_order=class_order,
        n_examples=1,
        radius=radius,
        k=k,
        min_threshold=min_threshold,
    )[0]


def plot_spike_removal_examples(
    spike_examples: Sequence[dict[str, object]],
    output_path: Path,
) -> Path:
    example_count = len(spike_examples)
    if example_count == 0:
        raise ValueError("spike_examples must contain at least one example.")

    fig, axes = plt.subplots(
        example_count,
        2,
        figsize=(14, 3.15 * example_count),
        sharex=True,
        squeeze=False,
    )

    for row_idx, spike_example in enumerate(spike_examples):
        raw_signal = np.asarray(spike_example["raw_signal"], dtype=np.float32)
        despiked_signal = np.asarray(spike_example["despiked_signal"], dtype=np.float32)
        label = str(spike_example["label"])
        sample_idx = int(spike_example["sample_idx"])
        color = CLASS_COLORS.get(label, "#777777")
        changed_points = np.flatnonzero(np.abs(raw_signal - despiked_signal) > 0.0)
        x_full = np.arange(raw_signal.size)

        raw_ax = axes[row_idx, 0]
        clean_ax = axes[row_idx, 1]

        raw_ax.plot(x_full, raw_signal, color="#555555", linewidth=1.0, label="raw")
        raw_ax.scatter(
            changed_points,
            raw_signal[changed_points],
            color="#EB5757",
            s=52,
            zorder=5,
            label="removed spike",
        )
        raw_ax.set_title(f"{label} sample {sample_idx}: raw measurement")
        raw_ax.set_ylabel("Sensor response")

        clean_ax.plot(
            x_full,
            despiked_signal,
            color=color,
            linewidth=1.8,
            label="after spike removal",
        )
        clean_ax.set_title(f"{label} sample {sample_idx}: after spike removal")
        clean_ax.set_ylabel("Sensor response")

        if row_idx == 0:
            raw_ax.legend(loc="best")
            clean_ax.legend(loc="best")
        if row_idx == example_count - 1:
            raw_ax.set_xlabel("Measurement point")
            clean_ax.set_xlabel("Measurement point")

    fig.suptitle("3. Spike Removal: Similar Artifacts Appear in Multiple Measurements")
    return _save(fig, output_path)


def plot_individual_raw_signal_subplots(
    raw_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
    output_path: Path,
    examples_per_class: int = RAW_SUBPLOTS_PER_CLASS,
    sample_indices_by_class: dict[str, Sequence[int]] | None = None,
) -> Path:
    fig, axes = plt.subplots(
        len(class_order),
        examples_per_class,
        figsize=(3.2 * examples_per_class, 2.4 * len(class_order)),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(len(class_order), examples_per_class)
    if sample_indices_by_class is None:
        sample_indices_by_class = select_raw_sample_indices(
            raw_by_class,
            class_order,
            examples_per_class=examples_per_class,
        )

    selected = []
    for row_idx, label in enumerate(class_order):
        values = raw_by_class[label].to_numpy(dtype=np.float32, copy=False)
        sample_indices = np.asarray(sample_indices_by_class[label], dtype=int)[:examples_per_class]
        sample_count = len(sample_indices)
        color = CLASS_COLORS.get(label, None)

        for col_idx in range(examples_per_class):
            ax = axes[row_idx, col_idx]
            if col_idx >= sample_count:
                ax.axis("off")
                continue

            sample_idx = int(sample_indices[col_idx])
            selected.append(values[sample_idx])
            ax.plot(values[sample_idx], color=color, linewidth=1.2)
            ax.set_title(f"{label} sample {sample_idx}", fontsize=9)
            ax.tick_params(labelsize=8)
            if col_idx == 0:
                ax.set_ylabel("Sensor response")
            if row_idx == len(class_order) - 1:
                ax.set_xlabel("Measurement point")

    if selected:
        selected_values = np.concatenate(selected)
        y_min = float(np.min(selected_values))
        y_max = float(np.max(selected_values))
        if y_max > y_min:
            y_margin = 0.08 * (y_max - y_min)
            for ax in axes.flat:
                if ax.has_data():
                    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    fig.suptitle("1. Individual Raw Measurements: Noisy Signals From Real Samples")
    return _save(fig, output_path)


def plot_raw_signals(
    raw_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
    output_path: Path,
    examples_per_class: int = RAW_EXAMPLES_PER_CLASS,
    sample_indices_by_class: dict[str, Sequence[int]] | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    if sample_indices_by_class is None:
        sample_indices_by_class = select_raw_sample_indices(
            raw_by_class,
            class_order,
            examples_per_class=examples_per_class,
        )

    for label in class_order:
        values = raw_by_class[label].to_numpy(dtype=np.float32, copy=False)
        sample_indices = np.asarray(sample_indices_by_class[label], dtype=int)[:examples_per_class]
        color = CLASS_COLORS.get(label, None)

        for sample_idx in sample_indices:
            axes[0].plot(
                values[sample_idx],
                color=color,
                alpha=0.50,
                linewidth=1.2,
            )

        x_axis = np.arange(values.shape[1])
        axes[1].plot(
            x_axis,
            values[sample_indices].mean(axis=0),
            color=color,
            linewidth=2.3,
            label=f"{label} average",
        )

    axes[0].set_title("Noisy individual raw measurements")
    axes[0].set_xlabel("Measurement point")
    axes[0].set_ylabel("Sensor response")
    _add_class_legend(ax=axes[0], class_order=class_order)

    axes[1].set_title("Raw class averages across the full measurement")
    axes[1].set_xlabel("Measurement point")
    axes[1].set_ylabel("Average sensor response")
    axes[1].legend(loc="best")

    fig.suptitle("2. Raw Sensor Signals: Overlay Reveals Class Patterns")
    return _save(fig, output_path)


def plot_spike_removal_example(
    spike_example: dict[str, object],
    output_path: Path,
) -> Path:
    """Backward-compatible wrapper plotting one before/after spike example."""
    return plot_spike_removal_examples([spike_example], output_path)


def plot_processed_class_average_before_after_slicing(
    full_processed_no_sqrt_by_class: dict[str, pd.DataFrame],
    full_processed_by_class: dict[str, pd.DataFrame],
    sliced_processed_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=False)
    full_length = next(iter(full_processed_by_class.values())).shape[1]

    for ax, data_by_class, title, x_label, y_label in (
        (
            axes[0],
            full_processed_no_sqrt_by_class,
            "Before slicing: no sqrt transform",
            "Full measurement point",
            "Processed sensor response",
        ),
        (
            axes[1],
            full_processed_by_class,
            "Before slicing: with sqrt transform",
            "Full measurement point",
            "Processed sqrt sensor response",
        ),
        (
            axes[2],
            sliced_processed_by_class,
            "After slicing: with sqrt transform",
            "Sliced measurement point",
            "Processed sqrt sensor response",
        ),
    ):
        if ax in (axes[0], axes[1]):
            _shade_removed_signal_regions(ax, full_length=full_length)

        for label in class_order:
            values = data_by_class[label].to_numpy(dtype=np.float32, copy=False)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            x_axis = np.arange(values.shape[1])
            color = CLASS_COLORS.get(label, "#777777")
            ax.plot(x_axis, mean, color=color, linewidth=2.2, label=label)
            ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="best")

    slice_boundary = SIGNAL_SEGMENTS[0][1] - SIGNAL_SEGMENTS[0][0]
    if len(SIGNAL_SEGMENTS) > 1 and slice_boundary > 0:
        axes[2].axvline(
            slice_boundary,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )

    fig.suptitle("4. Processed Class Average and Variation: Sqrt Transform and Slicing")
    return _save(fig, output_path)


def _shade_removed_signal_regions(ax, full_length: int) -> None:
    ordered_segments = sorted(SIGNAL_SEGMENTS)
    cursor = 0
    label_added = False

    for start, end in ordered_segments:
        safe_start = max(0, min(int(start), full_length))
        safe_end = max(0, min(int(end), full_length))
        if cursor < safe_start:
            ax.axvspan(
                cursor,
                safe_start,
                color="#999999",
                alpha=0.14,
                label="removed by slicing" if not label_added else None,
            )
            label_added = True
        cursor = max(cursor, safe_end)

    if cursor < full_length:
        ax.axvspan(
            cursor,
            full_length,
            color="#999999",
            alpha=0.14,
            label="removed by slicing" if not label_added else None,
        )


def plot_processing_comparison(
    sliced_raw_by_class: dict[str, pd.DataFrame],
    processed_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
    output_path: Path,
    sample_indices_by_class: dict[str, Sequence[int]] | None = None,
    examples_per_class: int = RAW_SUBPLOTS_PER_CLASS,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=False, sharey=False)
    if sample_indices_by_class is None:
        sample_indices_by_class = select_raw_sample_indices(
            sliced_raw_by_class,
            class_order,
            examples_per_class=examples_per_class,
        )

    for label in class_order:
        raw_values = sliced_raw_by_class[label].to_numpy(dtype=np.float32, copy=False)
        processed_values = processed_by_class[label].to_numpy(dtype=np.float32, copy=False)
        sample_indices = np.asarray(sample_indices_by_class[label], dtype=int)[:examples_per_class]
        color = CLASS_COLORS.get(label, None)

        for sample_idx in sample_indices:
            axes[0, 0].plot(
                raw_values[sample_idx],
                color=color,
                alpha=0.40,
                linewidth=1.0,
            )
            axes[0, 1].plot(
                processed_values[sample_idx],
                color=color,
                alpha=0.55,
                linewidth=1.2,
            )

    axes[0, 0].set_title("Same samples before processing")
    axes[0, 0].set_xlabel("Selected raw measurement point")
    axes[0, 0].set_ylabel("Sensor response")
    _add_class_legend(axes[0, 0], class_order)

    axes[0, 1].set_title("Same samples after processing")
    axes[0, 1].set_xlabel("Processed measurement point")
    axes[0, 1].set_ylabel("Processed response")
    _add_class_legend(axes[0, 1], class_order)

    for ax, data_by_class, title, y_label in (
        (axes[1, 0], sliced_raw_by_class, "Raw class average +/- variation", "Sensor response"),
        (axes[1, 1], processed_by_class, "Processed class average +/- variation", "Processed response"),
    ):
        for label in class_order:
            values = data_by_class[label].to_numpy(dtype=np.float32, copy=False)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            x_axis = np.arange(values.shape[1])
            color = CLASS_COLORS.get(label, None)
            ax.plot(x_axis, mean, color=color, linewidth=2.0, label=label)
            ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Measurement point")
        ax.set_ylabel(y_label)

    axes[1, 0].legend(loc="best")
    axes[1, 1].legend(loc="best")
    fig.suptitle("5. Processing Reduces Raw Noise and Makes Class Patterns Easier to Compare")
    return _save(fig, output_path)


def plot_feature_map(
    result: DemoClassificationResult,
    output_path: Path,
    show_unknown: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6.5))

    for label in result.class_order:
        train_mask = result.y_train == label
        test_mask = result.y_test == label
        color = CLASS_COLORS.get(label, None)
        ax.scatter(
            result.x_train_map[train_mask, 0],
            result.x_train_map[train_mask, 1],
            s=38,
            color=color,
            alpha=0.55,
            label=f"{label} known",
        )
        ax.scatter(
            result.x_test_map[test_mask, 0],
            result.x_test_map[test_mask, 1],
            s=62,
            facecolor="none",
            edgecolor=color,
            linewidth=1.5,
            label=f"{label} test",
        )

    if show_unknown:
        ax.scatter(
            result.unknown_map_point[0],
            result.unknown_map_point[1],
            marker="*",
            s=260,
            color="#111111",
            edgecolor="white",
            linewidth=1.0,
            label="unknown example",
            zorder=5,
        )

    if result.method_name.startswith("CNN"):
        ax.set_title("5. Feature Map of CNN-Learned Signal Features")
    else:
        ax.set_title("5. Feature Map of Simple Signal Measurements")
    ax.set_xlabel("PCA feature 1")
    ax.set_ylabel("PCA feature 2")
    ax.legend(loc="best", fontsize=8)
    return _save(fig, output_path)


def plot_cnn_feature_learning_comparison(
    result: DemoClassificationResult,
    output_path: Path,
    show_unknown: bool = True,
) -> Path:
    if (
        result.pre_cnn_x_train_map is None
        or result.pre_cnn_x_test_map is None
        or result.pre_cnn_unknown_map_point is None
    ):
        raise ValueError("CNN feature comparison requires pre-CNN PCA coordinates.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=False, sharey=False)
    _plot_2d_feature_points(
        axes[0],
        x_train_map=result.pre_cnn_x_train_map,
        x_test_map=result.pre_cnn_x_test_map,
        unknown_map_point=result.pre_cnn_unknown_map_point,
        y_train=result.y_train,
        y_test=result.y_test,
        class_order=result.class_order,
        show_unknown=show_unknown,
    )
    _plot_2d_feature_points(
        axes[1],
        x_train_map=result.x_train_map,
        x_test_map=result.x_test_map,
        unknown_map_point=result.unknown_map_point,
        y_train=result.y_train,
        y_test=result.y_test,
        class_order=result.class_order,
        show_unknown=show_unknown,
    )

    axes[0].set_title("Before CNN: PCA of processed signals")
    axes[1].set_title("After CNN: PCA of learned features")
    for ax in axes:
        ax.set_xlabel("PCA feature 1")
        ax.set_ylabel("PCA feature 2")
    axes[1].legend(loc="best", fontsize=8)

    fig.suptitle("5. How CNN Learning Changes the Feature Space")
    return _save(fig, output_path)


def plot_feature_map_3d(
    result: DemoClassificationResult,
    output_path: Path,
    show_unknown: bool = True,
) -> Path:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    _plot_3d_feature_points(
        ax,
        x_train_map_3d=result.x_train_map_3d,
        x_test_map_3d=result.x_test_map_3d,
        unknown_map_point_3d=result.unknown_map_point_3d,
        y_train=result.y_train,
        y_test=result.y_test,
        class_order=result.class_order,
        show_unknown=show_unknown,
    )

    if result.method_name.startswith("CNN"):
        ax.set_title("6. 3D PCA Feature Map of CNN-Learned Signal Features")
    else:
        ax.set_title("6. 3D PCA Feature Map of Simple Signal Measurements")
    ax.set_xlabel("PCA feature 1")
    ax.set_ylabel("PCA feature 2")
    ax.set_zlabel("PCA feature 3")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, output_path)


def plot_cnn_feature_learning_comparison_3d(
    result: DemoClassificationResult,
    output_path: Path,
    show_unknown: bool = True,
) -> Path:
    if (
        result.pre_cnn_x_train_map_3d is None
        or result.pre_cnn_x_test_map_3d is None
        or result.pre_cnn_unknown_map_point_3d is None
    ):
        raise ValueError("CNN 3D feature comparison requires pre-CNN 3D PCA coordinates.")

    fig = plt.figure(figsize=(15, 7))
    axes = [
        fig.add_subplot(121, projection="3d"),
        fig.add_subplot(122, projection="3d"),
    ]

    _plot_3d_feature_points(
        axes[0],
        x_train_map_3d=result.pre_cnn_x_train_map_3d,
        x_test_map_3d=result.pre_cnn_x_test_map_3d,
        unknown_map_point_3d=result.pre_cnn_unknown_map_point_3d,
        y_train=result.y_train,
        y_test=result.y_test,
        class_order=result.class_order,
        show_unknown=show_unknown,
    )
    _plot_3d_feature_points(
        axes[1],
        x_train_map_3d=result.x_train_map_3d,
        x_test_map_3d=result.x_test_map_3d,
        unknown_map_point_3d=result.unknown_map_point_3d,
        y_train=result.y_train,
        y_test=result.y_test,
        class_order=result.class_order,
        show_unknown=show_unknown,
    )

    axes[0].set_title("Before CNN: PCA of processed signals")
    axes[1].set_title("After CNN: PCA of learned features")
    for ax in axes:
        ax.set_xlabel("PCA feature 1")
        ax.set_ylabel("PCA feature 2")
        ax.set_zlabel("PCA feature 3")
        ax.view_init(elev=24, azim=-58)
    axes[1].legend(loc="upper left", fontsize=8)

    fig.suptitle("6. 3D PCA: Before and After CNN Feature Learning")
    return _save(fig, output_path)


def plot_feature_map_3d_interactive_html(
    result: DemoClassificationResult,
    output_path: Path,
    show_unknown: bool = True,
) -> Path:
    import plotly.graph_objects as go

    fig = go.Figure()

    for label in result.class_order:
        train_mask = result.y_train == label
        test_mask = result.y_test == label
        color = CLASS_COLORS.get(label, "#777777")

        fig.add_trace(
            _plotly_3d_scatter(
                go=go,
                points=result.x_train_map_3d[train_mask],
                name=f"{label} known",
                color=color,
                opacity=0.42,
                size=4,
                symbol="circle",
            )
        )
        fig.add_trace(
            _plotly_3d_scatter(
                go=go,
                points=result.x_test_map_3d[test_mask],
                name=f"{label} test",
                color=color,
                opacity=0.95,
                size=5,
                symbol="circle-open",
            )
        )

    if show_unknown:
        fig.add_trace(
            _plotly_3d_scatter(
                go=go,
                points=np.asarray(result.unknown_map_point_3d, dtype=np.float32)[np.newaxis, :],
                name="unknown example",
                color="#111111",
                opacity=1.0,
                size=9,
                symbol="diamond",
            )
        )

    title = (
        "6. Interactive 3D PCA Feature Map of CNN-Learned Signal Features"
        if result.method_name.startswith("CNN")
        else "6. Interactive 3D PCA Feature Map of Simple Signal Measurements"
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "PCA feature 1",
            "yaxis_title": "PCA feature 2",
            "zaxis_title": "PCA feature 3",
        },
        legend={"orientation": "v"},
        margin={"l": 0, "r": 0, "b": 0, "t": 52},
        template="plotly_white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    return output_path


def plot_cnn_feature_learning_comparison_3d_interactive_html(
    result: DemoClassificationResult,
    output_path: Path,
    show_unknown: bool = True,
) -> Path:
    if (
        result.pre_cnn_x_train_map_3d is None
        or result.pre_cnn_x_test_map_3d is None
        or result.pre_cnn_unknown_map_point_3d is None
    ):
        raise ValueError("CNN 3D feature comparison requires pre-CNN 3D PCA coordinates.")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            "Before CNN: PCA of processed signals",
            "After CNN: PCA of learned features",
        ],
    )
    _add_plotly_3d_feature_points(
        fig=fig,
        go=go,
        row=1,
        col=1,
        x_train_map_3d=result.pre_cnn_x_train_map_3d,
        x_test_map_3d=result.pre_cnn_x_test_map_3d,
        unknown_map_point_3d=result.pre_cnn_unknown_map_point_3d,
        y_train=result.y_train,
        y_test=result.y_test,
        class_order=result.class_order,
        showlegend=False,
        show_unknown=show_unknown,
    )
    _add_plotly_3d_feature_points(
        fig=fig,
        go=go,
        row=1,
        col=2,
        x_train_map_3d=result.x_train_map_3d,
        x_test_map_3d=result.x_test_map_3d,
        unknown_map_point_3d=result.unknown_map_point_3d,
        y_train=result.y_train,
        y_test=result.y_test,
        class_order=result.class_order,
        showlegend=True,
        show_unknown=show_unknown,
    )

    fig.update_layout(
        title="6. Interactive 3D PCA: Before and After CNN Feature Learning",
        scene={
            "xaxis_title": "PCA feature 1",
            "yaxis_title": "PCA feature 2",
            "zaxis_title": "PCA feature 3",
        },
        scene2={
            "xaxis_title": "PCA feature 1",
            "yaxis_title": "PCA feature 2",
            "zaxis_title": "PCA feature 3",
        },
        legend={"orientation": "v"},
        margin={"l": 0, "r": 0, "b": 0, "t": 70},
        template="plotly_white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    return output_path


def _plot_3d_feature_points(
    ax,
    x_train_map_3d: np.ndarray,
    x_test_map_3d: np.ndarray,
    unknown_map_point_3d: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_order: Sequence[str],
    show_unknown: bool = True,
) -> None:
    for label in class_order:
        train_mask = y_train == label
        test_mask = y_test == label
        color = CLASS_COLORS.get(label, None)
        ax.scatter(
            x_train_map_3d[train_mask, 0],
            x_train_map_3d[train_mask, 1],
            x_train_map_3d[train_mask, 2],
            s=28,
            color=color,
            alpha=0.42,
            label=f"{label} known",
        )
        ax.scatter(
            x_test_map_3d[test_mask, 0],
            x_test_map_3d[test_mask, 1],
            x_test_map_3d[test_mask, 2],
            s=52,
            facecolor="none",
            edgecolor=color,
            linewidth=1.3,
            label=f"{label} test",
        )

    if show_unknown:
        ax.scatter(
            unknown_map_point_3d[0],
            unknown_map_point_3d[1],
            unknown_map_point_3d[2],
            marker="*",
            s=260,
            color="#111111",
            edgecolor="white",
            linewidth=1.0,
            label="unknown example",
            zorder=5,
        )


def _add_plotly_3d_feature_points(
    fig,
    go,
    row: int,
    col: int,
    x_train_map_3d: np.ndarray,
    x_test_map_3d: np.ndarray,
    unknown_map_point_3d: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_order: Sequence[str],
    showlegend: bool,
    show_unknown: bool = True,
) -> None:
    for label in class_order:
        train_mask = y_train == label
        test_mask = y_test == label
        color = CLASS_COLORS.get(label, "#777777")
        fig.add_trace(
            _plotly_3d_scatter(
                go=go,
                points=x_train_map_3d[train_mask],
                name=f"{label} known",
                color=color,
                opacity=0.42,
                size=4,
                symbol="circle",
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            _plotly_3d_scatter(
                go=go,
                points=x_test_map_3d[test_mask],
                name=f"{label} test",
                color=color,
                opacity=0.95,
                size=5,
                symbol="circle-open",
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )

    if show_unknown:
        fig.add_trace(
            _plotly_3d_scatter(
                go=go,
                points=np.asarray(unknown_map_point_3d, dtype=np.float32)[np.newaxis, :],
                name="unknown example",
                color="#111111",
                opacity=1.0,
                size=9,
                symbol="diamond",
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )


def _plotly_3d_scatter(
    go,
    points: np.ndarray,
    name: str,
    color: str,
    opacity: float,
    size: int,
    symbol: str,
    showlegend: bool = True,
):
    points = np.asarray(points, dtype=np.float32)
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        showlegend=showlegend,
        marker={
            "color": color,
            "opacity": opacity,
            "size": size,
            "symbol": symbol,
        },
    )


def plot_classifier_result_comparison(
    knn_result: DemoClassificationResult,
    cnn_result: DemoClassificationResult,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.3),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.72]},
    )

    _plot_confusion_matrix_recall(
        axes[0],
        confusion_count=knn_result.confusion_count,
        class_order=knn_result.class_order,
        title=f"Original accuracy: {knn_result.test_accuracy:.3f}",
    )
    _plot_confusion_matrix_recall(
        axes[1],
        confusion_count=cnn_result.confusion_count,
        class_order=cnn_result.class_order,
        title=f"CNN accuracy: {cnn_result.test_accuracy:.3f}",
    )

    names = ["Original", "CNN"]
    accuracies = [knn_result.test_accuracy, cnn_result.test_accuracy]
    bars = axes[2].bar(names, accuracies, color=["#777777", "#2F80ED"])
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Holdout accuracy")
    axes[2].set_title("Accuracy")
    for bar, accuracy in zip(bars, accuracies):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.025,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle("7. Classifier Results: Original vs CNN")
    return _save(fig, output_path)


def _plot_confusion_matrix_recall(
    ax,
    confusion_count: np.ndarray,
    class_order: Sequence[str],
    title: str,
) -> None:
    confusion_count = np.asarray(confusion_count, dtype=np.float32)
    row_totals = confusion_count.sum(axis=1, keepdims=True)
    recall_percent = np.divide(
        confusion_count,
        row_totals,
        out=np.zeros_like(confusion_count, dtype=np.float32),
        where=row_totals != 0,
    ) * 100.0
    image = ax.imshow(recall_percent, cmap="Blues", vmin=0.0, vmax=100.0)

    for row_idx in range(recall_percent.shape[0]):
        for col_idx in range(recall_percent.shape[1]):
            recall = float(recall_percent[row_idx, col_idx])
            text_color = "white" if recall > 55.0 else "#111111"
            ax.text(
                col_idx,
                row_idx,
                f"{recall:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    ax.set_xticks(np.arange(len(class_order)))
    ax.set_yticks(np.arange(len(class_order)))
    ax.set_xticklabels(class_order, rotation=35, ha="right")
    ax.set_yticklabels(class_order)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Recall (%)")


def _plot_2d_feature_points(
    ax,
    x_train_map: np.ndarray,
    x_test_map: np.ndarray,
    unknown_map_point: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_order: Sequence[str],
    show_unknown: bool = True,
) -> None:
    for label in class_order:
        train_mask = y_train == label
        test_mask = y_test == label
        color = CLASS_COLORS.get(label, None)
        ax.scatter(
            x_train_map[train_mask, 0],
            x_train_map[train_mask, 1],
            s=30,
            color=color,
            alpha=0.46,
            label=f"{label} known",
        )
        ax.scatter(
            x_test_map[test_mask, 0],
            x_test_map[test_mask, 1],
            s=54,
            facecolor="none",
            edgecolor=color,
            linewidth=1.3,
            label=f"{label} test",
        )

    if show_unknown:
        ax.scatter(
            unknown_map_point[0],
            unknown_map_point[1],
            marker="*",
            s=230,
            color="#111111",
            edgecolor="white",
            linewidth=1.0,
            label="unknown example",
            zorder=5,
        )


def plot_unknown_prediction(
    processed_by_class: dict[str, pd.DataFrame],
    result: DemoClassificationResult,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    unknown_signal = _global_signal_by_index(
        processed_by_class=processed_by_class,
        class_order=result.class_order,
        global_index=result.unknown_global_index,
    )

    axes[0].plot(unknown_signal, color="#111111", linewidth=2.0)
    axes[0].set_title("Unknown Measurement")
    axes[0].set_xlabel("Processed measurement point")
    axes[0].set_ylabel("Sensor response")

    x_pos = np.arange(len(result.class_order))
    bar_colors = [CLASS_COLORS.get(label, "#777777") for label in result.class_order]
    axes[1].bar(x_pos, result.unknown_prob, color=bar_colors)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(result.class_order)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Classifier confidence")
    axes[1].set_title(f"Prediction: {result.unknown_pred_label}")

    fig.suptitle(
        f"8. Classifying an Unknown Example "
        f"(true label: {result.unknown_true_label})"
    )
    return _save(fig, output_path)


def plot_unknown_classification_game_html(
    raw_by_class: dict[str, pd.DataFrame],
    processed_by_class: dict[str, pd.DataFrame],
    result: DemoClassificationResult,
    output_path: Path,
    max_unknown_candidates_per_class: int = 4,
) -> Path:
    from plotly.offline import get_plotlyjs

    unknown_samples = _build_unknown_candidate_payloads(
        raw_by_class=raw_by_class,
        processed_by_class=processed_by_class,
        result=result,
        max_per_class=max_unknown_candidates_per_class,
    )
    feature_traces_base = _plotly_2d_feature_traces(
        result=result,
        include_unknown=False,
        unknown_visible=False,
    )
    feature_traces_3d_base = _plotly_3d_feature_traces(
        result=result,
        include_unknown=False,
        unknown_visible=False,
    )
    payload = {
        "classOrder": list(result.class_order),
        "unknownSamples": unknown_samples,
        "featureTracesBase": feature_traces_base,
        "featureTraces3dBase": feature_traces_3d_base,
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>8. Unknown Sample Classification Game</title>
  <script>{get_plotlyjs()}</script>
  <style>
    :root {{
      --ink: #111111;
      --muted: #666666;
      --line: #d6d6d6;
      --blue: #2F80ED;
      --panel: #ffffff;
      --soft: #f5f6f8;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--ink);
      background: var(--soft);
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 28px;
      font-weight: 700;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }}
    button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 15px;
      cursor: pointer;
    }}
    button.primary {{
      background: var(--blue);
      border-color: var(--blue);
      color: white;
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.42;
    }}
    .status {{
      min-height: 24px;
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 340px;
      padding: 10px;
    }}
    .plot {{
      width: 100%;
      height: 330px;
    }}
    .feature-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
    }}
    #featurePlot2d,
    #featurePlot3d {{
      height: 520px;
    }}
    .prediction {{
      display: grid;
      grid-template-columns: minmax(240px, 0.42fr) minmax(0, 1fr);
      gap: 14px;
      align-items: stretch;
    }}
    .answer {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 18px;
      font-size: 20px;
    }}
    .answer strong {{
      display: block;
      margin-top: 10px;
      font-size: 32px;
    }}
    .hidden {{
      display: none;
    }}
  </style>
</head>
<body>
  <main>
    <h1>8. Unknown Sample Classification Game</h1>
    <div class="controls">
      <button id="loadRaw" class="primary">Load Unknown Sample</button>
      <button id="processSample" disabled>Process Sample</button>
      <button id="classifySample" disabled>Classify</button>
      <button id="revealTruth" disabled>Reveal True Label</button>
      <button id="resetGame">Reset</button>
    </div>
    <p id="status" class="status">Imagine we do not know what this sample is.</p>

    <section class="grid">
      <div id="rawPanel" class="panel hidden">
        <div id="rawPlot" class="plot"></div>
      </div>
      <div id="processedPanel" class="panel hidden">
        <div id="processedPlot" class="plot"></div>
      </div>
      <div id="featurePanel" class="panel wide hidden">
        <div class="feature-grid">
          <div id="featurePlot2d" class="plot"></div>
          <div id="featurePlot3d" class="plot"></div>
        </div>
      </div>
      <div id="predictionPanel" class="prediction wide hidden">
        <div class="answer">
          Prediction
          <strong id="predictionText"></strong>
          <div id="truthText" class="hidden"></div>
        </div>
        <div class="panel">
          <div id="confidencePlot" class="plot"></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const data = {json.dumps(payload)};
    const rawPanel = document.getElementById("rawPanel");
    const processedPanel = document.getElementById("processedPanel");
    const featurePanel = document.getElementById("featurePanel");
    const predictionPanel = document.getElementById("predictionPanel");
    const statusEl = document.getElementById("status");
    const predictionText = document.getElementById("predictionText");
    const truthText = document.getElementById("truthText");
    let activeSample = null;
    let previousSampleId = null;

    function signalLayout(title, xTitle, yTitle) {{
      return {{
        title,
        xaxis: {{ title: xTitle }},
        yaxis: {{ title: yTitle }},
        margin: {{ l: 64, r: 18, b: 55, t: 48 }},
        template: "plotly_white",
      }};
    }}

    function featureLayout() {{
      return {{
        title: "2D CNN-learned feature map",
        xaxis: {{ title: "PCA feature 1" }},
        yaxis: {{ title: "PCA feature 2" }},
        margin: {{ l: 64, r: 18, b: 55, t: 50 }},
        template: "plotly_white",
        legend: {{ orientation: "v" }},
      }};
    }}

    function feature3dLayout() {{
      return {{
        title: "3D CNN-learned feature map",
        scene: {{
          xaxis: {{ title: "PCA feature 1" }},
          yaxis: {{ title: "PCA feature 2" }},
          zaxis: {{ title: "PCA feature 3" }},
        }},
        margin: {{ l: 0, r: 0, b: 0, t: 50 }},
        template: "plotly_white",
        showlegend: false,
      }};
    }}

    function confidenceLayout() {{
      return {{
        title: "Confidence for each class",
        xaxis: {{ title: "Class" }},
        yaxis: {{ title: "Confidence", range: [0, 1] }},
        margin: {{ l: 64, r: 18, b: 55, t: 48 }},
        template: "plotly_white",
      }};
    }}

    function chooseUnknownSample() {{
      let candidates = data.unknownSamples;
      if (candidates.length > 1 && previousSampleId !== null) {{
        candidates = candidates.filter(sample => sample.id !== previousSampleId);
      }}
      activeSample = candidates[Math.floor(Math.random() * candidates.length)];
      previousSampleId = activeSample.id;
      return activeSample;
    }}

    function cloneTraces(traces) {{
      return traces.map(trace => JSON.parse(JSON.stringify(trace)));
    }}

    function unknown2dTrace(sample) {{
      return {{
        type: "scatter",
        mode: "markers",
        name: "unknown sample",
        x: [sample.mapPoint[0]],
        y: [sample.mapPoint[1]],
        marker: {{
          color: "#111111",
          size: 18,
          symbol: "star",
          line: {{ color: "white", width: 1.2 }},
        }},
      }};
    }}

    function unknown3dTrace(sample) {{
      return {{
        type: "scatter3d",
        mode: "markers",
        name: "unknown sample",
        x: [sample.mapPoint3d[0]],
        y: [sample.mapPoint3d[1]],
        z: [sample.mapPoint3d[2]],
        marker: {{
          color: "#111111",
          opacity: 1.0,
          size: 9,
          symbol: "diamond",
        }},
      }};
    }}

    function loadRaw() {{
      const sample = chooseUnknownSample();
      rawPanel.classList.remove("hidden");
      processedPanel.classList.add("hidden");
      featurePanel.classList.add("hidden");
      predictionPanel.classList.add("hidden");
      predictionText.textContent = "";
      truthText.textContent = "";
      truthText.classList.add("hidden");
      Plotly.newPlot("rawPlot", [{{
        type: "scatter",
        mode: "lines",
        x: sample.rawSignal.map((_, idx) => idx),
        y: sample.rawSignal,
        line: {{ color: "#111111", width: 1.4 }},
        name: "unknown raw signal",
      }}], signalLayout("Unknown sample: raw measurement", "Measurement point", "Sensor response"), {{ responsive: true, displaylogo: false }});
      statusEl.textContent = "The raw signal is visible, but the class label is hidden.";
      document.getElementById("processSample").disabled = false;
      document.getElementById("classifySample").disabled = true;
      document.getElementById("revealTruth").disabled = true;
    }}

    function processSample() {{
      if (!activeSample) return;
      processedPanel.classList.remove("hidden");
      featurePanel.classList.remove("hidden");
      Plotly.newPlot("processedPlot", [{{
        type: "scatter",
        mode: "lines",
        x: activeSample.processedSignal.map((_, idx) => idx),
        y: activeSample.processedSignal,
        line: {{ color: "#2F80ED", width: 2.0 }},
        name: "processed signal",
      }}], signalLayout("Same sample after processing", "Processed measurement point", "Processed response"), {{ responsive: true, displaylogo: false }});
      Plotly.newPlot(
        "featurePlot2d",
        [...cloneTraces(data.featureTracesBase), unknown2dTrace(activeSample)],
        featureLayout(),
        {{ responsive: true, displaylogo: false }}
      );
      Plotly.newPlot(
        "featurePlot3d",
        [...cloneTraces(data.featureTraces3dBase), unknown3dTrace(activeSample)],
        feature3dLayout(),
        {{ responsive: true, displaylogo: false }}
      );
      statusEl.textContent = "After processing, the unknown sample appears in both the 2D and 3D feature maps.";
      document.getElementById("classifySample").disabled = false;
    }}

    function classifySample() {{
      if (!activeSample) return;
      predictionPanel.classList.remove("hidden");
      predictionText.textContent = activeSample.predictedLabel;
      truthText.classList.add("hidden");
      Plotly.newPlot("confidencePlot", [{{
        type: "bar",
        x: data.classOrder,
        y: activeSample.probabilities,
        marker: {{ color: data.classOrder.map(label => ({json.dumps(CLASS_COLORS)}[label] || "#777777")) }},
        text: activeSample.probabilities.map(value => `${{(100 * value).toFixed(1)}}%`),
        textposition: "outside",
      }}], confidenceLayout(), {{ responsive: true, displaylogo: false }});
      statusEl.textContent = "The classifier has made a prediction. Now reveal the answer.";
      document.getElementById("revealTruth").disabled = false;
    }}

    function revealTruth() {{
      if (!activeSample) return;
      truthText.textContent = `True label: ${{activeSample.trueLabel}}`;
      truthText.classList.remove("hidden");
      statusEl.textContent = activeSample.predictedLabel === activeSample.trueLabel
        ? "The prediction matches the true label."
        : "The prediction does not match the true label.";
    }}

    function resetGame() {{
      rawPanel.classList.add("hidden");
      processedPanel.classList.add("hidden");
      featurePanel.classList.add("hidden");
      predictionPanel.classList.add("hidden");
      predictionText.textContent = "";
      truthText.textContent = "";
      truthText.classList.add("hidden");
      activeSample = null;
      statusEl.textContent = "Imagine we do not know what this sample is.";
      document.getElementById("processSample").disabled = true;
      document.getElementById("classifySample").disabled = true;
      document.getElementById("revealTruth").disabled = true;
    }}

    document.getElementById("loadRaw").addEventListener("click", loadRaw);
    document.getElementById("processSample").addEventListener("click", processSample);
    document.getElementById("classifySample").addEventListener("click", classifySample);
    document.getElementById("revealTruth").addEventListener("click", revealTruth);
    document.getElementById("resetGame").addEventListener("click", resetGame);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_unknown_candidate_payloads(
    raw_by_class: dict[str, pd.DataFrame],
    processed_by_class: dict[str, pd.DataFrame],
    result: DemoClassificationResult,
    max_per_class: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []

    for label in result.class_order:
        class_positions = np.flatnonzero(result.y_test == label)
        correct_positions = class_positions[result.y_pred[class_positions] == result.y_test[class_positions]]
        preferred_positions = correct_positions if len(correct_positions) else class_positions
        if len(preferred_positions) == 0:
            continue

        confidence = result.y_prob[preferred_positions].max(axis=1)
        selected_positions = preferred_positions[np.argsort(confidence)[::-1]][:max_per_class]
        for test_position in selected_positions:
            global_index = int(result.test_indices[int(test_position)])
            raw_label, raw_sample_idx, raw_signal = _global_signal_info_by_index(
                data_by_class=raw_by_class,
                class_order=result.class_order,
                global_index=global_index,
            )
            _, _, processed_signal = _global_signal_info_by_index(
                data_by_class=processed_by_class,
                class_order=result.class_order,
                global_index=global_index,
            )
            candidates.append(
                {
                    "id": f"{raw_label}-{raw_sample_idx}",
                    "rawLabel": raw_label,
                    "rawSampleIdx": raw_sample_idx,
                    "rawSignal": _to_float_list(raw_signal),
                    "processedSignal": _to_float_list(processed_signal),
                    "probabilities": _to_float_list(result.y_prob[int(test_position)]),
                    "predictedLabel": str(result.y_pred[int(test_position)]),
                    "trueLabel": str(result.y_test[int(test_position)]),
                    "mapPoint": _to_float_list(result.x_test_map[int(test_position)]),
                    "mapPoint3d": _to_float_list(result.x_test_map_3d[int(test_position)]),
                }
            )

    if not candidates:
        raise ValueError("No unknown candidate samples are available for the classification game.")
    return candidates


def _global_signal_by_index(
    processed_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
    global_index: int,
) -> np.ndarray:
    start = 0
    for label in class_order:
        values = processed_by_class[label].to_numpy(dtype=np.float32, copy=False)
        end = start + len(values)
        if start <= global_index < end:
            return values[global_index - start]
        start = end
    raise IndexError(f"global_index {global_index} is outside the processed signal dataset")


def _global_signal_info_by_index(
    data_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
    global_index: int,
) -> tuple[str, int, np.ndarray]:
    start = 0
    for label in class_order:
        values = data_by_class[label].to_numpy(dtype=np.float32, copy=False)
        end = start + len(values)
        if start <= global_index < end:
            local_index = int(global_index - start)
            return str(label), local_index, values[local_index]
        start = end
    raise IndexError(f"global_index {global_index} is outside the signal dataset")


def _plotly_2d_feature_traces(
    result: DemoClassificationResult,
    include_unknown: bool,
    unknown_visible: bool,
) -> list[dict[str, object]]:
    traces = []
    for label in result.class_order:
        train_mask = result.y_train == label
        test_mask = result.y_test == label
        color = CLASS_COLORS.get(label, "#777777")
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": f"{label} known",
                "x": _to_float_list(result.x_train_map[train_mask, 0]),
                "y": _to_float_list(result.x_train_map[train_mask, 1]),
                "marker": {"color": color, "opacity": 0.44, "size": 6},
            }
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": f"{label} test",
                "x": _to_float_list(result.x_test_map[test_mask, 0]),
                "y": _to_float_list(result.x_test_map[test_mask, 1]),
                "marker": {
                    "color": color,
                    "opacity": 0.95,
                    "size": 8,
                    "symbol": "circle-open",
                    "line": {"color": color, "width": 1.4},
                },
            }
        )

    if include_unknown:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": "unknown sample",
                "x": [float(result.unknown_map_point[0])],
                "y": [float(result.unknown_map_point[1])],
                "visible": bool(unknown_visible),
                "marker": {
                    "color": "#111111",
                    "size": 18,
                    "symbol": "star",
                    "line": {"color": "white", "width": 1.2},
                },
            }
        )
    return traces


def _plotly_3d_feature_traces(
    result: DemoClassificationResult,
    include_unknown: bool,
    unknown_visible: bool,
) -> list[dict[str, object]]:
    traces = []
    for label in result.class_order:
        train_mask = result.y_train == label
        test_mask = result.y_test == label
        color = CLASS_COLORS.get(label, "#777777")
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"{label} known",
                "x": _to_float_list(result.x_train_map_3d[train_mask, 0]),
                "y": _to_float_list(result.x_train_map_3d[train_mask, 1]),
                "z": _to_float_list(result.x_train_map_3d[train_mask, 2]),
                "marker": {"color": color, "opacity": 0.42, "size": 4},
            }
        )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"{label} test",
                "x": _to_float_list(result.x_test_map_3d[test_mask, 0]),
                "y": _to_float_list(result.x_test_map_3d[test_mask, 1]),
                "z": _to_float_list(result.x_test_map_3d[test_mask, 2]),
                "marker": {
                    "color": color,
                    "opacity": 0.95,
                    "size": 5,
                    "symbol": "circle-open",
                },
            }
        )

    if include_unknown:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": "unknown sample",
                "x": [float(result.unknown_map_point_3d[0])],
                "y": [float(result.unknown_map_point_3d[1])],
                "z": [float(result.unknown_map_point_3d[2])],
                "visible": bool(unknown_visible),
                "marker": {
                    "color": "#111111",
                    "opacity": 1.0,
                    "size": 9,
                    "symbol": "diamond",
                },
            }
        )
    return traces


def _to_float_list(values: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float32).tolist()


def _add_class_legend(ax, class_order: Sequence[str]) -> None:
    handles = [
        plt.Line2D([0], [0], color=CLASS_COLORS.get(label, "#777777"), lw=2, label=label)
        for label in class_order
    ]
    ax.legend(handles=handles, loc="best")


def _save(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
