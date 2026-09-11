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
STD_BAND_ALPHA = 0.26

CLASS_DISPLAY_LABELS = {
    "LOW": "Purified Water",
    "TARGET": "Tap Water",
    "HIGH": "Dirty Water",
}


def display_class_label(label: str) -> str:
    return CLASS_DISPLAY_LABELS.get(str(label), str(label))


def display_class_order(class_order: Sequence[str]) -> list[str]:
    return [display_class_label(label) for label in class_order]


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
        raw_ax.set_title(f"{display_class_label(label)} sample {sample_idx}: raw measurement")
        raw_ax.set_ylabel("Sensor response")

        clean_ax.plot(
            x_full,
            despiked_signal,
            color=color,
            linewidth=1.8,
            label="after spike removal",
        )
        clean_ax.set_title(f"{display_class_label(label)} sample {sample_idx}: after spike removal")
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
            ax.set_title(f"{display_class_label(label)} sample {sample_idx}", fontsize=9)
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
            label=f"{display_class_label(label)} average",
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
            ax.plot(x_axis, mean, color=color, linewidth=2.2, label=display_class_label(label))
            ax.fill_between(
                x_axis,
                mean - std,
                mean + std,
                color=color,
                alpha=STD_BAND_ALPHA,
            )

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
            ax.plot(x_axis, mean, color=color, linewidth=2.0, label=display_class_label(label))
            ax.fill_between(
                x_axis,
                mean - std,
                mean + std,
                color=color,
                alpha=STD_BAND_ALPHA,
            )
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
            label=f"{display_class_label(label)} known",
        )
        ax.scatter(
            result.x_test_map[test_mask, 0],
            result.x_test_map[test_mask, 1],
            s=62,
            facecolor="none",
            edgecolor=color,
            linewidth=1.5,
            label=f"{display_class_label(label)} test",
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
        ax.set_title("5. Feature Map of Manual Signal Measurements")
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
        ax.set_title("6. 3D PCA Feature Map of Manual Signal Measurements")
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
                name=f"{display_class_label(label)} known",
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
                name=f"{display_class_label(label)} test",
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
        else "6. Interactive 3D PCA Feature Map of Manual Signal Measurements"
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
            label=f"{display_class_label(label)} known",
        )
        ax.scatter(
            x_test_map_3d[test_mask, 0],
            x_test_map_3d[test_mask, 1],
            x_test_map_3d[test_mask, 2],
            s=52,
            facecolor="none",
            edgecolor=color,
            linewidth=1.3,
            label=f"{display_class_label(label)} test",
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
                name=f"{display_class_label(label)} known",
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
                name=f"{display_class_label(label)} test",
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
    pca_result: DemoClassificationResult,
    simple_feature_result: DemoClassificationResult,
    complex_feature_result: DemoClassificationResult,
    cnn_result: DemoClassificationResult,
    output_path: Path,
) -> Path:
    method_results = [
        ("PCA Feature", pca_result, "#9B51E0"),
        ("Simple Feature", simple_feature_result, "#F2994A"),
        ("Complex Feature", complex_feature_result, "#27AE60"),
        ("CNN Feature", cnn_result, "#2F80ED"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.2))

    for ax, (name, method_result, _color) in zip(axes.ravel()[:4], method_results):
        _plot_confusion_matrix_recall(
            ax,
            confusion_count=method_result.confusion_count,
            class_order=method_result.class_order,
            title=f"{name}: {method_result.test_accuracy:.3f}",
        )

    names = [name for name, _result, _color in method_results]
    accuracies = [method_result.test_accuracy for _name, method_result, _color in method_results]
    colors = [color for _name, _result, color in method_results]
    accuracy_ax = axes.ravel()[4]
    bars = accuracy_ax.bar(names, accuracies, color=colors)
    accuracy_ax.set_ylim(0.0, 1.0)
    accuracy_ax.set_ylabel("Holdout accuracy")
    accuracy_ax.set_title("Accuracy")
    accuracy_ax.tick_params(axis="x", rotation=20)
    for bar, accuracy in zip(bars, accuracies):
        accuracy_ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.025,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axes.ravel()[5].axis("off")
    fig.suptitle("7. Classifier Results: Four Feature Views")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
    ax.set_xticklabels(display_class_order(class_order), rotation=35, ha="right")
    ax.set_yticklabels(display_class_order(class_order))
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
            label=f"{display_class_label(label)} known",
        )
        ax.scatter(
            x_test_map[test_mask, 0],
            x_test_map[test_mask, 1],
            s=54,
            facecolor="none",
            edgecolor=color,
            linewidth=1.3,
            label=f"{display_class_label(label)} test",
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
    axes[1].set_xticklabels(display_class_order(result.class_order))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Classifier confidence")
    axes[1].set_title(f"Prediction: {display_class_label(result.unknown_pred_label)}")

    fig.suptitle(
        f"8. Classifying an Unknown Example "
        f"(true label: {display_class_label(result.unknown_true_label)})"
    )
    return _save(fig, output_path)


def plot_unknown_classification_game_html(
    raw_by_class: dict[str, pd.DataFrame],
    processed_by_class: dict[str, pd.DataFrame],
    result: DemoClassificationResult,
    output_path: Path,
    simple_feature_result: DemoClassificationResult | None = None,
    complex_feature_result: DemoClassificationResult | None = None,
    pca_result: DemoClassificationResult | None = None,
    max_unknown_candidates_total: int = 18,
    max_clickable_references_per_class: int = 50,
) -> Path:
    from plotly.offline import get_plotlyjs

    transform_methods, transform_map_lookup_by_method = _build_game_transform_method_payloads(
        result=result,
        simple_feature_result=simple_feature_result,
        complex_feature_result=complex_feature_result,
        pca_result=pca_result,
    )
    prediction_lookup_by_method = _build_game_prediction_lookup_by_method(
        cnn_result=result,
        simple_feature_result=simple_feature_result,
        complex_feature_result=complex_feature_result,
        pca_result=pca_result,
    )
    unknown_samples = _build_unknown_candidate_payloads(
        raw_by_class=raw_by_class,
        processed_by_class=processed_by_class,
        result=result,
        max_total=max_unknown_candidates_total,
        transform_map_lookup_by_method=transform_map_lookup_by_method,
        prediction_lookup_by_method=prediction_lookup_by_method,
    )
    class_patterns = _build_class_pattern_payloads(
        processed_by_class=processed_by_class,
        class_order=result.class_order,
    )
    reference_samples = _build_clickable_reference_payloads(
        processed_by_class=processed_by_class,
        result=result,
        max_per_class=max_clickable_references_per_class,
        transform_map_lookup_by_method=transform_map_lookup_by_method,
    )
    payload = {
        "classOrder": display_class_order(result.class_order),
        "unknownSamples": unknown_samples,
        "classPatterns": class_patterns,
        "referenceSamples": reference_samples,
        "transformMethods": transform_methods,
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
    h2 {{
      margin: 0 0 10px;
      font-size: 19px;
    }}
    .top-row {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 16px;
    }}
    .top-row .status {{
      flex: 1;
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
    button.choice {{
      min-width: 104px;
      font-size: 18px;
      font-weight: 700;
      background: var(--choice-bg, var(--panel));
      border-color: var(--choice-color, var(--line));
      color: var(--choice-color, var(--ink));
    }}
    button.choice:hover:not(:disabled) {{
      background: var(--choice-hover-bg, var(--panel));
    }}
    button.choice.selected {{
      background: var(--choice-color, #111111);
      border-color: var(--choice-color, #111111);
      color: white;
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.42;
    }}
    .scoreboard {{
      display: grid;
      grid-template-columns: repeat(3, minmax(140px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .score {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px 14px;
    }}
    .score .value {{
      display: block;
      font-size: 26px;
      font-weight: 700;
    }}
    .score .label {{
      color: var(--muted);
      font-size: 14px;
    }}
    .status {{
      min-height: 24px;
      margin: 0;
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
    .guess-panel {{
      position: sticky;
      top: 10px;
      z-index: 20;
      min-height: 126px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    }}
    .sample-panel {{
      display: flex;
      flex-direction: column;
      min-height: 340px;
      padding: 16px;
    }}
    .sample-picker-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 2px;
    }}
    .sample-list {{
      display: contents;
    }}
    .sample-panel .panel-action {{
      margin-top: auto;
      padding-top: 10px;
    }}
    .panel-action {{
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 10px;
    }}
    button.sample-choice {{
      min-height: 44px;
      font-size: 15px;
      font-weight: 700;
    }}
    button.sample-reset {{
      grid-column: 4;
      min-height: 44px;
      font-size: 15px;
      font-weight: 700;
    }}
    button.sample-choice.selected {{
      background: #111111;
      border-color: #111111;
      color: white;
    }}
    .guess-buttons {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .method-panel {{
      min-height: 0;
      margin-bottom: 12px;
      padding: 14px;
    }}
    .method-buttons {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
      margin-bottom: 8px;
    }}
    button.method-choice {{
      font-weight: 700;
      text-align: center;
      background: var(--method-bg, var(--panel));
      border-color: var(--method-color, var(--line));
      color: var(--method-color, var(--ink));
    }}
    button.method-choice:hover:not(:disabled) {{
      background: var(--method-hover-bg, var(--panel));
    }}
    button.method-choice.selected {{
      background: var(--method-color, var(--blue));
      border-color: var(--method-color, var(--blue));
      color: white;
    }}
    .hint {{
      margin: 0;
      min-height: 22px;
      color: var(--muted);
      font-size: 16px;
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
    #featurePlot3d {{
      height: 520px;
    }}
    #selectedCurvePlot {{
      height: 470px;
    }}
    .curve-note {{
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 15px;
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
      font-size: 18px;
    }}
    .answer-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 8px 0;
      border-bottom: 1px solid #eeeeee;
    }}
    .answer-row:last-child {{
      border-bottom: 0;
    }}
    .answer strong,
    .answer .placeholder {{
      display: block;
      font-size: 24px;
      line-height: 1.1;
    }}
    .placeholder {{
      color: var(--muted);
    }}
    .result-message {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #eeeeee;
      font-size: 16px;
      color: var(--muted);
    }}
    .chart-note {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.4;
    }}
    .hidden {{
      display: none;
    }}
    @media (max-width: 900px) {{
      .grid,
      .feature-grid,
      .prediction,
      .scoreboard {{
        grid-template-columns: 1fr;
      }}
      .top-row {{
        flex-direction: column;
      }}
      .guess-panel {{
        position: static;
        box-shadow: none;
      }}
      .sample-picker-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      button.sample-reset {{
        grid-column: 2;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>8. Unknown Sample Classification Game</h1>
    <div class="scoreboard">
      <div class="score">
        <span id="roundCount" class="value">0</span>
        <span class="label">samples revealed</span>
      </div>
      <div class="score">
        <span id="studentScore" class="value">0</span>
        <span class="label">student correct</span>
      </div>
      <div class="score">
        <span id="classifierScore" class="value">0</span>
        <span class="label">classifier correct</span>
      </div>
    </div>
    <div class="top-row">
      <p id="status" class="status">Choose a mystery sample from the list.</p>
      <button id="resetScore">Reset Score</button>
    </div>
    <div id="guessPanel" class="panel guess-panel">
      <h2>What do you think it is?</h2>
      <div id="guessButtons" class="guess-buttons"></div>
      <p id="guessStatus" class="hint">Choose a mystery sample first.</p>
    </div>

    <section class="grid">
      <div id="samplePanel" class="panel sample-panel">
        <h2>Choose a mystery sample</h2>
        <div class="sample-picker-grid">
          <div id="sampleList" class="sample-list"></div>
          <button id="tryAnother" class="sample-reset">Choose Another Sample</button>
        </div>
      </div>
      <div id="rawPanel" class="panel hidden">
        <h2>Raw Signal</h2>
        <div id="rawPlot" class="plot"></div>
        <div class="panel-action">
          <button id="processSample" class="primary" disabled>Process Sample</button>
        </div>
      </div>
      <div id="processedPanel" class="panel hidden">
        <h2>Processed Signal</h2>
        <div id="processedPlot" class="plot"></div>
      </div>
      <div id="patternPanel" class="panel hidden">
        <h2>Known Pattern Comparison</h2>
        <div id="patternPlot" class="plot"></div>
        <div class="panel-action">
          <button id="transformSample" class="primary" disabled>Transform Sample</button>
        </div>
      </div>
      <div id="featurePanel" class="wide hidden">
        <div id="methodPanel" class="panel method-panel">
          <h2>Choose a transform method</h2>
          <div id="methodButtons" class="method-buttons"></div>
          <p id="methodNote" class="hint"></p>
          <div class="panel-action">
            <button id="classifySample" class="primary" disabled>Ask Classifier</button>
          </div>
        </div>
        <div class="feature-grid">
          <div class="panel">
            <h2>Transformed View</h2>
            <div id="featurePlot3d" class="plot"></div>
          </div>
          <div id="selectedCurvePanel" class="panel">
            <h2>Clicked Signal Curve</h2>
            <p id="selectedCurveNote" class="curve-note">Click a highlighted map point to inspect its signal curve.</p>
            <div id="selectedCurvePlot" class="plot"></div>
          </div>
        </div>
      </div>
      <div id="predictionPanel" class="prediction wide hidden">
        <div class="answer">
          <h2>Prediction Results</h2>
          <div class="answer-row">
            <span>Raw signal guess</span>
            <strong id="rawGuessText" class="placeholder">?</strong>
          </div>
          <div class="answer-row">
            <span>After cleaning guess</span>
            <strong id="cleanGuessText" class="placeholder">?</strong>
          </div>
          <div class="answer-row">
            <span>After transform guess</span>
            <strong id="mapGuessText" class="placeholder">?</strong>
          </div>
          <div class="answer-row">
            <span>True label</span>
            <strong id="truthText" class="placeholder">?</strong>
          </div>
          <div id="resultMessage" class="result-message">Reveal the true label to update the score.</div>
          <div class="panel-action">
            <button id="revealTruth" class="primary" disabled>Reveal True Label</button>
          </div>
        </div>
        <div class="panel">
          <h2>Classifier Confidence for This Sample</h2>
          <div id="confidencePlot" class="plot"></div>
          <p id="confidenceNote" class="chart-note">Choose a transform method and click Ask Classifier. Tested methods will be added to this chart one at a time.</p>
        </div>
      </div>
    </section>
  </main>

  <script>
    const data = {json.dumps(payload)};
    const classColors = {json.dumps({display_class_label(label): color for label, color in CLASS_COLORS.items()})};
    const methodBarColors = {{
      pca: "#9B51E0",
      simple: "#F2994A",
      complex: "#27AE60",
      cnn: "#2F80ED",
    }};
    const samplePanel = document.getElementById("samplePanel");
    const sampleList = document.getElementById("sampleList");
    const rawPanel = document.getElementById("rawPanel");
    const guessPanel = document.getElementById("guessPanel");
    const guessButtons = document.getElementById("guessButtons");
    const guessStatus = document.getElementById("guessStatus");
    const processedPanel = document.getElementById("processedPanel");
    const patternPanel = document.getElementById("patternPanel");
    const featurePanel = document.getElementById("featurePanel");
    const methodButtons = document.getElementById("methodButtons");
    const methodNote = document.getElementById("methodNote");
    const selectedCurvePanel = document.getElementById("selectedCurvePanel");
    const selectedCurveNote = document.getElementById("selectedCurveNote");
    const predictionPanel = document.getElementById("predictionPanel");
    const statusEl = document.getElementById("status");
    const rawGuessText = document.getElementById("rawGuessText");
    const cleanGuessText = document.getElementById("cleanGuessText");
    const mapGuessText = document.getElementById("mapGuessText");
    const truthText = document.getElementById("truthText");
    const resultMessage = document.getElementById("resultMessage");
    const confidenceNote = document.getElementById("confidenceNote");
    const roundCount = document.getElementById("roundCount");
    const studentScore = document.getElementById("studentScore");
    const classifierScore = document.getElementById("classifierScore");
    let activeSample = null;
    let guessStage = null;
    let rawGuess = null;
    let cleanGuess = null;
    let mapGuess = null;
    let rounds = 0;
    let studentCorrect = 0;
    let classifierCorrect = 0;
    let roundFinished = false;
    let currentTransformMethod = "pca";
    let classifierGuess = null;
    let classifierMethodLabel = null;
    let testedClassifierResults = {{}};
    let testedClassifierMethodOrder = [];
    const methodById = Object.fromEntries(
      data.transformMethods.map(method => [method.id, method])
    );
    const referenceSampleById = Object.fromEntries(
      data.referenceSamples.map(sample => [sample.id, sample])
    );

    function methodDescription(method) {{
      const baseDescription = `${{method.intuitiveDescription}} ${{method.technicalDescription}}`;
      if (
        testedClassifierResults[method.id]
        && Number.isFinite(method.testAccuracy)
      ) {{
        return `${{baseDescription}} Overall classifier accuracy on held-out examples: ${{(100 * method.testAccuracy).toFixed(1)}}%.`;
      }}
      return baseDescription;
    }}

    function signalLayout(title, xTitle, yTitle) {{
      return {{
        title,
        xaxis: {{ title: xTitle }},
        yaxis: {{ title: yTitle }},
        margin: {{ l: 64, r: 18, b: 55, t: 48 }},
        template: "plotly_white",
      }};
    }}

    function patternLayout() {{
      return {{
        title: "Compare with known clean patterns",
        xaxis: {{ title: "Processed measurement point" }},
        yaxis: {{ title: "Processed response" }},
        margin: {{ l: 64, r: 18, b: 55, t: 48 }},
        template: "plotly_white",
        legend: {{ orientation: "v" }},
      }};
    }}

    function feature3dLayout(method) {{
      return {{
        title: `${{method.label}}: 3D Similarity Map`,
        scene: {{
          xaxis: {{ title: "Feature 1" }},
          yaxis: {{ title: "Feature 2" }},
          zaxis: {{ title: "Feature 3" }},
        }},
        margin: {{ l: 0, r: 0, b: 0, t: 50 }},
        template: "plotly_white",
        showlegend: true,
        legend: {{ x: 0.02, y: 0.98, bgcolor: "rgba(255,255,255,0.82)" }},
      }};
    }}

    function confidenceLayout() {{
      return {{
        title: "Classifier Confidence Comparison",
        xaxis: {{ title: "Class" }},
        yaxis: {{ title: "Confidence", range: [0, 1] }},
        margin: {{ l: 64, r: 18, b: 55, t: 48 }},
        template: "plotly_white",
        barmode: "group",
        legend: {{
          x: 0.99,
          y: 0.99,
          xanchor: "right",
          yanchor: "top",
          bgcolor: "rgba(255,255,255,0.86)",
          bordercolor: "#dddddd",
          borderwidth: 1,
        }},
      }};
    }}

    function hexToRgba(hex, alpha) {{
      const value = hex.replace("#", "");
      const r = parseInt(value.slice(0, 2), 16);
      const g = parseInt(value.slice(2, 4), 16);
      const b = parseInt(value.slice(4, 6), 16);
      return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
    }}

    function classPatternTraces(sample) {{
      const traces = [];
      data.classPatterns.forEach(pattern => {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          name: `${{pattern.label}} usual range`,
          x: [...pattern.x, ...pattern.x.slice().reverse()],
          y: [...pattern.upper, ...pattern.lower.slice().reverse()],
          fill: "toself",
          fillcolor: hexToRgba(pattern.color, 0.14),
          line: {{ color: "rgba(0, 0, 0, 0)", width: 0 }},
          hoverinfo: "skip",
          showlegend: false,
        }});
        traces.push({{
          type: "scatter",
          mode: "lines",
          name: `${{pattern.label}} known pattern`,
          x: pattern.x,
          y: pattern.mean,
          line: {{ color: pattern.color, width: 2.2 }},
        }});
      }});
      traces.push({{
        type: "scatter",
        mode: "lines",
        name: "mystery sample",
        x: sample.processedSignal.map((_, idx) => idx),
        y: sample.processedSignal,
        line: {{ color: "#111111", width: 3.0 }},
      }});
      return traces;
    }}

    function updateScoreboard() {{
      roundCount.textContent = String(rounds);
      studentScore.textContent = String(studentCorrect);
      classifierScore.textContent = String(classifierCorrect);
    }}

    function setButtonEnabled(id, enabled) {{
      document.getElementById(id).disabled = !enabled;
    }}

    function initializeGuessButtons() {{
      guessButtons.innerHTML = "";
      data.classOrder.forEach(label => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "choice";
        button.textContent = label;
        const color = classColors[label] || "#777777";
        button.style.setProperty("--choice-color", color);
        button.style.setProperty("--choice-bg", hexToRgba(color, 0.11));
        button.style.setProperty("--choice-hover-bg", hexToRgba(color, 0.18));
        button.disabled = true;
        button.addEventListener("click", () => makeGuess(label));
        guessButtons.appendChild(button);
      }});
    }}

    function initializeMethodButtons() {{
      methodButtons.innerHTML = "";
      data.transformMethods.forEach(method => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "method-choice";
        button.dataset.methodId = method.id;
        button.textContent = method.label;
        const color = methodBarColors[method.id] || "#777777";
        button.style.setProperty("--method-color", color);
        button.style.setProperty("--method-bg", hexToRgba(color, 0.11));
        button.style.setProperty("--method-hover-bg", hexToRgba(color, 0.18));
        button.addEventListener("click", () => plotTransformMethod(method.id));
        methodButtons.appendChild(button);
      }});
    }}

    function markSelectedMethod(methodId) {{
      document.querySelectorAll(".method-choice").forEach(button => {{
        const selected = button.dataset.methodId === methodId;
        button.classList.toggle("selected", selected);
      }});
    }}

    function setMethodButtonsEnabled(enabled) {{
      document.querySelectorAll(".method-choice").forEach(button => {{
        button.disabled = !enabled;
      }});
    }}

    function shuffledIndices(count) {{
      const indices = Array.from({{ length: count }}, (_, index) => index);
      for (let index = indices.length - 1; index > 0; index -= 1) {{
        const swapIndex = Math.floor(Math.random() * (index + 1));
        [indices[index], indices[swapIndex]] = [indices[swapIndex], indices[index]];
      }}
      return indices;
    }}

    function initializeSampleButtons() {{
      sampleList.innerHTML = "";
      document.getElementById("tryAnother").style.order = String(data.unknownSamples.length + 1);
      shuffledIndices(data.unknownSamples.length).forEach((sampleIndex, displayIndex) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "sample-choice";
        button.textContent = `Mystery Sample ${{displayIndex + 1}}`;
        button.style.order = String(displayIndex + 1);
        button.addEventListener("click", () => selectUnknownSample(sampleIndex, button));
        sampleList.appendChild(button);
      }});
    }}

    function setSampleButtonsEnabled(enabled) {{
      document.querySelectorAll(".sample-choice").forEach(button => {{
        button.disabled = !enabled;
      }});
    }}

    function clearSampleSelection() {{
      document.querySelectorAll(".sample-choice").forEach(button => {{
        button.classList.remove("selected");
      }});
    }}

    function setGuessButtonsEnabled(enabled) {{
      document.querySelectorAll(".choice").forEach(button => {{
        button.disabled = !enabled;
      }});
    }}

    function clearGuessSelection() {{
      document.querySelectorAll(".choice").forEach(button => {{
        button.classList.remove("selected");
      }});
    }}

    function cloneTraces(traces) {{
      return traces.map(trace => JSON.parse(JSON.stringify(trace)));
    }}

    function clearSelectedCurve() {{
      selectedCurveNote.textContent = "Click a highlighted map point to inspect its signal curve.";
      if (document.getElementById("selectedCurvePlot").data) {{
        Plotly.purge("selectedCurvePlot");
      }}
    }}

    function purgePlotIfDrawn(id) {{
      const plot = document.getElementById(id);
      if (plot && plot.data) {{
        Plotly.purge(plot);
      }}
    }}

    function unknown3dTrace(sample, methodId) {{
      const point = sample.mapPoints3d[methodId];
      return {{
        type: "scatter3d",
        mode: "markers",
        name: "Mystery sample",
        x: [point[0]],
        y: [point[1]],
        z: [point[2]],
        customdata: ["mystery"],
        text: ["Mystery sample"],
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{
          color: "#111111",
          opacity: 1.0,
          size: 12,
          symbol: "diamond",
        }},
      }};
    }}

    function selectedCurveLayout(title) {{
      return {{
        title,
        xaxis: {{ title: "Processed measurement point" }},
        yaxis: {{ title: "Processed response" }},
        margin: {{ l: 64, r: 18, b: 55, t: 48 }},
        template: "plotly_white",
      }};
    }}

    function selectedCurveTrace(sample, color, name) {{
      return {{
        type: "scatter",
        mode: "lines",
        x: sample.processedSignal.map((_, idx) => idx),
        y: sample.processedSignal,
        name,
        line: {{ color, width: 2.6 }},
      }};
    }}

    function showSelectedCurve(sample, title, color, name) {{
      selectedCurvePanel.classList.remove("hidden");
      selectedCurveNote.textContent = title;
      Plotly.newPlot(
        "selectedCurvePlot",
        [selectedCurveTrace(sample, color, name)],
        selectedCurveLayout(title),
        {{ responsive: true, displaylogo: false }}
      );
    }}

    function handleFeatureClick(event) {{
      if (!event.points || event.points.length === 0) return;
      const clickedId = event.points[0].customdata;
      if (!clickedId) return;
      if (clickedId === "mystery") {{
        showSelectedCurve(activeSample, "Clicked curve: mystery sample", "#111111", "mystery sample");
        return;
      }}
      const referenceSample = referenceSampleById[clickedId];
      if (!referenceSample) return;
      showSelectedCurve(
        referenceSample,
        `Clicked curve: ${{referenceSample.displayName}}`,
        referenceSample.color,
        referenceSample.displayName
      );
    }}

    function reference3dTraces(methodId) {{
      return data.classOrder.map(label => {{
        const samples = data.referenceSamples.filter(sample => sample.label === label && sample.mapPoints3d[methodId]);
        return {{
          type: "scatter3d",
          mode: "markers",
          name: label,
          x: samples.map(sample => sample.mapPoints3d[methodId][0]),
          y: samples.map(sample => sample.mapPoints3d[methodId][1]),
          z: samples.map(sample => sample.mapPoints3d[methodId][2]),
          customdata: samples.map(sample => sample.id),
          text: samples.map(sample => sample.displayName),
          hovertemplate: "%{{text}}<extra></extra>",
          marker: {{
            color: samples.map(sample => sample.color),
            size: 8,
            symbol: "square-open",
            line: {{ width: 5 }},
          }},
        }};
      }});
    }}

    function attachFeatureClickHandlers() {{
      const plot3d = document.getElementById("featurePlot3d");
      if (plot3d.removeAllListeners) plot3d.removeAllListeners("plotly_click");
      plot3d.on("plotly_click", handleFeatureClick);
    }}

    function resetClassifierResults() {{
      testedClassifierResults = {{}};
      testedClassifierMethodOrder = [];
      classifierGuess = null;
      classifierMethodLabel = null;
      truthText.textContent = "?";
      truthText.classList.add("placeholder");
      confidenceNote.textContent = "Choose a transform method and click Ask Classifier. Tested methods will be added to this chart one at a time.";
      const confidencePlot = document.getElementById("confidencePlot");
      if (confidencePlot.data) {{
        Plotly.purge(confidencePlot);
      }}
    }}

    function renderClassifierResults() {{
      if (testedClassifierMethodOrder.length === 0) {{
        return;
      }}
      const traces = testedClassifierMethodOrder.map(methodId => {{
        const method = methodById[methodId];
        const prediction = testedClassifierResults[methodId];
        return {{
          type: "bar",
          name: `${{method.label}} -> ${{prediction.predictedLabel}}`,
          x: data.classOrder,
          y: prediction.probabilities,
          marker: {{ color: methodBarColors[methodId] || "#777777" }},
          text: prediction.probabilities.map(value => `${{(100 * value).toFixed(1)}}%`),
          textposition: "outside",
        }};
      }});
      Plotly.newPlot("confidencePlot", traces, confidenceLayout(), {{ responsive: true, displaylogo: false }});
      confidenceNote.textContent = "The tallest bar is the class each tested method chooses. Comparing methods to see how different transformations can change the classification evidence and results.";
    }}

    function allClassifierMethodsTested() {{
      return testedClassifierMethodOrder.length >= data.transformMethods.length;
    }}

    function updateClassifyButtonAvailability() {{
      const canClassify = Boolean(activeSample && mapGuess && !allClassifierMethodsTested());
      setButtonEnabled("classifySample", canClassify);
    }}

    function plotTransformMethod(methodId) {{
      if (!activeSample || !methodById[methodId]) return;
      const method = methodById[methodId];
      currentTransformMethod = methodId;
      markSelectedMethod(methodId);
      methodNote.textContent = methodDescription(method);
      selectedCurvePanel.classList.remove("hidden");
      if (guessStage === "map") {{
        if (roundFinished) {{
          updateClassifyButtonAvailability();
          guessStatus.textContent = allClassifierMethodsTested()
            ? "The true label is revealed. All classifier methods have been tested."
            : "The true label is revealed. You can still compare views and add remaining classifier methods.";
        }} else if (mapGuess) {{
          updateClassifyButtonAvailability();
          guessStatus.textContent = "Click Ask Classifier to add this method to the comparison.";
        }} else {{
          setButtonEnabled("classifySample", false);
          guessStatus.textContent = "After seeing this transform, make your final guess.";
        }}
      }}

      clearSelectedCurve();
      selectedCurveNote.textContent = "Click an open square or the black mystery marker to inspect its cleaned curve.";
      Plotly.newPlot(
        "featurePlot3d",
        [
          ...cloneTraces(method.baseTraces),
          ...reference3dTraces(methodId),
          unknown3dTrace(activeSample, methodId),
        ],
        feature3dLayout(method),
        {{ responsive: true, displaylogo: false }}
      );
      attachFeatureClickHandlers();
    }}

    function resetRoundDisplay() {{
      processedPanel.classList.add("hidden");
      patternPanel.classList.add("hidden");
      featurePanel.classList.add("hidden");
      selectedCurvePanel.classList.add("hidden");
      predictionPanel.classList.add("hidden");
      clearSelectedCurve();
      purgePlotIfDrawn("rawPlot");
      purgePlotIfDrawn("processedPlot");
      purgePlotIfDrawn("patternPlot");
      purgePlotIfDrawn("featurePlot3d");
      currentTransformMethod = "pca";
      markSelectedMethod("pca");
      guessStage = null;
      rawGuess = null;
      cleanGuess = null;
      mapGuess = null;
      resetClassifierResults();
      roundFinished = false;
      clearGuessSelection();
      setGuessButtonsEnabled(false);
      setMethodButtonsEnabled(false);
      guessStatus.textContent = "Choose a mystery sample first.";
      rawGuessText.textContent = "?";
      rawGuessText.classList.add("placeholder");
      cleanGuessText.textContent = "?";
      cleanGuessText.classList.add("placeholder");
      mapGuessText.textContent = "?";
      mapGuessText.classList.add("placeholder");
      methodNote.textContent = "";
      resultMessage.textContent = "Reveal the true label to update the score.";
      setButtonEnabled("processSample", false);
      setButtonEnabled("transformSample", false);
      setButtonEnabled("classifySample", false);
      setButtonEnabled("revealTruth", false);
      setButtonEnabled("tryAnother", true);
    }}

    function selectUnknownSample(sampleIndex, button) {{
      const sample = data.unknownSamples[sampleIndex];
      activeSample = sample;
      resetRoundDisplay();
      clearSampleSelection();
      button.classList.add("selected");
      setSampleButtonsEnabled(false);
      rawPanel.classList.remove("hidden");
      guessStage = "raw";
      setGuessButtonsEnabled(true);
      Plotly.newPlot("rawPlot", [{{
        type: "scatter",
        mode: "lines",
        x: sample.rawSignal.map((_, idx) => idx),
        y: sample.rawSignal,
        line: {{ color: "#111111", width: 1.4 }},
        name: "unknown raw signal",
      }}], signalLayout("Unknown sample: raw measurement", "Measurement point", "Sensor response"), {{ responsive: true, displaylogo: false }});
      guessStatus.textContent = "Make your guess from the raw signal.";
      statusEl.textContent = "The class label is hidden. Choose Purified Water, Tap Water, or Dirty Water before processing.";
    }}

    function makeGuess(label) {{
      if (!activeSample || !guessStage || roundFinished) return;
      clearGuessSelection();
      const selectedButton = [...document.querySelectorAll(".choice")].find(button => button.textContent === label);
      if (selectedButton) selectedButton.classList.add("selected");
      if (guessStage === "raw") {{
        rawGuess = label;
        rawGuessText.textContent = label;
        rawGuessText.classList.remove("placeholder");
        guessStatus.textContent = `Raw signal guess: ${{label}}`;
        statusEl.textContent = "Now clean the signal and compare it with known patterns.";
        setButtonEnabled("processSample", true);
      }} else if (guessStage === "clean") {{
        cleanGuess = label;
        cleanGuessText.textContent = label;
        cleanGuessText.classList.remove("placeholder");
        guessStatus.textContent = `After cleaning guess: ${{label}}`;
        statusEl.textContent = "Now transform the signal into a map point.";
        setButtonEnabled("transformSample", true);
      }} else if (guessStage === "map") {{
        mapGuess = label;
        mapGuessText.textContent = label;
        mapGuessText.classList.remove("placeholder");
        guessStatus.textContent = `After transform guess: ${{label}}`;
        statusEl.textContent = "Now ask the classifier to classify the same sample.";
        setButtonEnabled("classifySample", true);
      }}
    }}

    function processSample() {{
      if (!activeSample || !rawGuess) return;
      processedPanel.classList.remove("hidden");
      patternPanel.classList.remove("hidden");
      featurePanel.classList.add("hidden");
      predictionPanel.classList.add("hidden");
      resetClassifierResults();
      resultMessage.textContent = "Reveal the true label to update the score.";
      Plotly.newPlot("processedPlot", [{{
        type: "scatter",
        mode: "lines",
        x: activeSample.processedSignal.map((_, idx) => idx),
        y: activeSample.processedSignal,
        line: {{ color: "#2F80ED", width: 2.0 }},
        name: "processed signal",
      }}], signalLayout("Same sample after processing", "Processed measurement point", "Processed response"), {{ responsive: true, displaylogo: false }});
      Plotly.newPlot("patternPlot", classPatternTraces(activeSample), patternLayout(), {{ responsive: true, displaylogo: false }});
      guessStage = "clean";
      clearGuessSelection();
      setGuessButtonsEnabled(true);
      guessStatus.textContent = "After seeing the cleaned signal and known patterns, choose again.";
      statusEl.textContent = "The noisy signal has been cleaned. Compare it with the known pattern bands, then guess again.";
      setButtonEnabled("processSample", false);
      setButtonEnabled("transformSample", false);
      setButtonEnabled("classifySample", false);
      setButtonEnabled("revealTruth", false);
    }}

    function transformSample() {{
      if (!activeSample || !cleanGuess) return;
      featurePanel.classList.remove("hidden");
      selectedCurvePanel.classList.remove("hidden");
      plotTransformMethod("pca");
      guessStage = "map";
      clearGuessSelection();
      setGuessButtonsEnabled(true);
      setMethodButtonsEnabled(true);
      guessStatus.textContent = "Choose a method, then make your final guess.";
      statusEl.textContent = "Try different ways to view the same sample. Good transformations make similar samples gather together.";
      setButtonEnabled("transformSample", false);
      setButtonEnabled("classifySample", false);
    }}

    function classifySample() {{
      if (!activeSample || !mapGuess || allClassifierMethodsTested()) return;
      const methodPrediction = activeSample.methodPredictions[currentTransformMethod];
      const method = methodById[currentTransformMethod];
      if (!methodPrediction || !method) return;
      classifierGuess = methodPrediction.predictedLabel;
      classifierMethodLabel = method.label;
      const alreadyTested = testedClassifierMethodOrder.includes(currentTransformMethod);
      testedClassifierResults[currentTransformMethod] = methodPrediction;
      if (!alreadyTested) {{
        testedClassifierMethodOrder.push(currentTransformMethod);
      }}
      predictionPanel.classList.remove("hidden");
      if (!roundFinished) {{
        truthText.textContent = "?";
        truthText.classList.add("placeholder");
      }}
      renderClassifierResults();
      methodNote.textContent = methodDescription(method);
      const testedCount = testedClassifierMethodOrder.length;
      const allTested = allClassifierMethodsTested();
      if (roundFinished) {{
        resultMessage.textContent = `${{method.label}} added after reveal. You have tested ${{testedCount}} method${{testedCount === 1 ? "" : "s"}}.${{allTested ? " All methods are now shown." : " Choose another transform and ask again to add more."}}`;
        statusEl.textContent = allTested
          ? "All classifier methods have been compared for this sample."
          : "Classifier comparison updated after reveal. You can still test remaining methods.";
      }} else {{
        resultMessage.textContent = `${{method.label}} added. You have tested ${{testedCount}} method${{testedCount === 1 ? "" : "s"}}.${{allTested ? " All methods are now shown. Reveal the true label when ready." : " Choose another transform and ask again, or reveal the true label."}}`;
        statusEl.textContent = "Classifier comparison updated. You can test another method before revealing the answer.";
      }}
      guessStage = "map";
      setGuessButtonsEnabled(!roundFinished);
      setMethodButtonsEnabled(true);
      setButtonEnabled("processSample", false);
      setButtonEnabled("transformSample", false);
      setButtonEnabled("classifySample", !allTested);
      setButtonEnabled("revealTruth", !roundFinished);
    }}

    function revealTruth() {{
      if (!activeSample || roundFinished) return;
      const studentWasCorrect = mapGuess === activeSample.trueLabel;
      const classifierWasCorrect = classifierGuess === activeSample.trueLabel;
      roundFinished = true;
      rounds += 1;
      if (studentWasCorrect) studentCorrect += 1;
      if (classifierWasCorrect) classifierCorrect += 1;
      updateScoreboard();
      truthText.textContent = activeSample.trueLabel;
      truthText.classList.remove("placeholder");
      const changedGuess = rawGuess !== mapGuess || cleanGuess !== mapGuess;
      const testedSummary = testedClassifierMethodOrder.map(methodId => {{
        const method = methodById[methodId];
        const prediction = testedClassifierResults[methodId];
        return `${{method.label}}: ${{prediction.predictedLabel}}`;
      }}).join(", ");
      resultMessage.textContent = `Final student guess: ${{mapGuess}}. Tested classifier methods: ${{testedSummary}}. Score uses the last tested method: ${{classifierMethodLabel}} guessed ${{classifierGuess}}. You were ${{studentWasCorrect ? "correct" : "not correct"}}. The classifier was ${{classifierWasCorrect ? "correct" : "not correct"}}.${{changedGuess ? " Your guess changed as the evidence changed." : ""}}`;
      statusEl.textContent = "Round complete. Try another mystery sample.";
      setButtonEnabled("revealTruth", false);
      setButtonEnabled("classifySample", !allClassifierMethodsTested());
      setButtonEnabled("tryAnother", true);
      setGuessButtonsEnabled(false);
      setMethodButtonsEnabled(true);
      guessStatus.textContent = allClassifierMethodsTested()
        ? "The true label is revealed. All classifier methods have been tested."
        : "The true label is revealed. You can still compare views and add remaining classifier methods.";
    }}

    function resetGame() {{
      rawPanel.classList.add("hidden");
      activeSample = null;
      resetRoundDisplay();
      clearSampleSelection();
      setSampleButtonsEnabled(true);
      statusEl.textContent = "Choose a mystery sample from the list.";
    }}

    function resetScore() {{
      rounds = 0;
      studentCorrect = 0;
      classifierCorrect = 0;
      updateScoreboard();
      resetGame();
    }}

    initializeSampleButtons();
    initializeGuessButtons();
    initializeMethodButtons();
    setMethodButtonsEnabled(false);
    updateScoreboard();
    document.getElementById("processSample").addEventListener("click", processSample);
    document.getElementById("transformSample").addEventListener("click", transformSample);
    document.getElementById("classifySample").addEventListener("click", classifySample);
    document.getElementById("revealTruth").addEventListener("click", revealTruth);
    document.getElementById("tryAnother").addEventListener("click", resetGame);
    document.getElementById("resetScore").addEventListener("click", resetScore);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_class_pattern_payloads(
    processed_by_class: dict[str, pd.DataFrame],
    class_order: Sequence[str],
) -> list[dict[str, object]]:
    patterns = []
    for label in class_order:
        values = processed_by_class[label].to_numpy(dtype=np.float32, copy=False)
        if values.size == 0:
            continue
        patterns.append(
            {
                "label": display_class_label(str(label)),
                "color": CLASS_COLORS.get(str(label), "#777777"),
                "x": list(range(values.shape[1])),
                "mean": _to_float_list(np.mean(values, axis=0)),
                "lower": _to_float_list(np.percentile(values, 10, axis=0)),
                "upper": _to_float_list(np.percentile(values, 90, axis=0)),
            }
        )
    return patterns


def _build_game_transform_method_payloads(
    result: DemoClassificationResult,
    simple_feature_result: DemoClassificationResult | None,
    complex_feature_result: DemoClassificationResult | None,
    pca_result: DemoClassificationResult | None,
) -> tuple[list[dict[str, object]], dict[str, dict[int, np.ndarray]]]:
    methods: list[dict[str, object]] = []
    map_lookup_by_method: dict[str, dict[int, np.ndarray]] = {}

    if pca_result is not None:
        methods.append(
            {
                "id": "pca",
                "label": "PCA Feature",
                "plotKind": "map",
                "testAccuracy": float(pca_result.test_accuracy),
                "intuitiveDescription": "Turn the whole cleaned curve into a point on a map, keeping the biggest overall differences.",
                "technicalDescription": "PCA projects the scaled processed signal into three principal-component coordinates, then KNN classifies those coordinates.",
                "baseTraces": _plotly_3d_feature_traces_from_arrays(
                    class_order=pca_result.class_order,
                    x_train_map_3d=pca_result.x_train_map_3d,
                    x_test_map_3d=pca_result.x_test_map_3d,
                    y_train=pca_result.y_train,
                    y_test=pca_result.y_test,
                ),
            }
        )
        map_lookup_by_method["pca"] = _build_3d_map_lookup_by_global_index(
            train_indices=pca_result.train_indices,
            x_train_map_3d=pca_result.x_train_map_3d,
            test_indices=pca_result.test_indices,
            x_test_map_3d=pca_result.x_test_map_3d,
        )

    if simple_feature_result is not None:
        methods.append(
            {
                "id": "simple",
                "label": "Simple Feature",
                "plotKind": "map",
                "testAccuracy": float(simple_feature_result.test_accuracy),
                "intuitiveDescription": "Measure each curve with a small set of easy clues, like height, spread, area, and peak position.",
                "technicalDescription": "A KNN classifier uses compact global summary features extracted from each processed signal channel.",
                "baseTraces": _plotly_3d_feature_traces_from_arrays(
                    class_order=simple_feature_result.class_order,
                    x_train_map_3d=simple_feature_result.x_train_map_3d,
                    x_test_map_3d=simple_feature_result.x_test_map_3d,
                    y_train=simple_feature_result.y_train,
                    y_test=simple_feature_result.y_test,
                ),
            }
        )
        map_lookup_by_method["simple"] = _build_3d_map_lookup_by_global_index(
            train_indices=simple_feature_result.train_indices,
            x_train_map_3d=simple_feature_result.x_train_map_3d,
            test_indices=simple_feature_result.test_indices,
            x_test_map_3d=simple_feature_result.x_test_map_3d,
        )

    if complex_feature_result is not None:
        methods.append(
            {
                "id": "complex",
                "label": "Complex Feature",
                "plotKind": "map",
                "testAccuracy": float(complex_feature_result.test_accuracy),
                "intuitiveDescription": "Measure the curve with more detailed clues, including peaks, dips, valleys, and how sharply it changes.",
                "technicalDescription": "The more complex extractor builds global, peak-dip, area, and derivative features before KNN classification.",
                "baseTraces": _plotly_3d_feature_traces_from_arrays(
                    class_order=complex_feature_result.class_order,
                    x_train_map_3d=complex_feature_result.x_train_map_3d,
                    x_test_map_3d=complex_feature_result.x_test_map_3d,
                    y_train=complex_feature_result.y_train,
                    y_test=complex_feature_result.y_test,
                ),
            }
        )
        map_lookup_by_method["complex"] = _build_3d_map_lookup_by_global_index(
            train_indices=complex_feature_result.train_indices,
            x_train_map_3d=complex_feature_result.x_train_map_3d,
            test_indices=complex_feature_result.test_indices,
            x_test_map_3d=complex_feature_result.x_test_map_3d,
        )

    methods.append(
        {
            "id": "cnn",
            "label": "CNN Feature",
            "plotKind": "map",
            "testAccuracy": float(result.test_accuracy),
            "intuitiveDescription": "Let the computer learn its own clues from many examples instead of hand-picking the measurements.",
            "technicalDescription": "A CNN slides small filters along the signal to learn local patterns and combine them into stronger clues; its learned features are then mapped into 3D with PCA.",
            "baseTraces": _plotly_3d_feature_traces_from_arrays(
                class_order=result.class_order,
                x_train_map_3d=result.x_train_map_3d,
                x_test_map_3d=result.x_test_map_3d,
                y_train=result.y_train,
                y_test=result.y_test,
            ),
        }
    )
    map_lookup_by_method["cnn"] = _build_3d_map_lookup_by_global_index(
        train_indices=result.train_indices,
        x_train_map_3d=result.x_train_map_3d,
        test_indices=result.test_indices,
        x_test_map_3d=result.x_test_map_3d,
    )

    return methods, map_lookup_by_method


def _build_3d_map_lookup_by_global_index(
    train_indices: np.ndarray,
    x_train_map_3d: np.ndarray,
    test_indices: np.ndarray,
    x_test_map_3d: np.ndarray,
) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    for position, global_index in enumerate(train_indices):
        lookup[int(global_index)] = x_train_map_3d[int(position)]
    for position, global_index in enumerate(test_indices):
        lookup[int(global_index)] = x_test_map_3d[int(position)]
    return lookup


def _map_points_for_global_index(
    global_index: int,
    transform_map_lookup_by_method: dict[str, dict[int, np.ndarray]],
) -> dict[str, list[float]]:
    map_points: dict[str, list[float]] = {}
    for method_id, lookup in transform_map_lookup_by_method.items():
        point = lookup.get(int(global_index))
        if point is not None:
            map_points[method_id] = _to_float_list(point)
    return map_points


def _build_game_prediction_lookup_by_method(
    cnn_result: DemoClassificationResult,
    simple_feature_result: DemoClassificationResult | None,
    complex_feature_result: DemoClassificationResult | None,
    pca_result: DemoClassificationResult | None,
) -> dict[str, dict[int, dict[str, object]]]:
    method_results = {
        "pca": pca_result,
        "simple": simple_feature_result,
        "complex": complex_feature_result,
        "cnn": cnn_result,
    }
    lookups: dict[str, dict[int, dict[str, object]]] = {}
    for method_id, method_result in method_results.items():
        if method_result is None:
            continue
        lookups[method_id] = _build_prediction_lookup_by_global_index(method_result)
    return lookups


def _build_prediction_lookup_by_global_index(
    result: DemoClassificationResult,
) -> dict[int, dict[str, object]]:
    lookup: dict[int, dict[str, object]] = {}
    for test_position, global_index in enumerate(result.test_indices):
        lookup[int(global_index)] = {
            "predictedLabel": display_class_label(str(result.y_pred[int(test_position)])),
            "probabilities": _to_float_list(result.y_prob[int(test_position)]),
        }
    return lookup


def _predictions_for_global_index(
    global_index: int,
    prediction_lookup_by_method: dict[str, dict[int, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    predictions: dict[str, dict[str, object]] = {}
    for method_id, lookup in prediction_lookup_by_method.items():
        prediction = lookup.get(int(global_index))
        if prediction is not None:
            predictions[method_id] = prediction
    return predictions


def _build_clickable_reference_payloads(
    processed_by_class: dict[str, pd.DataFrame],
    result: DemoClassificationResult,
    max_per_class: int,
    transform_map_lookup_by_method: dict[str, dict[int, np.ndarray]],
) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []

    for label in result.class_order:
        class_positions = np.flatnonzero(result.y_train == label)
        if len(class_positions) == 0:
            continue

        if len(class_positions) <= max_per_class:
            selected_positions = class_positions
        else:
            selected_offsets = np.linspace(0, len(class_positions) - 1, max_per_class, dtype=int)
            selected_positions = class_positions[selected_offsets]

        for local_count, train_position in enumerate(selected_positions, start=1):
            global_index = int(result.train_indices[int(train_position)])
            ref_label, ref_sample_idx, processed_signal = _global_signal_info_by_index(
                data_by_class=processed_by_class,
                class_order=result.class_order,
                global_index=global_index,
            )
            references.append(
                {
                    "id": f"known-{ref_label}-{ref_sample_idx}",
                    "displayName": f"Known {display_class_label(ref_label)} example {local_count}",
                    "label": display_class_label(ref_label),
                    "color": CLASS_COLORS.get(ref_label, "#777777"),
                    "processedSignal": _to_float_list(processed_signal),
                    "mapPoints3d": _map_points_for_global_index(
                        global_index=global_index,
                        transform_map_lookup_by_method=transform_map_lookup_by_method,
                    ),
                }
            )

    return references


def _build_unknown_candidate_payloads(
    raw_by_class: dict[str, pd.DataFrame],
    processed_by_class: dict[str, pd.DataFrame],
    result: DemoClassificationResult,
    max_total: int,
    transform_map_lookup_by_method: dict[str, dict[int, np.ndarray]],
    prediction_lookup_by_method: dict[str, dict[int, dict[str, object]]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    class_quotas = _balanced_class_quotas(
        y_test=result.y_test,
        class_order=result.class_order,
        max_total=max_total,
    )
    train_centroids_by_method = _build_train_centroids_by_method(
        result=result,
        transform_map_lookup_by_method=transform_map_lookup_by_method,
    )

    for label in result.class_order:
        class_positions = np.flatnonzero(result.y_test == label)
        if len(class_positions) == 0:
            continue

        teaching_scores = _score_unknown_teaching_positions(
            positions=class_positions,
            result=result,
            transform_map_lookup_by_method=transform_map_lookup_by_method,
            prediction_lookup_by_method=prediction_lookup_by_method,
            train_centroids_by_method=train_centroids_by_method,
        )
        selected_positions = _select_teaching_positions(
            positions=class_positions,
            scores=teaching_scores,
            max_count=class_quotas.get(str(label), 0),
        )
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
                    "rawLabel": display_class_label(raw_label),
                    "rawSampleIdx": raw_sample_idx,
                    "rawSignal": _to_float_list(raw_signal),
                    "processedSignal": _to_float_list(processed_signal),
                    "trueLabel": display_class_label(str(result.y_test[int(test_position)])),
                    "methodPredictions": _predictions_for_global_index(
                        global_index=global_index,
                        prediction_lookup_by_method=prediction_lookup_by_method,
                    ),
                    "mapPoints3d": _map_points_for_global_index(
                        global_index=global_index,
                        transform_map_lookup_by_method=transform_map_lookup_by_method,
                    ),
                }
            )

    if not candidates:
        raise ValueError("No unknown candidate samples are available for the classification game.")
    return candidates


def _balanced_class_quotas(
    y_test: np.ndarray,
    class_order: Sequence[str],
    max_total: int,
) -> dict[str, int]:
    if max_total < 1:
        raise ValueError("max_total must be at least 1")

    available = {
        str(label): int(np.count_nonzero(y_test == label))
        for label in class_order
    }
    labels_with_data = [label for label in class_order if available[str(label)] > 0]
    if not labels_with_data:
        return {}

    quotas = {str(label): 0 for label in class_order}
    for label in labels_with_data:
        quotas[str(label)] = min(max_total // len(labels_with_data), available[str(label)])

    remaining = min(max_total, sum(available.values())) - sum(quotas.values())
    label_cycle = sorted(
        (str(label) for label in labels_with_data),
        key=lambda label: available[label],
        reverse=True,
    )
    while remaining > 0:
        assigned_this_pass = False
        for label in label_cycle:
            if quotas[label] < available[label]:
                quotas[label] += 1
                remaining -= 1
                assigned_this_pass = True
                if remaining == 0:
                    break
        if not assigned_this_pass:
            break

    return quotas


def _build_train_centroids_by_method(
    result: DemoClassificationResult,
    transform_map_lookup_by_method: dict[str, dict[int, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    centroids_by_method: dict[str, dict[str, np.ndarray]] = {}
    for method_id, point_lookup in transform_map_lookup_by_method.items():
        method_centroids: dict[str, np.ndarray] = {}
        for label in result.class_order:
            points = [
                point_lookup[int(global_index)]
                for train_position, global_index in enumerate(result.train_indices)
                if result.y_train[int(train_position)] == label
                and int(global_index) in point_lookup
            ]
            if points:
                method_centroids[display_class_label(str(label))] = np.mean(
                    np.asarray(points, dtype=np.float32),
                    axis=0,
                )
        if method_centroids:
            centroids_by_method[method_id] = method_centroids
    return centroids_by_method


def _score_unknown_teaching_positions(
    positions: np.ndarray,
    result: DemoClassificationResult,
    transform_map_lookup_by_method: dict[str, dict[int, np.ndarray]],
    prediction_lookup_by_method: dict[str, dict[int, dict[str, object]]],
    train_centroids_by_method: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    scores = []
    weak_methods = ("pca", "simple")
    strong_methods = ("complex", "cnn")
    all_methods = ("pca", "simple", "complex", "cnn")

    for test_position in positions:
        global_index = int(result.test_indices[int(test_position)])
        true_label = display_class_label(str(result.y_test[int(test_position)]))
        predictions = {
            method_id: lookup[global_index]["predictedLabel"]
            for method_id, lookup in prediction_lookup_by_method.items()
            if global_index in lookup
        }
        wrong_methods = [
            method_id
            for method_id in all_methods
            if predictions.get(method_id) is not None
            and predictions[method_id] != true_label
        ]
        weak_wrong = sum(method_id in wrong_methods for method_id in weak_methods)
        strong_correct = sum(
            predictions.get(method_id) == true_label
            for method_id in strong_methods
        )
        disagreement = max(0, len(set(predictions.values())) - 1)

        visual_wrong = 0
        visual_weak_wrong = 0
        for method_id in all_methods:
            point = transform_map_lookup_by_method.get(method_id, {}).get(global_index)
            centroids = train_centroids_by_method.get(method_id, {})
            if point is None or not centroids:
                continue
            nearest_label = min(
                centroids,
                key=lambda label: float(np.linalg.norm(point - centroids[label])),
            )
            if nearest_label != true_label:
                visual_wrong += 1
                if method_id in weak_methods:
                    visual_weak_wrong += 1

        cnn_confidence = 1.0
        cnn_prediction = prediction_lookup_by_method.get("cnn", {}).get(global_index)
        if cnn_prediction is not None:
            cnn_confidence = float(max(cnn_prediction["probabilities"]))

        score = (
            8.0 * weak_wrong
            + 5.0 * len(wrong_methods)
            + 4.0 * disagreement
            + 3.0 * visual_weak_wrong
            + 2.0 * visual_wrong
            + 2.0 * weak_wrong * strong_correct
            + (1.0 - cnn_confidence)
        )
        scores.append(score)

    return np.asarray(scores, dtype=np.float32)


def _select_teaching_positions(
    positions: np.ndarray,
    scores: np.ndarray,
    max_count: int,
) -> np.ndarray:
    """Prefer mystery samples where feature views disagree or look visually ambiguous."""
    positions = np.asarray(positions)
    scores = np.asarray(scores, dtype=np.float32)
    if max_count <= 0:
        return positions[:0]
    if len(positions) <= max_count:
        return positions

    order = np.lexsort((positions, -scores))
    return positions[order[:max_count]]


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
                "name": f"{display_class_label(label)} known",
                "x": _to_float_list(result.x_train_map[train_mask, 0]),
                "y": _to_float_list(result.x_train_map[train_mask, 1]),
                "marker": {"color": color, "opacity": 0.44, "size": 6},
            }
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": f"{display_class_label(label)} test",
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


def _plotly_3d_feature_traces_from_arrays(
    class_order: Sequence[str],
    x_train_map_3d: np.ndarray,
    x_test_map_3d: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, object]]:
    traces = []
    for label in class_order:
        train_mask = y_train == label
        test_mask = y_test == label
        color = CLASS_COLORS.get(label, "#777777")
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"{display_class_label(label)} known",
                "x": _to_float_list(x_train_map_3d[train_mask, 0]),
                "y": _to_float_list(x_train_map_3d[train_mask, 1]),
                "z": _to_float_list(x_train_map_3d[train_mask, 2]),
                "marker": {"color": color, "opacity": 0.34, "size": 3},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"{display_class_label(label)} test",
                "x": _to_float_list(x_test_map_3d[test_mask, 0]),
                "y": _to_float_list(x_test_map_3d[test_mask, 1]),
                "z": _to_float_list(x_test_map_3d[test_mask, 2]),
                "marker": {
                    "color": color,
                    "opacity": 0.62,
                    "size": 4,
                    "symbol": "circle",
                },
                "hoverinfo": "skip",
                "showlegend": False,
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
                "name": f"{display_class_label(label)} known",
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
                "name": f"{display_class_label(label)} test",
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
        plt.Line2D(
            [0],
            [0],
            color=CLASS_COLORS.get(label, "#777777"),
            lw=2,
            label=display_class_label(label),
        )
        for label in class_order
    ]
    ax.legend(handles=handles, loc="best")


def _save(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
