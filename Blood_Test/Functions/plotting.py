from __future__ import annotations

from itertools import combinations
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


CHANNEL_LABELS = {
    0: "Real",
    1: "Imaginary",
    2: "Amplitude",
    3: "Phase (rad)",
    4: "PBS-subtracted real",
    5: "PBS-subtracted imaginary",
    6: "PBS-subtracted amplitude",
    7: "PBS-subtracted phase (rad)",
    8: "Complex ratio phase (rad)",
    9: "Complex difference magnitude",
    10: "Conjugate-product phase difference (rad)",
    11: "Cosine of phase difference",
    12: "Sine of phase difference",
}

CIRCULAR_CHANNEL_INDICES = {3, 8, 10}


def _channel_title(channel_index: int) -> str:
    """Return a plot title containing both the channel number and its meaning."""
    channel_name = CHANNEL_LABELS.get(channel_index, "Unknown channel")
    return f"Channel {channel_index}: {channel_name}"


def _concentration_sort_key(value: str) -> int:
    try:
        base, exponent = value.split("-", maxsplit=1)
        if base != "10":
            raise ValueError
        return int(exponent)
    except ValueError as error:
        raise ValueError(
            f"Expected concentration label in the form '10-N', got {value!r}"
        ) from error


def _valid_sample_indices(set_block: np.ndarray) -> np.ndarray:
    """Return biological sample indices containing at least one valid repetition."""
    values = np.asarray(set_block)
    if values.ndim != 4:
        raise ValueError(
            "Expected one set with shape "
            f"[sample, repetition, frequency, channel], got {values.shape}"
        )
    valid = ~np.isnan(values).all(axis=(1, 2, 3))
    return np.flatnonzero(valid)


def _average_sample_repetitions(
    sample_block: np.ndarray,
    *,
    channel_index: int,
) -> np.ndarray | None:
    """Average valid repetitions for one biological sample."""
    values = np.asarray(sample_block)[..., channel_index]
    if values.ndim != 2:
        raise ValueError(
            "Expected one sample with shape [repetition, frequency, channel], "
            f"got {np.asarray(sample_block).shape}"
        )

    all_nan = np.isnan(values).all(axis=1)
    partial_nan = np.isnan(values).any(axis=1) & ~all_nan
    if np.any(partial_nan):
        raise ValueError("A partially-NaN repetition cannot be averaged")
    valid_repetitions = values[~all_nan]
    if len(valid_repetitions) == 0:
        return None

    if channel_index in CIRCULAR_CHANNEL_INDICES:
        # Phase is circular: +pi and -pi represent almost the same angle.
        return np.angle(np.mean(np.exp(1j * valid_repetitions), axis=0))
    return valid_repetitions.mean(axis=0)


def _difference_plot_groups(
    data: Dict[str, Dict[str, np.ndarray]],
) -> tuple[list[tuple[str, np.ndarray | None]], list[np.ndarray]]:
    if "SAU" not in data or not data["SAU"]:
        raise KeyError("data must contain at least one SAU concentration")
    concentrations = sorted(data["SAU"], key=_concentration_sort_key)
    pbs_block = data.get("PBS", {}).get("reference")
    groups: list[tuple[str, np.ndarray | None]] = [
        ("PBS", None if pbs_block is None else np.asarray(pbs_block))
    ]
    groups.extend(
        (
            concentration.replace("10-", "10^-"),
            np.asarray(data["SAU"][concentration]),
        )
        for concentration in concentrations
    )
    blocks = [block for _, block in groups if block is not None]
    return groups, blocks


def plot_pbs_set_differences(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare the mean complex PBS signal between every pair of sets.

    PBS repetitions are averaged in the complex plane within each set. Each
    column then shows one set pair, with signed real/imaginary differences,
    complex-difference magnitude, and conjugate-product phase difference.
    """
    try:
        pbs_block = np.asarray(data["PBS"]["reference"], dtype=np.float32)
    except KeyError as error:
        raise KeyError("data must contain data['PBS']['reference']") from error

    if pbs_block.ndim != 5 or pbs_block.shape[-1] < 2:
        raise ValueError(
            "Expected PBS data with shape "
            "[set, sample, repetition, frequency, channel>=2], "
            f"got {pbs_block.shape}"
        )
    if pbs_block.shape[0] < 2:
        raise ValueError("At least two PBS measurement sets are required")

    # Average samples and repetitions while preserving set and frequency axes.
    mean_real = np.nanmean(pbs_block[..., 0], axis=(1, 2))
    mean_imaginary = np.nanmean(pbs_block[..., 1], axis=(1, 2))
    mean_complex = mean_real + 1j * mean_imaginary
    set_pairs = list(combinations(range(pbs_block.shape[0]), 2))

    row_labels = (
        "Real difference",
        "Imaginary difference",
        "Complex difference magnitude",
        "Relative phase (rad)",
    )
    fig, axes = plt.subplots(
        len(row_labels),
        len(set_pairs),
        figsize=(6.0 * len(set_pairs), 3.2 * len(row_labels)),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    for column_index, (first_set, second_set) in enumerate(set_pairs):
        first = mean_complex[first_set]
        second = mean_complex[second_set]
        traces = (
            first.real - second.real,
            first.imag - second.imag,
            np.abs(first - second),
            np.angle(first * np.conj(second)),
        )
        for row_index, trace in enumerate(traces):
            ax = axes[row_index, column_index]
            ax.plot(trace, color="tab:blue", linewidth=1.2)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            if row_index == 0:
                ax.set_title(f"Set {first_set + 1} - Set {second_set + 1}")
            if column_index == 0:
                ax.set_ylabel(row_labels[row_index])
            if row_index == len(row_labels) - 1:
                ax.set_xlabel("Frequency point index")
            ax.grid(True, alpha=0.25)

    fig.suptitle("Pairwise Differences Between Mean PBS Signals", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def _validate_plot_blocks(
    blocks: list[np.ndarray],
    *,
    channel_indices: tuple[int, ...],
) -> int:
    if not blocks:
        raise ValueError("No signal blocks were found")
    if not channel_indices:
        raise ValueError("channel_indices must not be empty")
    for block in blocks:
        if block.ndim != 5:
            raise ValueError(
                "Expected blocks with shape "
                "[set, sample, repetition, frequency, channel], "
                f"got {block.shape}"
            )
        invalid_channels = [
            index for index in channel_indices if index < 0 or index >= block.shape[-1]
        ]
        if invalid_channels:
            raise IndexError(
                f"Channel indices {invalid_channels} are invalid for shape {block.shape}"
            )
    return max(block.shape[0] for block in blocks)


def _sample_mean_traces(
    set_block: np.ndarray,
    *,
    channel_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_indices = _valid_sample_indices(set_block)
    traces = []
    retained_indices = []
    for sample_index in sample_indices:
        trace = _average_sample_repetitions(
            set_block[sample_index],
            channel_index=channel_index,
        )
        if trace is not None:
            retained_indices.append(int(sample_index))
            traces.append(trace)
    if not traces:
        return (
            np.asarray([], dtype=np.int64),
            np.empty((0, set_block.shape[-2]), dtype=np.float32),
        )
    return (
        np.asarray(retained_indices, dtype=np.int64),
        np.stack(traces, axis=0),
    )


def plot_bacterial_set_differences(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    channel_indices: tuple[int, ...] = (0, 1, 2, 3),
    comparison_label: str | None = None,
    show: bool = True,
) -> dict[str, tuple[plt.Figure, np.ndarray]]:
    """Compare set-level mean SAU signals within each concentration.

    Each biological sample is first averaged across its repetitions. Those
    sample-level traces are then averaged with equal weight within each set.
    One figure per concentration shows pairwise set differences for the
    requested channels. Channels listed in ``CIRCULAR_CHANNEL_INDICES`` use
    circular means and wrapped pairwise subtraction.
    """
    if "SAU" not in data or not data["SAU"]:
        raise KeyError("data must contain at least one SAU concentration")

    if not channel_indices:
        raise ValueError("channel_indices must not be empty")
    row_labels = tuple(
        f"Channel {channel_index}: "
        f"{CHANNEL_LABELS.get(channel_index, 'Unknown channel')}\n"
        "set-mean difference"
        for channel_index in channel_indices
    )
    row_colors = plt.get_cmap("tab10")
    figures: dict[str, tuple[plt.Figure, np.ndarray]] = {}

    for concentration in sorted(data["SAU"], key=_concentration_sort_key):
        block = np.asarray(data["SAU"][concentration], dtype=np.float32)
        if block.ndim != 5:
            raise ValueError(
                "Expected SAU data with shape "
                "[set, sample, repetition, frequency, channel], "
                f"got {block.shape} for {concentration}"
            )
        invalid_channels = [
            channel_index
            for channel_index in channel_indices
            if channel_index < 0 or channel_index >= block.shape[-1]
        ]
        if invalid_channels:
            raise IndexError(
                f"Channel indices {invalid_channels} are invalid for "
                f"{concentration} with shape {block.shape}"
            )
        if block.shape[0] < 2:
            raise ValueError("At least two SAU measurement sets are required")

        set_means = np.full(
            (block.shape[0], len(channel_indices), block.shape[-2]),
            np.nan,
            dtype=np.float32,
        )
        sample_counts = np.zeros(block.shape[0], dtype=np.int64)
        for set_index in range(block.shape[0]):
            for channel_position, channel_index in enumerate(channel_indices):
                _, sample_traces = _sample_mean_traces(
                    block[set_index],
                    channel_index=channel_index,
                )
                if len(sample_traces) == 0:
                    continue
                if channel_position == 0:
                    sample_counts[set_index] = len(sample_traces)
                if channel_index in CIRCULAR_CHANNEL_INDICES:
                    set_mean = np.angle(
                        np.mean(np.exp(1j * sample_traces), axis=0)
                    )
                else:
                    set_mean = sample_traces.mean(axis=0)
                set_means[set_index, channel_position] = set_mean

        set_pairs = list(combinations(range(block.shape[0]), 2))
        fig, axes = plt.subplots(
            len(channel_indices),
            len(set_pairs),
            figsize=(6.0 * len(set_pairs), 3.2 * len(channel_indices)),
            squeeze=False,
            sharex=True,
            sharey="row",
        )
        for column_index, (first_set, second_set) in enumerate(set_pairs):
            for row_index, channel_index in enumerate(channel_indices):
                first_mean = set_means[first_set, row_index]
                second_mean = set_means[second_set, row_index]
                if channel_index in CIRCULAR_CHANNEL_INDICES:
                    difference = np.angle(
                        np.exp(1j * (first_mean - second_mean))
                    )
                else:
                    difference = first_mean - second_mean

                ax = axes[row_index, column_index]
                ax.plot(difference, color=row_colors(row_index % 10), linewidth=1.2)
                ax.axhline(
                    0.0,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
                if row_index == 0:
                    ax.set_title(
                        f"Set {first_set + 1} (n={sample_counts[first_set]}) - "
                        f"Set {second_set + 1} (n={sample_counts[second_set]})"
                    )
                if column_index == 0:
                    ax.set_ylabel(row_labels[row_index])
                if row_index == len(channel_indices) - 1:
                    ax.set_xlabel("Frequency point index")
                ax.grid(True, alpha=0.25)

        concentration_label = concentration.replace("10-", "10^-")
        figure_title = (
            f"SAU {concentration_label}: Pairwise Differences Between Set Means"
        )
        if comparison_label:
            figure_title += f"\n{comparison_label}"
        fig.suptitle(figure_title, fontsize=15)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        figures[concentration] = (fig, axes)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures


def plot_raw_signal_representatives(
    raw_data: Dict[str, Dict[str, np.ndarray]],
    *,
    samples_per_group: int = 4,
    random_seed: int | None = 42,
    show: bool = True,
) -> dict[int, tuple[plt.Figure, np.ndarray]]:
    """
    Plot representative unprocessed PBS and SAU signals in separate panels.

    One figure is created per measurement set. Its rows are amplitude and phase,
    and its columns are PBS and the available SAU concentrations. Several
    biological samples are selected without replacement in each group. Each
    curve averages that sample's valid repetitions; phase uses a circular mean.
    All-NaN excluded blocks are shown as unavailable.
    """
    if "SAU" not in raw_data or not raw_data["SAU"]:
        raise KeyError("raw_data must contain at least one SAU concentration")
    if samples_per_group < 1:
        raise ValueError("samples_per_group must be >= 1")

    concentrations = sorted(
        raw_data["SAU"],
        key=_concentration_sort_key,
    )
    blocks = [np.asarray(raw_data["SAU"][value]) for value in concentrations]
    pbs_block = None
    if "PBS" in raw_data and "reference" in raw_data["PBS"]:
        pbs_block = np.asarray(raw_data["PBS"]["reference"])
        blocks.append(pbs_block)

    for block in blocks:
        if block.ndim != 5 or block.shape[-1] < 4:
            raise ValueError(
                "Expected raw blocks with shape "
                "[set, sample, repetition, frequency, channel>=4], "
                f"got {block.shape}"
            )

    number_of_sets = max(block.shape[0] for block in blocks)
    if number_of_sets < 1:
        raise ValueError("No measurement sets were found")

    color_map = plt.get_cmap("tab10")
    groups: list[tuple[str, np.ndarray | None, str]] = [
        ("PBS", pbs_block, "--")
    ]
    groups.extend(
        (
            concentration.replace("10-", "10^-"),
            np.asarray(raw_data["SAU"][concentration]),
            "-",
        )
        for concentration in concentrations
    )

    rng = np.random.default_rng(random_seed)
    plotted_channels = (0, 1, 2, 3)
    figures: dict[int, tuple[plt.Figure, np.ndarray]] = {}
    for set_index in range(number_of_sets):
        selected_samples: list[np.ndarray] = []
        for _, block, _ in groups:
            if block is None or set_index >= block.shape[0]:
                selected_samples.append(np.asarray([], dtype=np.int64))
                continue
            available = _valid_sample_indices(block[set_index])
            selection_count = min(samples_per_group, len(available))
            if selection_count == 0:
                selected_samples.append(np.asarray([], dtype=np.int64))
                continue
            selected_samples.append(
                np.sort(
                    rng.choice(
                        available,
                        size=selection_count,
                        replace=False,
                    )
                )
            )

        fig, axes = plt.subplots(
            len(plotted_channels),
            len(groups),
            figsize=(4.2 * len(groups), 11.5),
            squeeze=False,
            sharex=True,
            sharey="row",
        )
        for row_index, channel_index in enumerate(plotted_channels):
            for group_index, (label, block, linestyle) in enumerate(groups):
                ax = axes[row_index, group_index]
                group_samples = selected_samples[group_index]

                if block is None or len(group_samples) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid traces",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                else:
                    for sample_index in group_samples:
                        averaged_trace = _average_sample_repetitions(
                            block[set_index, sample_index],
                            channel_index=channel_index,
                        )
                        if averaged_trace is None:
                            continue
                        is_pbs = group_index == 0
                        ax.plot(
                            averaged_trace,
                            color=(
                                "black"
                                if is_pbs
                                else color_map(int(sample_index) % 10)
                            ),
                            linestyle=linestyle,
                            linewidth=1.1,
                            alpha=0.9,
                            label=f"S{int(sample_index) + 1}",
                        )

                if row_index == 0:
                    ax.set_title(label)
                if group_index == 0:
                    ax.set_ylabel(CHANNEL_LABELS[channel_index])
                if row_index == len(plotted_channels) - 1:
                    ax.set_xlabel("Frequency point index")
                ax.grid(True, alpha=0.25)
                if len(group_samples) > 0:
                    ax.legend(loc="upper right", fontsize=7, frameon=False)

        fig.suptitle(
            f"Set {set_index + 1}: Random Sample-Averaged Raw Signals",
            fontsize=15,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        figures[set_index + 1] = (fig, axes)
        # PyCharm's plot backend may retain only the final active figure when a
        # single show() follows a loop. Showing here registers every set figure.
        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures


def plot_filtering_comparison(
    raw_data: Dict[str, Dict[str, np.ndarray]],
    filtered_data: Dict[str, Dict[str, np.ndarray]],
    *,
    samples_per_group: int = 4,
    random_seed: int | None = 42,
    show: bool = True,
) -> dict[int, tuple[plt.Figure, np.ndarray]]:
    """Overlay raw and filtered sample-averaged amplitude/phase traces."""
    if "SAU" not in raw_data or not raw_data["SAU"]:
        raise KeyError("raw_data must contain at least one SAU concentration")
    if "SAU" not in filtered_data:
        raise KeyError("filtered_data must contain SAU signals")
    if samples_per_group < 1:
        raise ValueError("samples_per_group must be >= 1")

    concentrations = sorted(raw_data["SAU"], key=_concentration_sort_key)
    missing_concentrations = [
        value for value in concentrations if value not in filtered_data["SAU"]
    ]
    if missing_concentrations:
        raise KeyError(
            "filtered_data is missing SAU concentrations: "
            f"{missing_concentrations}"
        )

    raw_pbs = raw_data.get("PBS", {}).get("reference")
    filtered_pbs = filtered_data.get("PBS", {}).get("reference")
    groups: list[tuple[str, np.ndarray | None, np.ndarray | None]] = [
        (
            "PBS",
            None if raw_pbs is None else np.asarray(raw_pbs),
            None if filtered_pbs is None else np.asarray(filtered_pbs),
        )
    ]
    groups.extend(
        (
            concentration.replace("10-", "10^-"),
            np.asarray(raw_data["SAU"][concentration]),
            np.asarray(filtered_data["SAU"][concentration]),
        )
        for concentration in concentrations
    )

    available_blocks = [
        block
        for _, raw_block, filtered_block in groups
        for block in (raw_block, filtered_block)
        if block is not None
    ]
    for block in available_blocks:
        if block.ndim != 5 or block.shape[-1] < 4:
            raise ValueError(
                "Expected signal blocks with shape "
                "[set, sample, repetition, frequency, channel>=4], "
                f"got {block.shape}"
            )
    for label, raw_block, filtered_block in groups:
        if (raw_block is None) != (filtered_block is None):
            raise ValueError(
                f"Raw and filtered data must both contain {label}, or both omit it"
            )
        if raw_block is not None and raw_block.shape != filtered_block.shape:
            raise ValueError(
                f"Raw/filtered shape mismatch for {label}: "
                f"{raw_block.shape} versus {filtered_block.shape}"
            )

    number_of_sets = max(block.shape[0] for block in available_blocks)
    rng = np.random.default_rng(random_seed)
    color_map = plt.get_cmap("tab10")
    plotted_channels = (0, 1, 2, 3)
    figures: dict[int, tuple[plt.Figure, np.ndarray]] = {}

    for set_index in range(number_of_sets):
        selected_samples: list[np.ndarray] = []
        for _, raw_block, filtered_block in groups:
            if (
                raw_block is None
                or filtered_block is None
                or set_index >= raw_block.shape[0]
            ):
                selected_samples.append(np.asarray([], dtype=np.int64))
                continue
            raw_valid = _valid_sample_indices(raw_block[set_index])
            filtered_valid = _valid_sample_indices(filtered_block[set_index])
            available = np.intersect1d(raw_valid, filtered_valid)
            selection_count = min(samples_per_group, len(available))
            if selection_count == 0:
                selected_samples.append(np.asarray([], dtype=np.int64))
                continue
            selected_samples.append(
                np.sort(rng.choice(available, size=selection_count, replace=False))
            )

        fig, axes = plt.subplots(
            len(plotted_channels),
            len(groups),
            figsize=(4.2 * len(groups), 11.5),
            squeeze=False,
            sharex=True,
            sharey="row",
        )
        for row_index, channel_index in enumerate(plotted_channels):
            for group_index, (label, raw_block, filtered_block) in enumerate(groups):
                ax = axes[row_index, group_index]
                group_samples = selected_samples[group_index]
                if raw_block is None or len(group_samples) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid traces",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                else:
                    for sample_index in group_samples:
                        raw_trace = _average_sample_repetitions(
                            raw_block[set_index, sample_index],
                            channel_index=channel_index,
                        )
                        filtered_trace = _average_sample_repetitions(
                            filtered_block[set_index, sample_index],
                            channel_index=channel_index,
                        )
                        if raw_trace is None or filtered_trace is None:
                            continue
                        color = (
                            "black"
                            if group_index == 0
                            else color_map(int(sample_index) % 10)
                        )
                        ax.plot(
                            raw_trace,
                            color=color,
                            linestyle="--",
                            linewidth=0.8,
                            alpha=0.35,
                            label="_nolegend_",
                        )
                        ax.plot(
                            filtered_trace,
                            color=color,
                            linestyle="-",
                            linewidth=1.35,
                            alpha=0.95,
                            label=f"S{int(sample_index) + 1}",
                        )

                if row_index == 0:
                    ax.set_title(label)
                if group_index == 0:
                    ax.set_ylabel(CHANNEL_LABELS[channel_index])
                if row_index == len(plotted_channels) - 1:
                    ax.set_xlabel("Frequency point index")
                ax.grid(True, alpha=0.25)
                if len(group_samples) > 0:
                    ax.legend(loc="upper right", fontsize=7, frameon=False)

        style_handles = [
            Line2D(
                [0],
                [0],
                color="dimgray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                label="Raw",
            ),
            Line2D(
                [0],
                [0],
                color="dimgray",
                linestyle="-",
                linewidth=1.35,
                label="Filtered",
            ),
        ]
        fig.suptitle(
            f"Set {set_index + 1}: Raw vs Filtered Sample-Averaged Signals",
            fontsize=15,
            y=0.995,
        )
        fig.legend(
            handles=style_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=2,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

        figures[set_index + 1] = (fig, axes)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures


def plot_difference_channel_samples(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    channel_indices: tuple[int, ...] = (8, 9),
    samples_per_group: int = 4,
    random_seed: int | None = 42,
    show: bool = True,
) -> dict[int, tuple[plt.Figure, np.ndarray]]:
    """Plot random sample-averaged PBS-referenced feature traces by set."""
    if samples_per_group < 1:
        raise ValueError("samples_per_group must be >= 1")
    groups, blocks = _difference_plot_groups(data)
    number_of_sets = _validate_plot_blocks(
        blocks,
        channel_indices=channel_indices,
    )

    rng = np.random.default_rng(random_seed)
    color_map = plt.get_cmap("tab10")
    figures: dict[int, tuple[plt.Figure, np.ndarray]] = {}
    for set_index in range(number_of_sets):
        selected_samples: list[np.ndarray] = []
        for _, block in groups:
            if block is None or set_index >= block.shape[0]:
                selected_samples.append(np.asarray([], dtype=np.int64))
                continue
            available = _valid_sample_indices(block[set_index])
            selection_count = min(samples_per_group, len(available))
            if selection_count == 0:
                selected_samples.append(np.asarray([], dtype=np.int64))
                continue
            selected_samples.append(
                np.sort(rng.choice(available, size=selection_count, replace=False))
            )

        fig, axes = plt.subplots(
            len(groups),
            len(channel_indices),
            figsize=(7.5 * len(channel_indices), 3.0 * len(groups)),
            squeeze=False,
            sharex=True,
            sharey="col",
        )
        for group_index, (label, block) in enumerate(groups):
            for column_index, channel_index in enumerate(channel_indices):
                ax = axes[group_index, column_index]
                group_samples = selected_samples[group_index]
                if block is None or len(group_samples) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid traces",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                else:
                    for sample_index in group_samples:
                        trace = _average_sample_repetitions(
                            block[set_index, sample_index],
                            channel_index=channel_index,
                        )
                        if trace is None:
                            continue
                        ax.plot(
                            trace,
                            color=(
                                "black"
                                if group_index == 0
                                else color_map(int(sample_index) % 10)
                            ),
                            linestyle="--" if group_index == 0 else "-",
                            linewidth=1.1,
                            alpha=0.9,
                            label=f"S{int(sample_index) + 1}",
                        )

                if group_index == 0:
                    ax.set_title(_channel_title(channel_index))
                if column_index == 0:
                    ax.set_ylabel(label)
                if group_index == len(groups) - 1:
                    ax.set_xlabel("Frequency point index")
                ax.grid(True, alpha=0.25)
                if len(group_samples) > 0:
                    ax.legend(loc="upper right", fontsize=7, frameon=False)

        fig.suptitle(
            f"Set {set_index + 1}: PBS-Referenced Channels "
            + ", ".join(str(index) for index in channel_indices),
            fontsize=15,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        figures[set_index + 1] = (fig, axes)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return figures


def plot_difference_channel_summary(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    channel_indices: tuple[int, ...] = (8, 9),
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot set rows by channel columns with concentration mean +/- sample SD."""
    groups, blocks = _difference_plot_groups(data)
    number_of_sets = _validate_plot_blocks(
        blocks,
        channel_indices=channel_indices,
    )

    color_map = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        number_of_sets,
        len(channel_indices),
        figsize=(6.0 * len(channel_indices), 3.7 * number_of_sets),
        squeeze=False,
        sharex=True,
        sharey="col",
    )
    for set_index in range(number_of_sets):
        for column_index, channel_index in enumerate(channel_indices):
            ax = axes[set_index, column_index]
            for group_index, (label, block) in enumerate(groups):
                if block is None or set_index >= block.shape[0]:
                    continue
                _, traces = _sample_mean_traces(
                    block[set_index],
                    channel_index=channel_index,
                )
                if len(traces) == 0:
                    continue

                if channel_index in CIRCULAR_CHANNEL_INDICES:
                    mean_trace = np.angle(np.mean(np.exp(1j * traces), axis=0))
                    angular_residuals = np.angle(
                        np.exp(1j * (traces - mean_trace[np.newaxis, :]))
                    )
                    spread = angular_residuals.std(axis=0)
                    lower = np.maximum(mean_trace - spread, -np.pi)
                    upper = np.minimum(mean_trace + spread, np.pi)
                else:
                    mean_trace = traces.mean(axis=0)
                    spread = traces.std(axis=0)
                    lower = mean_trace - spread
                    upper = mean_trace + spread

                color = "black" if group_index == 0 else color_map(group_index - 1)
                linestyle = "--" if group_index == 0 else "-"
                summary_label = f"{label} (n={len(traces)})"
                (line,) = ax.plot(
                    mean_trace,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    label=summary_label,
                )
                if len(traces) > 1:
                    ax.fill_between(
                        np.arange(mean_trace.shape[0]),
                        lower,
                        upper,
                        color=line.get_color(),
                        alpha=0.16,
                        linewidth=0.0,
                    )

            if set_index == 0:
                ax.set_title(_channel_title(channel_index))
            if set_index == number_of_sets - 1:
                ax.set_xlabel("Frequency point index")
            if column_index == 0:
                ax.set_ylabel(f"Set {set_index + 1}")
            ax.grid(True, alpha=0.25)
            if ax.lines:
                ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(
        "Concentration Mean +/- Sample SD by Measurement Set",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes
