"""Select actual biological samples closest to their set/concentration mean."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plotting import (
    CHANNEL_LABELS,
    CIRCULAR_CHANNEL_INDICES,
    _concentration_sort_key,
    _validate_plot_blocks,
)


def _closest_sample_to_mean(set_block: np.ndarray, *, channel_index: int):
    """Select by RMS distance across frequency, using circular phase statistics."""
    values = np.asarray(set_block[..., channel_index], dtype=np.float64)
    valid_repetitions = ~np.isnan(values).all(axis=-1)
    if not np.isfinite(values[valid_repetitions]).all():
        raise ValueError("Representative selection requires finite, complete repetitions")
    sample_indices = np.flatnonzero(valid_repetitions.any(axis=1))
    if len(sample_indices) == 0:
        return None

    circular = channel_index in CIRCULAR_CHANNEL_INDICES
    # Catch cancellation at the precision of the float32 source signals.
    resultant_tolerance = 1e-7
    sample_traces, defined_points = [], []
    for sample_index in sample_indices:
        repetitions = values[sample_index, valid_repetitions[sample_index]]
        if circular:
            resultant = np.exp(1j * repetitions).mean(axis=0)
            sample_traces.append(np.angle(resultant))
            defined_points.append(np.abs(resultant) > resultant_tolerance)
        else:
            sample_traces.append(repetitions.mean(axis=0))
            defined_points.append(np.ones(values.shape[-1], dtype=bool))
    traces = np.stack(sample_traces)
    defined_points = np.stack(defined_points)
    common_points = defined_points.all(axis=0)
    if circular:
        class_resultant = np.exp(1j * traces).mean(axis=0)
        class_mean = np.angle(class_resultant)
        common_points &= np.abs(class_resultant) > resultant_tolerance
        differences = np.angle(np.exp(1j * (traces - class_mean)))
    else:
        differences = traces - traces.mean(axis=0)
    if not common_points.any():
        raise ValueError("No frequency points have a defined mean for representative selection")
    distances = np.sqrt(np.mean(differences[:, common_points] ** 2, axis=1))
    # Ascending original sample IDs make exact ties deterministic.
    selected_position = int(np.argmin(distances))
    sample_index = int(sample_indices[selected_position])
    selected_trace = traces[selected_position].copy()
    selected_trace[~defined_points[selected_position]] = np.nan
    selection = {
        "channel_index": channel_index,
        "sample_number": sample_index + 1,
        "mean_trace": selected_trace,
        "distance_to_mean": float(distances[selected_position]),
        "number_of_samples": int(len(sample_indices)),
        "number_of_repetitions": int(valid_repetitions[sample_index].sum()),
        "scored_frequency_points": int(common_points.sum()),
        "total_frequency_points": int(values.shape[-1]),
    }
    return selected_trace, selection


def plot_representative_channel_samples(
    data: dict[str, dict[str, np.ndarray]],
    *,
    channel_index: int = 8,
    reference_label: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
    verbose: bool = True,
) -> tuple[dict[int, tuple[plt.Figure, plt.Axes]], dict[int, dict[str, dict]]]:
    """Plot one nearest-to-mean sample per concentration, one figure per set.

    Average repetitions per sample, then samples with equal weight per class
    within each set. Circular channels use circular means and circular RMS
    distance to that mean; other channels use arithmetic statistics. Points
    with undefined circular means are excluded on a common selection mask.
    Actual sample averages are plotted, not class means. All figures share
    a data-derived upper y limit and a lower limit of -0.15; no vertical offsets
    or amplitude normalization.

    Prefer ``complex_difference_data`` before second smoothing/downsampling.
    Return figures and selection metadata keyed by set number/concentration;
    metadata includes the selected ``mean_trace`` for subsequent comparisons.
    Saving appends ``_set_N`` to the requested filename stem.
    """
    if "SAU" not in data or not data["SAU"]:
        raise KeyError("data must contain at least one SAU concentration")
    concentrations = sorted(data["SAU"], key=_concentration_sort_key)
    blocks = [np.asarray(data["SAU"][c]) for c in concentrations]
    number_of_sets = _validate_plot_blocks(blocks, channel_indices=(channel_index,))
    selected_traces, selections, finite_limits = {}, {}, []
    circular = channel_index in CIRCULAR_CHANNEL_INDICES
    for set_index in range(number_of_sets):
        set_number = set_index + 1
        selected_traces[set_number], selections[set_number] = {}, {}
        for concentration, block in zip(concentrations, blocks):
            if set_index >= block.shape[0]:
                continue
            selected = _closest_sample_to_mean(block[set_index], channel_index=channel_index)
            if selected is None:
                if verbose:
                    print(f"Set {set_number}, {concentration}: no valid samples")
                continue
            trace, selection = selected
            selected_traces[set_number][concentration] = trace
            selections[set_number][concentration] = selection
            finite = trace[np.isfinite(trace)]
            if finite.size:
                finite_limits.extend((float(finite.min()), float(finite.max())))
            if verbose:
                unit = " rad" if circular else ""
                print(
                    f"Set {set_number}, {concentration}: S{selection['sample_number']}; "
                    f"RMSE to mean={selection['distance_to_mean']:.6g}{unit}; "
                    f"n={selection['number_of_samples']} samples, "
                    f"{selection['number_of_repetitions']} repetitions; "
                    f"scored {selection['scored_frequency_points']}/"
                    f"{selection['total_frequency_points']} frequency points"
                )

    color_map = plt.get_cmap("tab10")
    colors = {c: color_map(i % 10) for i, c in enumerate(concentrations)}
    figures = {}
    for set_number, traces in selected_traces.items():
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for concentration, trace in traces.items():
            selection = selections[set_number][concentration]
            frequency_indices = np.arange(len(trace), dtype=np.float64)
            plotted_trace = trace
            if circular:
                # Break connectors across +/-pi without dropping actual points.
                breaks = np.flatnonzero(np.abs(np.diff(trace)) > np.pi) + 1
                frequency_indices = np.insert(frequency_indices, breaks, np.nan)
                plotted_trace = np.insert(trace, breaks, np.nan)
            exponent = _concentration_sort_key(concentration)
            ax.plot(
                frequency_indices, plotted_trace,
                color=colors[concentration], linewidth=1.3,
                label=rf"$10^{{-{exponent}}}$ (S{selection['sample_number']})",
            )
        if finite_limits:
            lower, upper = min(finite_limits), max(finite_limits)
            margin = max((upper - lower) * 0.06, 1e-6)
            ax.set_ylim(lower - margin, upper + margin)
        ax.set_ylim(bottom=-0.15)
        if not traces:
            ax.text(0.5, 0.5, "No valid samples", transform=ax.transAxes, ha="center")
        title = f"Set {set_number}: Representative Samples"
        if reference_label:
            title += f"\n{reference_label}"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Frequency point index")
        ax.set_ylabel(
            "Relative phase (rad)" if channel_index in (8, 10)
            else CHANNEL_LABELS.get(channel_index, f"Channel {channel_index}")
        )
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        if traces:
            ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        if output_path is not None:
            requested_output = Path(output_path)
            suffix = requested_output.suffix or ".png"
            set_output = requested_output.with_name(
                f"{requested_output.stem}_set_{set_number}{suffix}"
            )
            set_output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(set_output, dpi=300, bbox_inches="tight")
        figures[set_number] = (fig, ax)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return figures, selections


def plot_representative_sets_by_concentration(
    representative_samples: dict[int, dict[str, dict]],
    *,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    """Overlay the existing set representatives in one figure per concentration.

    Reuse each stored repetition-averaged ``mean_trace`` exactly. No set
    subtraction, averaging, normalization, or new sample selection is done.
    Set colours and data-derived y limits are consistent across figures.
    The lower y-axis limit is fixed at -0.15.
    """
    set_numbers = sorted(representative_samples)
    concentrations = set()
    channel_indices = set()
    signal_lengths = set()
    finite_limits = []
    for set_number in set_numbers:
        for concentration, selection in representative_samples[set_number].items():
            if "mean_trace" not in selection or "channel_index" not in selection:
                raise ValueError(
                    "Rerun plot_representative_channel_samples to store selected mean traces"
                )
            channel_index = selection["channel_index"]
            if channel_index not in CIRCULAR_CHANNEL_INDICES:
                raise ValueError("Set phase comparisons require a phase channel")
            trace = np.asarray(selection["mean_trace"], dtype=np.float64)
            if trace.ndim != 1 or np.isinf(trace).any():
                raise ValueError(f"Invalid representative trace for set {set_number}, {concentration}")
            concentrations.add(concentration)
            channel_indices.add(channel_index)
            signal_lengths.add(len(trace))
            finite = trace[np.isfinite(trace)]
            if finite.size:
                finite_limits.extend((float(finite.min()), float(finite.max())))
    if not concentrations:
        raise ValueError("No representative samples are available")
    if len(channel_indices) > 1 or len(signal_lengths) > 1:
        raise ValueError("Representative curves must use one phase channel and matching lengths")

    directory = Path(output_dir) if output_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    color_map = plt.get_cmap("tab10")
    figures = {}
    for concentration in sorted(concentrations, key=_concentration_sort_key):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for set_number in set_numbers:
            if concentration not in representative_samples[set_number]:
                continue
            selection = representative_samples[set_number][concentration]
            trace = np.asarray(selection["mean_trace"], dtype=np.float64)
            frequency_indices = np.arange(len(trace), dtype=np.float64)
            breaks = np.flatnonzero(np.abs(np.diff(trace)) > np.pi) + 1
            ax.plot(
                np.insert(frequency_indices, breaks, np.nan),
                np.insert(trace, breaks, np.nan),
                color=color_map((set_number - 1) % 10),
                linewidth=1.3,
                label=f"Set {set_number} (S{selection['sample_number']})",
            )
        if finite_limits:
            lower, upper = min(finite_limits), max(finite_limits)
            margin = max((upper - lower) * 0.06, 1e-6)
            ax.set_ylim(lower - margin, upper + margin)
        ax.set_ylim(bottom=-0.15)
        exponent = _concentration_sort_key(concentration)
        ax.set_title(rf"Representative Samples: $10^{{-{exponent}}}$", fontsize=12)
        ax.set_xlabel("Frequency point index")
        channel_index = next(iter(channel_indices))
        ax.set_ylabel(
            "Relative phase (rad)" if channel_index in (8, 10)
            else CHANNEL_LABELS[channel_index]
        )
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        if directory is not None:
            filename = f"representative_sets_{concentration}.png"
            fig.savefig(directory / filename, dpi=300, bbox_inches="tight")
        figures[concentration] = (fig, ax)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return figures
