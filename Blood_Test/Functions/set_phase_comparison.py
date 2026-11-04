"""Compare PBS phase and the exact previously plotted representative curves."""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import PHASE_IDX
from .plotting import CIRCULAR_CHANNEL_INDICES, _concentration_sort_key


def _mean_complex_pbs(pbs_block: np.ndarray, set_number: int) -> np.ndarray:
    """Mean the valid complex PBS measurements, without averaging angles."""
    values = np.asarray(pbs_block[set_number - 1, ..., :2], dtype=np.float64)
    values = values.reshape(-1, values.shape[-2], 2)
    valid = ~np.isnan(values).all(axis=(1, 2))
    if not valid.any():
        return np.full(values.shape[1], complex(np.nan, np.nan))
    if not np.isfinite(values[valid]).all():
        raise ValueError(f"PBS set {set_number} contains a partial or nonfinite measurement")
    mean = values[valid].mean(axis=0)
    return mean[:, 0] + 1j * mean[:, 1]


def _phase_curve(ax, trace: np.ndarray, *, label: str, color):
    """Show wrapped phase without connecting opposite sides of +/-pi."""
    frequency_indices = np.arange(len(trace), dtype=np.float64)
    breaks = np.flatnonzero(np.abs(np.diff(trace)) > np.pi) + 1
    ax.plot(
        np.insert(frequency_indices, breaks, np.nan),
        np.insert(trace, breaks, np.nan),
        label=label,
        color=color,
        linewidth=1.3,
    )


def plot_pbs_representative_set_differences(
    pbs_data: dict[str, dict[str, np.ndarray]],
    representative_samples: dict[int, dict[str, dict]],
    *,
    include_representative_differences: bool = True,
    show: bool = True,
):
    """Overlay PBS set-pair differences, with separate representative figures.

    Pass ``filtered_data`` for PBS and the ``representative_samples`` returned
    by ``plot_representative_channel_samples``. PBS differences are the phase
    of mean_PBS_A * conj(mean_PBS_B). Representative differences are wrapped
    subtraction of the stored, repetition-averaged curves; no reselection or
    additional filtering occurs. Set A - Set B is the sign convention.

    Set ``include_representative_differences=False`` to calculate and plot
    only PBS differences, for use alongside direct representative overlays.
    The lower y-axis limit is -0.15; upper limits are derived from the data.

    Return (figures, differences). The PBS figure is stored under
    ``figures['pbs']['combined']``; representative figures retain one-based
    set-pair keys. Numerical differences retain their original set-pair keys,
    with representative arrays additionally keyed by concentration. Figures
    are displayed live; numerical results remain available in memory.
    """
    try:
        pbs_block = np.asarray(pbs_data["PBS"]["reference"])
    except KeyError as error:
        raise KeyError("pbs_data must contain ['PBS']['reference']") from error
    if pbs_block.ndim != 5 or pbs_block.shape[-1] < 2:
        raise ValueError("Expected PBS [set, sample, repetition, frequency, channel>=2]")
    set_numbers = sorted(representative_samples)
    if len(set_numbers) < 2:
        raise ValueError("At least two representative sets are required")
    if any(number < 1 or number > pbs_block.shape[0] for number in set_numbers):
        raise ValueError("Representative set numbers do not match available PBS sets")

    # Validate the original plot's stored curves rather than recovering or
    # selecting new samples from a potentially different processing branch.
    selected_channels = set()
    concentrations = set()
    for set_number, samples in representative_samples.items():
        for concentration, selection in samples.items():
            if "mean_trace" not in selection or "channel_index" not in selection:
                raise ValueError(
                    "Rerun plot_representative_channel_samples to store selected mean traces"
                )
            channel = selection["channel_index"]
            if channel not in CIRCULAR_CHANNEL_INDICES:
                raise ValueError("Representative comparisons require a phase channel")
            trace = np.asarray(selection["mean_trace"], dtype=np.float64)
            if trace.shape != (pbs_block.shape[-2],) or np.isinf(trace).any():
                raise ValueError(
                    f"Invalid or mismatched representative trace for set {set_number}, "
                    f"{concentration}"
                )
            selected_channels.add(channel)
            concentrations.add(concentration)
    if len(selected_channels) > 1:
        raise ValueError("All representative selections must use the same phase channel")

    mean_pbs = {
        number: _mean_complex_pbs(pbs_block, number) for number in set_numbers
    }
    differences = {"pbs": {}, "representative": {}}
    for pair in combinations(set_numbers, 2):
        first_set, second_set = pair
        first_pbs, second_pbs = mean_pbs[first_set], mean_pbs[second_set]
        pbs_difference = np.angle(first_pbs * np.conj(second_pbs))
        defined = (
            np.isfinite(first_pbs) & np.isfinite(second_pbs)
            & (np.abs(first_pbs) > 0) & (np.abs(second_pbs) > 0)
        )
        pbs_difference[~defined] = np.nan
        differences["pbs"][pair] = pbs_difference
        if not include_representative_differences:
            continue
        representative_differences = {}
        for concentration in sorted(concentrations, key=_concentration_sort_key):
            if (
                concentration not in representative_samples[first_set]
                or concentration not in representative_samples[second_set]
            ):
                continue
            first = np.asarray(
                representative_samples[first_set][concentration]["mean_trace"],
                dtype=np.float64,
            )
            second = np.asarray(
                representative_samples[second_set][concentration]["mean_trace"],
                dtype=np.float64,
            )
            representative_differences[concentration] = np.angle(np.exp(1j * (first - second)))
        differences["representative"][pair] = representative_differences

    concentrations = sorted(concentrations, key=_concentration_sort_key)
    color_map = plt.get_cmap("tab10")
    colors = {c: color_map(i % 10) for i, c in enumerate(concentrations)}
    figures = {"pbs": {}, "representative": {}}
    for family, pair_differences in differences.items():
        finite_limits = []
        for values in pair_differences.values():
            traces = [values] if family == "pbs" else values.values()
            for trace in traces:
                finite = trace[np.isfinite(trace)]
                if finite.size:
                    finite_limits.extend((float(finite.min()), float(finite.max())))
        plot_groups = (
            [("combined", pair_differences)]
            if family == "pbs" else pair_differences.items()
        )
        for figure_key, values in plot_groups:
            fig, ax = plt.subplots(figsize=(9, 4.6))
            if family == "pbs":
                for index, (pair, trace) in enumerate(values.items()):
                    first_set, second_set = pair
                    _phase_curve(
                        ax, trace,
                        label=f"Set {first_set} - Set {second_set}",
                        color=color_map(index % 10),
                    )
                title = "Baseline Phase Differences Between Sets"
            else:
                first_set, second_set = figure_key
                pair_label = f"Set {first_set} - Set {second_set}"
                for concentration, trace in values.items():
                    first_sample = representative_samples[first_set][concentration]["sample_number"]
                    second_sample = representative_samples[second_set][concentration]["sample_number"]
                    exponent = _concentration_sort_key(concentration)
                    _phase_curve(
                        ax, trace,
                        label=(rf"$10^{{-{exponent}}}$ "
                               f"(S{first_sample} - S{second_sample})"),
                        color=colors[concentration],
                    )
                title = f"Representative Phase Differences: {pair_label}"
                if not values:
                    ax.text(0.5, 0.5, "No shared concentration samples", transform=ax.transAxes, ha="center")
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.55)
            if finite_limits:
                lower, upper = min(0.0, min(finite_limits)), max(0.0, max(finite_limits))
                margin = max((upper - lower) * 0.06, 1e-6)
                ax.set_ylim(lower - margin, upper + margin)
            ax.set_ylim(bottom=-0.15)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Frequency point index")
            ax.set_ylabel("Phase difference (rad)")
            ax.grid(True, alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if family == "pbs" or values:
                ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            figures[family][figure_key] = (fig, ax)
            if show:
                plt.show()
            else:
                plt.close(fig)
    return figures, differences


def _selected_original_phase(filtered_data, concentration, set_number, sample_number):
    """Circularly average original phase repetitions for an existing sample ID."""
    block = np.asarray(filtered_data["SAU"][concentration])
    if block.ndim != 5 or block.shape[-1] != 4:
        raise ValueError(
            "Use filtered_data with four original channels "
            "[real, imaginary, amplitude, phase], before PBS correction"
        )
    if not (1 <= set_number <= block.shape[0] and 1 <= sample_number <= block.shape[1]):
        raise ValueError(f"Invalid selected sample: set {set_number}, S{sample_number}")
    phases = np.asarray(
        block[set_number - 1, sample_number - 1, ..., PHASE_IDX], dtype=np.float64
    )
    valid = ~np.isnan(phases).all(axis=-1)
    if not valid.any() or not np.isfinite(phases[valid]).all():
        raise ValueError(
            f"Selected sample {concentration}, set {set_number}, S{sample_number} "
            "has no complete finite phase repetitions"
        )
    resultant = np.exp(1j * phases[valid]).mean(axis=0)
    mean_phase = np.angle(resultant)
    mean_phase[np.abs(resultant) <= 1e-7] = np.nan
    return mean_phase


def plot_original_phase_set_differences(
    filtered_data: dict[str, dict[str, np.ndarray]],
    representative_samples: dict[int, dict[str, dict]],
    *,
    y_min: float | None = -0.15,
    show: bool = True,
):
    """Plot original-phase set differences in one figure per concentration.

    Reuse only the previously selected sample IDs, and read their ORIGINAL
    phase (channel 3) from filtered_data. The stored channel-8 mean_trace is
    deliberately not used. Circularly average each sample's repetitions,
    then compute wrapped Set A - Set B phase differences. No PBS correction,
    second smoothing, downsampling, or new sample selection is performed.

    Return (figures, differences), keyed by concentration. Difference arrays
    are additionally keyed by one-based set pair, e.g. differences['10-4'][(1,2)].
    Set-pair colours and upper limits are shared across figures. Set y_min=None
    to show the full data range. Figures are displayed live.
    """
    if "SAU" not in filtered_data:
        raise KeyError("filtered_data must contain SAU measurements")
    set_numbers = sorted(representative_samples)
    if len(set_numbers) < 2:
        raise ValueError("At least two selected sets are required")
    concentrations = sorted(
        {c for samples in representative_samples.values() for c in samples},
        key=_concentration_sort_key,
    )
    all_pairs = list(combinations(set_numbers, 2))
    differences = {}
    finite_limits = []
    for concentration in concentrations:
        if concentration not in filtered_data["SAU"]:
            raise KeyError(f"Original filtered SAU data missing for {concentration}")
        selected_phases = {}
        for set_number in set_numbers:
            if concentration not in representative_samples[set_number]:
                continue
            sample_number = representative_samples[set_number][concentration]["sample_number"]
            selected_phases[set_number] = _selected_original_phase(
                filtered_data, concentration, set_number, sample_number
            )
        curves = {}
        for pair in all_pairs:
            first_set, second_set = pair
            if first_set not in selected_phases or second_set not in selected_phases:
                continue
            first, second = selected_phases[first_set], selected_phases[second_set]
            curves[pair] = np.angle(np.exp(1j * (first - second)))
            finite = curves[pair][np.isfinite(curves[pair])]
            if finite.size:
                finite_limits.extend((float(finite.min()), float(finite.max())))
        if curves:
            differences[concentration] = curves
    if not differences:
        raise ValueError("No concentration has representative samples in at least two sets")

    color_map = plt.get_cmap("tab10")
    colors = {pair: color_map(index % 10) for index, pair in enumerate(all_pairs)}
    figures = {}
    for concentration, curves in differences.items():
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for pair, trace in curves.items():
            first_set, second_set = pair
            _phase_curve(
                ax, trace, color=colors[pair],
                label=f"Set {first_set} - Set {second_set}",
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.55)
        if finite_limits:
            lower, upper = min(0.0, min(finite_limits)), max(0.0, max(finite_limits))
            margin = max((upper - lower) * 0.06, 1e-6)
            ax.set_ylim(lower - margin, upper + margin)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        exponent = _concentration_sort_key(concentration)
        ax.set_title(rf"Original Filtered Phase Differences: $10^{{-{exponent}}}$", fontsize=12)
        ax.set_xlabel("Frequency point index")
        ax.set_ylabel("Phase difference (rad)")
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        figures[concentration] = (fig, ax)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return figures, differences
