from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np

CHANNEL_NAME_MAP = {
    4: "Re",
    5: "Im",
    6: "Amplitude",
    7: "Phase",
}




def _flatten_valid_traces(channel_block: np.ndarray) -> np.ndarray:
    flat = channel_block.reshape(-1, channel_block.shape[-1])
    valid_mask = ~np.isnan(flat).all(axis=1)
    return flat[valid_mask]


def select_representative_trace(channel_block: np.ndarray) -> np.ndarray:
    """
    Select the real trace closest to the concentration mean curve.
    """
    valid_traces = _flatten_valid_traces(channel_block)
    if valid_traces.size == 0:
        raise ValueError("No valid traces found in concentration block")

    mean_trace = np.nanmean(valid_traces, axis=0)
    distances = np.linalg.norm(valid_traces - mean_trace, axis=1)
    best_idx = int(np.argmin(distances))
    return valid_traces[best_idx]


def _compute_trace_summary(
    channel_block: np.ndarray,
    *,
    error_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    valid_traces = _flatten_valid_traces(channel_block)
    if valid_traces.size == 0:
        raise ValueError("No valid traces found in concentration block")

    mean_trace = np.nanmean(valid_traces, axis=0)
    spread = np.nanstd(valid_traces, axis=0)

    if error_mode.lower() == "sd":
        return mean_trace, spread
    if error_mode.lower() == "sem":
        sample_count = np.sum(~np.isnan(valid_traces), axis=0)
        sem = np.divide(
            spread,
            np.sqrt(sample_count),
            out=np.zeros_like(spread),
            where=sample_count > 0,
        )
        return mean_trace, sem

    raise ValueError("error_mode must be either 'sd' or 'sem'")


def plot_liquid_representatives(
    liquid_name: str,
    concentration_map: dict[str, np.ndarray],
    *,
    channel_idx: int,
) -> tuple[plt.Figure, np.ndarray]:
    concentrations = sorted(
        concentration_map,
        key=lambda value: tuple(int(part) for part in value.split("-"))
    )
    n_concentrations = len(concentrations)
    n_cols = min(4, n_concentrations)
    n_rows = ceil(n_concentrations / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    channel_name = CHANNEL_NAME_MAP.get(channel_idx, f"Channel {channel_idx}")
    fig.suptitle(f"{liquid_name} Representative Amplitude", fontsize=14)

    flat_axes = axes.ravel()
    for ax, concentration in zip(flat_axes, concentrations):
        channel_block = concentration_map[concentration][..., channel_idx]
        representative_trace = select_representative_trace(channel_block)
        ax.plot(representative_trace, linewidth=1.3)
        ax.set_ylim(top=0.02)
        ax.set_title(concentration)
        ax.set_xlabel("Point Index")
        ax.set_ylabel(f"Channel {channel_idx}")
        ax.grid(True, alpha=0.25)

    for ax in flat_axes[n_concentrations:]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_all_liquids(
    data: dict[str, dict[str, np.ndarray]],
    *, # everything after it must be passed by keyword, not by position.
    channel_idx: int = 2, # re:0, im:1, amplitude: 2, phase:3, re_diff:4, im_diff:5, amp_diff:6, phase_diff:7
) -> dict[str, tuple[plt.Figure, np.ndarray]]:
    figures = {}
    for liquid_name in sorted(data):
        figures[liquid_name] = plot_liquid_representatives(
            liquid_name,
            data[liquid_name],
            channel_idx=channel_idx,
        )
    return figures


def plot_liquids_by_concentration(
    data: dict[str, dict[str, np.ndarray]],
    concentration: str,
    *,
    channel_idx: int = 2,
) -> tuple[plt.Figure, np.ndarray]:
    available_liquids = [
        liquid_name for liquid_name in sorted(data)
        if concentration in data[liquid_name]
    ]
    if not available_liquids:
        raise ValueError(f"No liquids found for concentration '{concentration}'")

    n_liquids = len(available_liquids)
    n_cols = min(4, n_liquids)
    n_rows = ceil(n_liquids / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    channel_name = CHANNEL_NAME_MAP.get(channel_idx, f"Channel {channel_idx}")
    fig.suptitle(
        f"{concentration} Representative {channel_name} Across Liquids",
        fontsize=14,
    )

    flat_axes = axes.ravel()
    for ax, liquid_name in zip(flat_axes, available_liquids):
        channel_block = data[liquid_name][concentration][..., channel_idx]
        representative_trace = select_representative_trace(channel_block)
        ax.plot(representative_trace, linewidth=1.3)
        ax.set_ylim(top=0.025)
        ax.set_title(liquid_name)
        ax.set_xlabel("Point Index")
        ax.set_ylabel(f"Channel {channel_idx}")
        ax.grid(True, alpha=0.25)

    for ax in flat_axes[n_liquids:]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_liquids_by_concentration_overlay(
    data: dict[str, dict[str, np.ndarray]],
    concentration: str,
    *,
    channel_idx: int = 2,
) -> tuple[plt.Figure, plt.Axes]:
    available_liquids = [
        liquid_name for liquid_name in sorted(data)
        if concentration in data[liquid_name]
    ]
    if not available_liquids:
        raise ValueError(f"No liquids found for concentration '{concentration}'")

    fig, ax = plt.subplots(figsize=(10, 6))
    channel_name = CHANNEL_NAME_MAP.get(channel_idx, f"Channel {channel_idx}")

    for liquid_name in available_liquids:
        channel_block = data[liquid_name][concentration][..., channel_idx]
        representative_trace = select_representative_trace(channel_block)
        ax.plot(representative_trace, linewidth=1.3, label=liquid_name)

    ax.set_ylim(top=0.020)
    ax.set_title(f"{concentration} Representative Amplitude Across Liquids")
    ax.set_xlabel("Point Index")
    ax.set_ylabel(f"Channel {channel_idx}")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_selected_concentrations_overlay(
    data: dict[str, dict[str, np.ndarray]],
    concentrations: list[str],
    *,
    channel_idx: int = 2,
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    sorted_concentrations = sorted(
        concentrations,
        key=lambda value: tuple(int(part) for part in value.split("-")),
    )

    figures = {}
    for concentration in sorted_concentrations:
        figures[concentration] = plot_liquids_by_concentration_overlay(
            data,
            concentration,
            channel_idx=channel_idx,
        )
    return figures


def plot_selected_concentrations_by_liquid(
    data: dict[str, dict[str, np.ndarray]],
    concentrations: list[str],
    *,
    channel_idx: int = 2,
) -> dict[str, tuple[plt.Figure, np.ndarray]]:
    sorted_concentrations = sorted(
        concentrations,
        key=lambda value: tuple(int(part) for part in value.split("-")),
    )

    figures = {}
    for concentration in sorted_concentrations:
        figures[concentration] = plot_liquids_by_concentration(
            data,
            concentration,
            channel_idx=channel_idx,
        )
    return figures


def plot_concentrations_by_liquid_overlay(
    data: dict[str, dict[str, np.ndarray]],
    liquid_name: str,
    *,
    channel_idx: int = 2,
) -> tuple[plt.Figure, plt.Axes]:
    if liquid_name not in data:
        raise ValueError(f"Liquid '{liquid_name}' not found in data")

    concentration_map = data[liquid_name]
    concentrations = sorted(
        concentration_map,
        key=lambda value: tuple(int(part) for part in value.split("-")),
    )
    if not concentrations:
        raise ValueError(f"No concentrations found for liquid '{liquid_name}'")

    fig, ax = plt.subplots(figsize=(10, 6))
    channel_name = CHANNEL_NAME_MAP.get(channel_idx, f"Channel {channel_idx}")

    for concentration in concentrations:
        channel_block = concentration_map[concentration][..., channel_idx]
        representative_trace = select_representative_trace(channel_block)
        ax.plot(representative_trace, linewidth=1.3, label=concentration)

    ax.set_ylim(top=0.025)
    ax.set_title(f"{liquid_name} Representative Amplitude Across Concentrations")
    ax.set_xlabel("Point Index")
    ax.set_ylabel(f"Channel {channel_idx}")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Concentration")

    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_selected_liquids_concentration_overlay(
    data: dict[str, dict[str, np.ndarray]],
    liquid_names: list[str],
    *,
    channel_idx: int = 2,
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    figures = {}
    for liquid_name in liquid_names:
        figures[liquid_name] = plot_concentrations_by_liquid_overlay(
            data,
            liquid_name,
            channel_idx=channel_idx,
        )
    return figures


def plot_liquids_by_concentration_stats_overlay(
    data: dict[str, dict[str, np.ndarray]],
    concentration: str,
    *,
    channel_idx: int = 2,
    error_mode: str = "sd",
) -> tuple[plt.Figure, plt.Axes]:
    available_liquids = [
        liquid_name for liquid_name in sorted(data)
        if concentration in data[liquid_name]
    ]
    if not available_liquids:
        raise ValueError(f"No liquids found for concentration '{concentration}'")

    fig, ax = plt.subplots(figsize=(10, 6))
    channel_name = CHANNEL_NAME_MAP.get(channel_idx, f"Channel {channel_idx}")
    error_label = error_mode.upper()

    for liquid_name in available_liquids:
        channel_block = data[liquid_name][concentration][..., channel_idx]
        mean_trace, error_trace = _compute_trace_summary(
            channel_block,
            error_mode=error_mode,
        )
        (line,) = ax.plot(mean_trace, linewidth=1.5, label=liquid_name)
        ax.fill_between(
            np.arange(mean_trace.shape[0]),
            mean_trace - error_trace,
            mean_trace + error_trace,
            color=line.get_color(),
            alpha=0.18,
        )

    ax.set_ylim(top=0.025)
    ax.set_title(
        f"{concentration} Mean ± {error_label} {channel_name} Across Liquids"
    )
    ax.set_xlabel("Point Index")
    ax.set_ylabel(f"Channel {channel_idx}")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_selected_concentrations_stats_overlay(
    data: dict[str, dict[str, np.ndarray]],
    concentrations: list[str],
    *,
    channel_idx: int = 2,
    error_mode: str = "sd",
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    sorted_concentrations = sorted(
        concentrations,
        key=lambda value: tuple(int(part) for part in value.split("-")),
    )

    figures = {}
    for concentration in sorted_concentrations:
        figures[concentration] = plot_liquids_by_concentration_stats_overlay(
            data,
            concentration,
            channel_idx=channel_idx,
            error_mode=error_mode,
        )
    return figures


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_labels: list[str],
    *,
    title: str = "Test Confusion Matrix",
    cmap: str = "Blues",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(confusion_matrix, cmap=cmap)
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)

    threshold = float(confusion_matrix.max()) * 0.5 if confusion_matrix.size else 0.0
    for row_idx in range(confusion_matrix.shape[0]):
        for col_idx in range(confusion_matrix.shape[1]):
            value = confusion_matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    return fig, ax
