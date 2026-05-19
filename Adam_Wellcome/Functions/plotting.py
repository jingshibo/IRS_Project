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
    fig.suptitle(f"{liquid_name} Representative {channel_name}", fontsize=14)

    flat_axes = axes.ravel()
    for ax, concentration in zip(flat_axes, concentrations):
        channel_block = concentration_map[concentration][..., channel_idx]
        representative_trace = select_representative_trace(channel_block)
        ax.plot(representative_trace, linewidth=1.3)
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
