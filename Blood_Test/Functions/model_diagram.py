"""Display an architecture figure from the actual blood CNN layers."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch

from Blood_Test.Model.model import BloodConcentrationCNN1D


def _box(ax, x, y, width, height, *, edge, face, dashed=False):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.0,rounding_size=2.2",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=0 if dashed else 2,
    )
    ax.add_patch(patch)
    return patch


def _text(ax, x, y, label, *, size=11, color="#17202a", weight="normal"):
    return ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        fontweight=weight,
        zorder=3,
    )


def _shape_label(shape):
    return "[Batch, " + ", ".join(str(value) for value in shape[1:]) + "]"


def plot_model_architecture(
    *,
    in_channels: int = 1,
    frequency_points: int = 200,
    channel_label: str = "Channel 10",
    show: bool = True,
) -> plt.Figure:
    """Display the three-class CNN diagram and return its figure.

    Dimensions and layer labels come from a temporary model forward pass.
    The diagram describes one measurement; sample-level probability averaging
    is performed later by the trainer and is not a layer of the network.
    """
    if in_channels < 1 or frequency_points < 1:
        raise ValueError("in_channels and frequency_points must be positive")

    # Preserve the caller's random state: producing a figure must not change
    # the initialization of any subsequently trained model.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = BloodConcentrationCNN1D(
            in_channels=in_channels,
            num_classes=3,
        ).eval()
        value = torch.zeros(1, in_channels, frequency_points)
        block_shapes = []
        for block in model.features:
            value = block(value)
            block_shapes.append(tuple(value.shape))
        flat = model.classifier[0](value)
        hidden = model.classifier[1](flat)
        logits = model.classifier[4](hidden)

    blue = "#2451c5"
    green = "#21813a"
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
        }
    ):
        fig, ax = plt.subplots(figsize=(20, 6.6), facecolor="white")
        fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.98)
        ax.set(xlim=(0, 300), ylim=(0, 100))
        ax.axis("off")

        _text(
            ax,
            150,
            96,
            "Blood Concentration Classification — 1D CNN",
            size=18,
            weight="bold",
        )
        _box(ax, 40, 12, 134, 76, edge=blue, face="#fafcff", dashed=True)
        _box(ax, 178, 12, 120, 76, edge=green, face="#fbfefb", dashed=True)
        _text(ax, 107, 82, "Feature Extractor", size=17, color=blue, weight="bold")
        _text(ax, 238, 82, "Classifier", size=17, color=green, weight="bold")

        _box(ax, 2, 30, 33, 34, edge="#323942", face="#fafbfc")
        _text(ax, 18.5, 57, "Input", size=13, weight="bold")
        _text(ax, 18.5, 49, channel_label, size=11)
        _text(ax, 18.5, 42, f"{frequency_points} frequency points", size=10)
        _text(ax, 18.5, 35, f"[Batch, {in_channels}, {frequency_points}]", size=10)

        for index, (block, shape) in enumerate(zip(model.features, block_shapes)):
            x = 43 + 44 * index
            center = x + 20
            conv, _, activation, pool, dropout = block
            _box(ax, x, 19, 40, 54, edge=blue, face="#f0f5ff")
            _text(ax, center, 67, f"Conv Block {index + 1}", size=13, weight="bold")
            _text(
                ax, center, 58,
                f"Conv1D: {conv.in_channels} → {conv.out_channels}",
                size=11,
            )
            _text(
                ax, center, 52,
                f"k={conv.kernel_size[0]}, s={conv.stride[0]}, p={conv.padding[0]}",
                size=10,
            )
            _text(ax, center, 45, "BatchNorm1D")
            _text(ax, center, 39, f"LeakyReLU (α={activation.negative_slope:g})", size=10)
            _text(ax, center, 33, f"MaxPool1D (k={pool.kernel_size}, s={pool.stride})", size=10)
            _text(ax, center, 27, f"Dropout (p={dropout.p:g})", size=10)
            _text(ax, center, 21.5, f"Output: {_shape_label(shape)}", size=9.5, weight="bold")

        _box(ax, 181, 30, 29, 34, edge=green, face="#f0faef")
        _text(ax, 195.5, 53, "Flatten", size=13, weight="bold")
        _text(ax, 195.5, 43, _shape_label(tuple(flat.shape)), size=10)

        _box(ax, 215, 24, 40, 46, edge=green, face="#f0faef")
        _text(ax, 235, 63, "Fully Connected", size=12, weight="bold")
        linear = model.classifier[1]
        activation = model.classifier[2]
        dropout = model.classifier[3]
        _text(ax, 235, 54, f"Linear: {linear.in_features} → {linear.out_features}")
        _text(ax, 235, 46, f"LeakyReLU (α={activation.negative_slope:g})", size=10)
        _text(ax, 235, 38, f"Dropout (p={dropout.p:g})", size=10)
        _text(ax, 235, 29, f"Output: {_shape_label(tuple(hidden.shape))}", size=9.5, weight="bold")

        _box(ax, 260, 29, 35, 36, edge=green, face="#f0faef")
        _text(ax, 277.5, 58, "Output Layer", size=12, weight="bold")
        output = model.classifier[4]
        _text(ax, 277.5, 50, f"Linear: {output.in_features} → {output.out_features}")
        _text(ax, 277.5, 42, f"Logits: {_shape_label(tuple(logits.shape))}", size=10)
        _text(ax, 277.5, 34, r"$10^{-4}$     $10^{-5}$     $10^{-6}$", size=12)
        _text(ax, 277.5, 22, "Softmax at inference", size=9.5, color=green)

        for start, end in ((35, 43), (83, 87), (127, 131), (171, 181), (210, 215), (255, 260)):
            ax.add_patch(
                FancyArrowPatch(
                    (start, 47), (end, 47),
                    arrowstyle="-|>", mutation_scale=17,
                    linewidth=1.6, color="#17202a",
                    shrinkA=0.5, shrinkB=0.5, zorder=4,
                )
            )

        _text(
            ax, 150, 4,
            "Shapes: [Batch, Channels, Frequency] before flattening  •  "
            "Convolutions operate along frequency  •  k: kernel, s: stride, p: padding",
            size=10, color="#526071",
        )

        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig
