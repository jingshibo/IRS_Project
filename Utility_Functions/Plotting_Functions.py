from typing import Optional

import matplotlib.pyplot as plt

def plot_single_sample(dataframe, row_index, label_column=None):
    if row_index < 0 or row_index >= len(dataframe):
        raise IndexError(f"row_index {row_index} is out of range (0 to {len(dataframe) - 1})")

    row = dataframe.iloc[row_index]
    if label_column and label_column in dataframe.columns:
        y = row.drop(labels=[label_column]).astype(float)
        title = f"Row {row_index} (label={row[label_column]})"
    else:
        y = row.astype(float)
        title = f"Row {row_index}"

    plt.figure(figsize=(10, 4))
    plt.plot(y.values, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Feature index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _pick_classes(data_dict, class_order, target_count):
    selected = [cls for cls in class_order if cls in data_dict]
    if len(selected) < target_count:
        for cls in data_dict.keys():
            if cls not in selected:
                selected.append(cls)
            if len(selected) == target_count:
                break
    return selected[:target_count]


def plot_mean_std_curves(
    stats_dict,
    class_order=("LOW", "TARGET", "HIGH"),
    title="Mean-Std Curves",
    ylabel="Value",
    ylim: Optional[tuple] = None,
):
    plt.figure(figsize=(10, 4))
    for cls in class_order:
        if cls in stats_dict:
            mean_curve = stats_dict[cls]["mean"]
            std_curve = stats_dict[cls]["std"]
            x = range(len(mean_curve))
            y = mean_curve.values
            s = std_curve.values

            plt.plot(x, y, linewidth=2, label=f"{cls} mean")
            plt.fill_between(x, y - s, y + s, alpha=0.2, label=f"{cls} +/-1 std")

    plt.title(title)
    plt.xlabel("Feature index")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_class_samples_vertical(
    data_dict,
    class_order=("LOW", "TARGET", "HIGH"),
    n_samples=5,
    title="Class Samples",
    ylabel="Value",
):
    n_classes = len(class_order)
    selected_classes = _pick_classes(data_dict, class_order, n_classes)
    if len(selected_classes) < n_classes:
        return

    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 4 * n_classes), sharex=True, sharey=True)
    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, selected_classes):
        class_df = data_dict[cls]
        n_to_plot = min(n_samples, len(class_df))
        for i in range(n_to_plot):
            sample = class_df.iloc[i]
            ax.plot(sample.values, linewidth=1.5, alpha=0.9, label=f"sample {i}")

        ax.set_title(f"{cls} (first {n_to_plot} samples)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=2, fontsize=8)

    axes[-1].set_xlabel("Feature index")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
