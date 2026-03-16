from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

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


def plot_top_classification_examples(
    train_out,
    cv_folds,
    class_order=("LOW", "TARGET", "HIGH"),
    top_k=10,
    kind="misclassified",
    channel_idx=0,
    selected_class=None,
):
    """Plot top correct or misclassified validation samples per true class.

    The ranking score is the model confidence on the predicted class for
    misclassified samples, and the confidence on the true class for correctly
    classified samples.
    """
    if kind not in {"misclassified", "correct"}:
        raise ValueError(f"kind must be 'misclassified' or 'correct', got '{kind}'")
    if selected_class is not None and selected_class not in class_order:
        raise ValueError(
            f"selected_class must be one of {tuple(class_order)} or None, got '{selected_class}'"
        )

    rows = []
    idx_to_label = train_out["idx_to_label"]

    for fold_result, fold_data in zip(train_out["fold_results"], cv_folds):
        x_val = np.asarray(fold_data["X_val"], dtype=np.float32)
        y_true_idx = np.asarray(fold_result.y_true_idx)
        y_pred_idx = np.asarray(fold_result.y_pred_idx)
        y_prob = np.asarray(fold_result.y_prob)

        if channel_idx < 0 or channel_idx >= x_val.shape[1]:
            raise IndexError(f"channel_idx {channel_idx} is out of range for input with {x_val.shape[1]} channels")

        for sample_idx in range(len(y_true_idx)):
            true_idx = int(y_true_idx[sample_idx])
            pred_idx = int(y_pred_idx[sample_idx])
            is_correct = true_idx == pred_idx

            if kind == "correct" and not is_correct:
                continue
            if kind == "misclassified" and is_correct:
                continue

            score = float(y_prob[sample_idx, true_idx] if kind == "correct" else y_prob[sample_idx, pred_idx])
            rows.append(
                {
                    "fold": int(fold_result.fold),
                    "sample_idx": sample_idx,
                    "signal": x_val[sample_idx, channel_idx],
                    "true_idx": true_idx,
                    "pred_idx": pred_idx,
                    "true_label": idx_to_label[true_idx],
                    "pred_label": idx_to_label[pred_idx],
                    "score": score,
                }
            )

    classes_to_plot = [selected_class] if selected_class is not None else list(class_order)

    for cls in classes_to_plot:
        cls_rows = [row for row in rows if row["true_label"] == cls]
        cls_rows = sorted(cls_rows, key=lambda row: row["score"], reverse=True)[:top_k]

        if not cls_rows:
            print(f"No {kind} samples found for class {cls}")
            continue

        fig, axes = plt.subplots(len(cls_rows), 1, figsize=(12, 2.5 * len(cls_rows)), sharex=True)
        if len(cls_rows) == 1:
            axes = [axes]

        for ax, row in zip(axes, cls_rows):
            ax.plot(row["signal"], linewidth=1.2)
            ax.set_title(
                f"{cls} | fold={row['fold']} | sample={row['sample_idx']} | "
                f"true={row['true_label']} | pred={row['pred_label']} | score={row['score']:.4f}"
            )
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Point index")
        fig.tight_layout()
        plt.show()


def plot_reference_sample_from_fold(
    x_reference,
    cv_folds,
    fold_id,
    sample_idx,
    channel_idx=0,
    y_reference=None,
    title_prefix="Reference sample",
):
    """Plot one validation sample from an unnormalized reference array.

    `sample_idx` is the sample position inside the fold validation set, matching
    the indices shown by `plot_top_classification_examples`.
    """
    if fold_id < 0 or fold_id >= len(cv_folds):
        raise IndexError(f"fold_id {fold_id} is out of range for {len(cv_folds)} folds")

    fold_data = cv_folds[fold_id]
    val_idx = np.asarray(fold_data["val_idx"])
    if sample_idx < 0 or sample_idx >= len(val_idx):
        raise IndexError(f"sample_idx {sample_idx} is out of range for fold {fold_id} with {len(val_idx)} validation samples")

    x_reference = np.asarray(x_reference)
    global_idx = int(val_idx[sample_idx])
    if channel_idx < 0 or channel_idx >= x_reference.shape[1]:
        raise IndexError(f"channel_idx {channel_idx} is out of range for input with {x_reference.shape[1]} channels")

    signal = x_reference[global_idx, channel_idx]
    plt.figure(figsize=(12, 3.5))
    plt.plot(signal, linewidth=1.2)

    title = f"{title_prefix} | fold={fold_id} | sample={sample_idx} | global_idx={global_idx} | channel={channel_idx}"
    if y_reference is not None:
        y_reference = np.asarray(y_reference)
        title += f" | label={y_reference[global_idx]}"
    plt.title(title)
    plt.xlabel("Point index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fold_sample(
    cv_folds,
    fold_id,
    sample_idx,
    channel_idx=0,
    split="val",
    title_prefix="Fold sample",
    train_out=None,
):
    """Plot one sample directly from a normalized CV fold split.

    `sample_idx` is the position inside the selected fold split, e.g. the same
    validation-set indexing used by `plot_top_classification_examples` when
    `split="val"`.
    """
    if fold_id < 0 or fold_id >= len(cv_folds):
        raise IndexError(f"fold_id {fold_id} is out of range for {len(cv_folds)} folds")
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    fold_data = cv_folds[fold_id]
    x_key = "X_train" if split == "train" else "X_val"
    y_key = "y_train" if split == "train" else "y_val"
    idx_key = "train_idx" if split == "train" else "val_idx"

    x_split = np.asarray(fold_data[x_key])
    y_split = np.asarray(fold_data[y_key])
    split_indices = np.asarray(fold_data[idx_key])

    if sample_idx < 0 or sample_idx >= len(x_split):
        raise IndexError(
            f"sample_idx {sample_idx} is out of range for fold {fold_id} {split} split with {len(x_split)} samples"
        )
    if channel_idx < 0 or channel_idx >= x_split.shape[1]:
        raise IndexError(f"channel_idx {channel_idx} is out of range for input with {x_split.shape[1]} channels")

    signal = x_split[sample_idx, channel_idx]
    global_idx = int(split_indices[sample_idx])

    title = (
        f"{title_prefix} | fold={fold_id} | split={split} | sample={sample_idx} | "
        f"global_idx={global_idx} | channel={channel_idx} | label={y_split[sample_idx]}"
    )

    if train_out is not None and split == "val":
        fold_results = train_out.get("fold_results", [])
        idx_to_label = train_out.get("idx_to_label", {})
        if fold_id < len(fold_results):
            fold_result = fold_results[fold_id]
            y_pred_idx = np.asarray(fold_result.y_pred_idx)
            if sample_idx < len(y_pred_idx):
                pred_idx = int(y_pred_idx[sample_idx])
                pred_label = idx_to_label.get(pred_idx, pred_idx)
                title += f" | pred={pred_label}"

    plt.figure(figsize=(12, 3.5))
    plt.plot(signal, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Point index")
    plt.ylabel("Normalized value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fold_signals_grid(
    cv_folds,
    split="val",
    channel_idx=0,
    n_samples=16,
    nrows=4,
    ncols=4,
    title_prefix="Normalized fold signals",
):
    """Plot one figure per fold with multiple samples arranged as subplots."""
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if nrows < 1 or ncols < 1:
        raise ValueError(f"nrows and ncols must be >= 1, got nrows={nrows}, ncols={ncols}")

    max_plots = nrows * ncols
    n_to_plot = min(n_samples, max_plots)

    for fold_data in cv_folds:
        fold_id = int(fold_data["fold"])
        x_key = "X_train" if split == "train" else "X_val"
        y_key = "y_train" if split == "train" else "y_val"
        idx_key = "train_idx" if split == "train" else "val_idx"

        x_split = np.asarray(fold_data[x_key])
        y_split = np.asarray(fold_data[y_key])
        split_indices = np.asarray(fold_data[idx_key])

        if channel_idx < 0 or channel_idx >= x_split.shape[1]:
            raise IndexError(f"channel_idx {channel_idx} is out of range for input with {x_split.shape[1]} channels")

        actual_n = min(n_to_plot, len(x_split))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for sample_idx in range(actual_n):
            ax = axes[sample_idx]
            signal = x_split[sample_idx, channel_idx]
            global_idx = int(split_indices[sample_idx])
            ax.plot(signal, linewidth=1.0)
            ax.set_title(
                f"sample={sample_idx} | global={global_idx} | label={y_split[sample_idx]}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)

        for ax in axes[actual_n:]:
            ax.axis("off")

        fig.suptitle(
            f"{title_prefix} | fold={fold_id} | split={split} | channel={channel_idx}",
            fontsize=14,
        )
        fig.tight_layout()
        plt.show()


def plot_threshold_hit_signals_by_fold(
    cv_folds,
    threshold_hits,
    split=None,
    channel_idx=0,
    nrows=4,
    ncols=4,
    title_prefix="Threshold-hit signals",
):
    """Plot one figure per fold containing only threshold-hit samples."""
    if split not in {None, "train", "val"}:
        raise ValueError(f"split must be None, 'train' or 'val', got '{split}'")
    if nrows < 1 or ncols < 1:
        raise ValueError(f"nrows and ncols must be >= 1, got nrows={nrows}, ncols={ncols}")

    hits_by_fold = {}
    for row in threshold_hits:
        row_split = row.get("split")
        if split is not None and row_split != split:
            continue
        if int(row.get("channel_idx", -1)) != channel_idx:
            continue
        fold_id = int(row["fold"])
        hits_by_fold.setdefault(fold_id, []).append(row)

    for fold_data in cv_folds:
        fold_id = int(fold_data["fold"])
        fold_hits = hits_by_fold.get(fold_id, [])
        if not fold_hits:
            continue

        max_plots = nrows * ncols

        for start in range(0, len(fold_hits), max_plots):
            batch_hits = fold_hits[start:start + max_plots]
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), sharex=True, sharey=True)
            axes = np.atleast_1d(axes).ravel()

            for ax, hit in zip(axes, batch_hits):
                sample_idx = int(hit["sample_idx"])
                hit_split = hit["split"]
                x_key = "X_train" if hit_split == "train" else "X_val"
                x_split = np.asarray(fold_data[x_key])
                signal = x_split[sample_idx, channel_idx]
                ax.plot(signal, linewidth=1.0)
                ax.set_title(
                    f"{hit_split} | sample={sample_idx} | global={hit['global_idx']} | "
                    f"label={hit['label']}",
                    fontsize=9,
                )
                hit_points = np.asarray(hit.get("hit_points", []), dtype=int)
                if hit_points.size:
                    ax.scatter(hit_points, signal[hit_points], s=10, color="red")
                ax.grid(True, alpha=0.3)

            for ax in axes[len(batch_hits):]:
                ax.axis("off")

            page = start // max_plots + 1
            split_label = "train+val" if split is None else split
            fig.suptitle(
                f"{title_prefix} | fold={fold_id} | split={split_label} | channel={channel_idx} | page={page}",
                fontsize=14,
            )
            fig.tight_layout()
            plt.show()
