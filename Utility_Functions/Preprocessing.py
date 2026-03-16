import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


def compute_mean_std_stats(grouped_dict):
    stats = {}
    for key, group in grouped_dict.items():
        stats[key] = {
            "mean": group.mean(),
            "std": group.std()
        }
    return stats


def compute_central_diff_dict(grouped_dict):
    central_diff_dict = {}
    for key, group_df in grouped_dict.items():
        cd = group_df.copy()
        cd.iloc[:, 1:-1] = (group_df.iloc[:, 2:].values - group_df.iloc[:, :-2].values) / 2
        # handle edges (simple approximation)
        cd.iloc[:, 0] = group_df.iloc[:, 1] - group_df.iloc[:, 0]
        cd.iloc[:, -1] = group_df.iloc[:, -1] - group_df.iloc[:, -2]
        central_diff_dict[key] = cd
    return central_diff_dict


def compute_second_central_diff_dict(grouped_dict):
    second_diff_dict = {}
    for key, df in grouped_dict.items():
        values = df.values
        if values.shape[1] < 3:
            raise ValueError(f"Need at least 3 features for second difference, got {values.shape[1]}")
        second_diff = np.zeros_like(values, dtype=float)
        # central second difference
        second_diff[:, 1:-1] = (values[:, 2:] - 2 * values[:, 1:-1] + values[:, :-2])
        # handle edges (simple approximation)
        second_diff[:, 0] = values[:, 2] - 2 * values[:, 1] + values[:, 0]
        second_diff[:, -1] = values[:, -1] - 2 * values[:, -2] + values[:, -3]
        second_diff_dict[key] = pd.DataFrame(second_diff, columns=df.columns, index=df.index)
    return second_diff_dict


def apply_savgol_filter_dict(data_dict, window_length=11, polyorder=3, deriv=0, mode="interp"):
    filtered_dict = {}
    for key, data_df in data_dict.items():
        n_features = data_df.shape[1]
        safe_window = window_length if n_features >= window_length else (n_features if n_features % 2 == 1 else n_features - 1)
        if safe_window > polyorder and safe_window >= 3:
            filtered_values = savgol_filter(data_df.values, window_length=safe_window, polyorder=polyorder, deriv=deriv, mode=mode)
            filtered_dict[key] = pd.DataFrame(filtered_values, columns=data_df.columns, index=data_df.index)
        else:
            filtered_dict[key] = data_df.copy()
    return filtered_dict


def apply_median_filter_dict(data_dict, kernel_size=5, mode="nearest"):
    """Apply a row-wise median filter to each class DataFrame in a dict."""
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

    filtered_dict = {}
    for key, data_df in data_dict.items():
        n_features = data_df.shape[1]
        safe_kernel = min(kernel_size, n_features)
        if safe_kernel % 2 == 0:
            safe_kernel -= 1

        if safe_kernel >= 1:
            filtered_values = median_filter(
                data_df.values,
                size=(1, safe_kernel),
                mode=mode,
            )
            filtered_dict[key] = pd.DataFrame(filtered_values, columns=data_df.columns, index=data_df.index)
        else:
            filtered_dict[key] = data_df.copy()
    return filtered_dict


def build_two_channel_dataset(original_dict, diff_filtered_dict):
    """Backward-compatible wrapper for two-channel dataset building."""
    return build_multi_channel_dataset(
        data_dict_map={
            "original": original_dict,
            "diff": diff_filtered_dict,
        },
        selected_types=("original", "diff"),
    )


def build_multi_channel_dataset(data_dict_map, selected_types):
    """Build [N, C, L] dataset from selected value-type dictionaries.

    Args:
        data_dict_map: mapping like {"original": original_dict, "first_diff": first_diff_dict, ...}
                      Each value is a dict[label -> DataFrame] with identical class keys and shapes.
        selected_types: ordered iterable of keys in data_dict_map to stack as channels.
    """
    selected_types = tuple(selected_types)
    if len(selected_types) == 0:
        raise ValueError("selected_types must contain at least one value type.")

    missing = [name for name in selected_types if name not in data_dict_map]
    if missing:
        raise KeyError(f"selected_types contains unknown value types: {missing}")

    selected_dicts = [data_dict_map[name] for name in selected_types]
    base_keys = set(selected_dicts[0].keys())
    for idx, data_dict in enumerate(selected_dicts[1:], start=1):
        if set(data_dict.keys()) != base_keys:
            raise ValueError(
                f"Class-key mismatch between selected value types: '{selected_types[0]}' and '{selected_types[idx]}'"
            )

    x_list = []
    y_list = []

    for label in selected_dicts[0].keys():
        channel_values = []
        base_shape = None
        for type_name, data_dict in zip(selected_types, selected_dicts):
            values = data_dict[label].values
            if base_shape is None:
                base_shape = values.shape
            elif values.shape != base_shape:
                raise ValueError(
                    f"Channel shape mismatch for class '{label}' in '{type_name}': {values.shape} vs {base_shape}"
                )
            channel_values.append(values)

        x = np.stack(channel_values, axis=1)
        y = np.full(base_shape[0], label)

        x_list.append(x)
        y_list.append(y)

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return x_all, y_all


def slice_and_concat_signal_segments(x, segments=((000, 1000), (1800, 3500))):
    """Slice selected index ranges from the last axis and concatenate them.

    Expected x shape: [N, C, L]
    Segments use Python slicing semantics: (start, end) keeps x[..., start:end]
    """
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"x must have shape [N, C, L], got {x.shape}")

    signal_length = x.shape[-1]
    pieces = []
    for start, end in segments:
        if not (0 <= start < end <= signal_length):
            raise ValueError(
                f"Invalid segment ({start}, {end}) for signal length {signal_length}"
            )
        pieces.append(x[:, :, start:end])

    return np.concatenate(pieces, axis=-1)


def slice_dict_signal_segments(data_dict, segments=((000, 1000), (1800, 3500))):
    """Slice and concatenate signal segments for each class DataFrame in a dict.

    Args:
        data_dict: dict[label -> pd.DataFrame], each row is one sample and columns are signal indices.
        segments: iterable of (start, end) ranges using Python slicing semantics.
    Returns:
        dict[label -> pd.DataFrame] with sliced/concatenated columns.
    """
    sliced_dict = {}
    for key, df in data_dict.items():
        values = df.values
        values_sliced = slice_and_concat_signal_segments(values[:, np.newaxis, :], segments=segments)[:, 0, :]
        # Keep original column labels when possible by concatenating selected column names.
        selected_cols = []
        n_features = df.shape[1]
        for start, end in segments:
            if not (0 <= start < end <= n_features):
                raise ValueError(f"Invalid segment ({start}, {end}) for dataframe width {n_features}")
            selected_cols.extend(df.columns[start:end].tolist())
        sliced_dict[key] = pd.DataFrame(values_sliced, columns=selected_cols, index=df.index)
    return sliced_dict


def downsample_dict_signals(data_dict, step=2, offset=0):
    """Downsample each class DataFrame in a dict by selecting every `step` column.

    Args:
        data_dict: dict[label -> pd.DataFrame]
        step: keep one column every `step` columns (must be >= 1)
        offset: start index before stepping (0 <= offset < step)
    Returns:
        dict[label -> pd.DataFrame] with downsampled columns.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if not (0 <= offset < step):
        raise ValueError(f"offset must satisfy 0 <= offset < step, got offset={offset}, step={step}")

    downsampled = {}
    for key, df in data_dict.items():
        downsampled[key] = df.iloc[:, offset::step].copy()
    return downsampled


def split_holdout(x_all, y_all, test_size=0.15, random_seed=42):
    return train_test_split(
        x_all,
        y_all,
        test_size=test_size,
        stratify=y_all,
        random_state=random_seed,
    )


def build_stratified_cv_indices(y_trainval, n_splits=5, random_seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    return list(skf.split(np.zeros(len(y_trainval)), y_trainval))


def normalize_fold_channels(x_train, x_val):
    x_train_norm = x_train.copy()
    x_val_norm = x_val.copy()

    n_channels = x_train_norm.shape[1]
    scalers = []
    for ch in range(n_channels):
        scaler = StandardScaler()
        scaler.fit(x_train_norm[:, ch, :])
        x_train_norm[:, ch, :] = scaler.transform(x_train_norm[:, ch, :])
        x_val_norm[:, ch, :] = scaler.transform(x_val_norm[:, ch, :])
        scalers.append(scaler)

    return x_train_norm, x_val_norm, scalers


def clip_normalized_fold_channels(x_train, x_val, max_value=15.0):
    """Apply elementwise upper clipping to normalized fold arrays."""
    return np.clip(x_train, a_min=None, a_max=max_value), np.clip(x_val, a_min=None, a_max=max_value)


def build_normalized_cv_folds(x_trainval, y_trainval, n_splits=5, random_seed=42, clip_max_value=None):
    folds = []
    cv_indices = build_stratified_cv_indices(y_trainval, n_splits=n_splits, random_seed=random_seed)

    for fold_id, (train_idx, val_idx) in enumerate(cv_indices):
        x_train = x_trainval[train_idx]
        y_train = y_trainval[train_idx]
        x_val = x_trainval[val_idx]
        y_val = y_trainval[val_idx]

        x_train_norm, x_val_norm, scalers = normalize_fold_channels(x_train, x_val)
        if clip_max_value is not None:
            x_train_norm, x_val_norm = clip_normalized_fold_channels(
                x_train_norm,
                x_val_norm,
                max_value=clip_max_value,
            )

        fold_payload = {
            "fold": fold_id,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "X_train": x_train_norm,
            "y_train": y_train,
            "X_val": x_val_norm,
            "y_val": y_val,
            "scalers": scalers,
            "clip_max_value": clip_max_value,
        }
        # Backward-compatible keys for legacy 2-channel code paths.
        if len(scalers) >= 1:
            fold_payload["scaler_raw"] = scalers[0]
        if len(scalers) >= 2:
            fold_payload["scaler_diff"] = scalers[1]
        if len(scalers) >= 3:
            fold_payload["scaler_second_diff"] = scalers[2]

        folds.append(fold_payload)

    return folds


def find_cv_fold_channel_threshold_hits(cv_folds, channel_idx=0, threshold=15.0):
    """Return samples whose normalized channel values exceed a threshold.

    Scans both train and validation splits for every fold. One result row is
    returned per sample with at least one point satisfying `value > threshold`.
    """
    hits = []

    for fold_data in cv_folds:
        fold_id = int(fold_data["fold"])

        for split in ("train", "val"):
            x_key = "X_train" if split == "train" else "X_val"
            y_key = "y_train" if split == "train" else "y_val"
            idx_key = "train_idx" if split == "train" else "val_idx"

            x_split = np.asarray(fold_data[x_key])
            y_split = np.asarray(fold_data[y_key])
            global_indices = np.asarray(fold_data[idx_key])

            if channel_idx < 0 or channel_idx >= x_split.shape[1]:
                raise IndexError(f"channel_idx {channel_idx} is out of range for input with {x_split.shape[1]} channels")

            channel_values = x_split[:, channel_idx, :]

            for sample_idx, signal in enumerate(channel_values):
                hit_points = np.flatnonzero(signal > threshold)
                if hit_points.size == 0:
                    continue

                hit_values = signal[hit_points]
                hits.append(
                    {
                        "fold": fold_id,
                        "split": split,
                        "sample_idx": int(sample_idx),
                        "global_idx": int(global_indices[sample_idx]),
                        "label": y_split[sample_idx],
                        "channel_idx": int(channel_idx),
                        "threshold": float(threshold),
                        "num_hit_points": int(hit_points.size),
                        "first_hit_point": int(hit_points[0]),
                        "max_value": float(hit_values.max()),
                        "max_point": int(hit_points[np.argmax(hit_values)]),
                        "hit_points": hit_points.tolist(),
                    }
                )

    return hits
