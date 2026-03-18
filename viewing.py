import pandas as pd

from Utility_Functions import Plotting_Functions, Preprocessing


def plot_threshold_hits(
    cv_folds,
    channel_idx=0,
    threshold=25.0,
    split=None,
    nrows=4,
    ncols=4,
    title_prefix="Normalized threshold-hit signals",
):
    threshold_hits = Preprocessing.find_cv_fold_channel_threshold_hits(
        cv_folds=cv_folds,
        channel_idx=channel_idx,
        threshold=threshold,
    )
    Plotting_Functions.plot_threshold_hit_signals_by_fold(
        cv_folds=cv_folds,
        threshold_hits=threshold_hits,
        split=split,
        channel_idx=channel_idx,
        nrows=nrows,
        ncols=ncols,
        title_prefix=title_prefix,
    )


def plot_classification_examples(
    train_out,
    cv_folds,
    class_order,
    classes=None,
    top_k=10,
    channel_idx=0,
):
    classes_to_plot = class_order if classes is None else classes
    for selected_class in classes_to_plot:
        for kind in ("misclassified", "correct"):
            Plotting_Functions.plot_top_classification_examples(
                train_out=train_out,
                cv_folds=cv_folds,
                class_order=class_order,
                top_k=top_k,
                kind=kind,
                channel_idx=channel_idx,
                selected_class=selected_class,
            )


def plot_certain_samples(
    sliced_dict,
    sliced_filtered_dict,
    x_trainval,
    y_trainval,
    cv_folds,
    fold_id,
    sample_idx,
    channel_idx=0,
    train_out=None,
):
    x_all_original_raw, y_all_original_raw = Preprocessing.build_multi_channel_dataset(
        data_dict_map={"original": sliced_dict},
        selected_types=("original",),
    )
    x_trainval_original_raw, _, y_trainval_original_raw, _ = Preprocessing.split_holdout(
        x_all_original_raw,
        y_all_original_raw,
        test_size=0.15,
        random_seed=42,
    )

    x_all_original_median, y_all_original_median = Preprocessing.build_multi_channel_dataset(
        data_dict_map={"original": sliced_filtered_dict},
        selected_types=("original",),
    )
    x_trainval_original_median, _, y_trainval_original_median, _ = Preprocessing.split_holdout(
        x_all_original_median,
        y_all_original_median,
        test_size=0.15,
        random_seed=42,
    )

    Plotting_Functions.plot_reference_sample_from_fold(
        x_reference=x_trainval_original_raw,
        y_reference=y_trainval_original_raw,
        cv_folds=cv_folds,
        fold_id=fold_id,
        sample_idx=sample_idx,
        channel_idx=channel_idx,
        title_prefix="Original raw signal before filtering",
    )
    Plotting_Functions.plot_reference_sample_from_fold(
        x_reference=x_trainval_original_median,
        y_reference=y_trainval_original_median,
        cv_folds=cv_folds,
        fold_id=fold_id,
        sample_idx=sample_idx,
        channel_idx=channel_idx,
        title_prefix="Original raw signal after median filtering",
    )
    Plotting_Functions.plot_reference_sample_from_fold(
        x_reference=x_trainval,
        y_reference=y_trainval,
        cv_folds=cv_folds,
        fold_id=fold_id,
        sample_idx=sample_idx,
        channel_idx=channel_idx,
        title_prefix="Unnormalized signal after SG filtering",
    )
    Plotting_Functions.plot_fold_sample(
        cv_folds=cv_folds,
        fold_id=fold_id,
        sample_idx=sample_idx,
        channel_idx=channel_idx,
        split="val",
        title_prefix="Normalized signal after SG filtering",
        train_out=train_out,
    )


def plot_mean_std_overview(
    class_order,
    original_dict,
    original_filtered_dict,
    central_diff_dict,
    central_diff_filtered_dict,
    second_diff_dict,
    second_diff_filtered_dict,
):
    stats_to_plot = (
        (original_dict, "Class Mean-Std Curves", "Mean value", None),
        (original_filtered_dict, "Filtered Class Mean-Std Curves", "Mean value", None),
        (central_diff_dict, "Central Difference Mean-Std Curves", "Central difference value", (-15, 15)),
        (
            central_diff_filtered_dict,
            "Filtered Central Difference Mean-Std Curves",
            "Central difference value",
            (-5, 5),
        ),
        (second_diff_dict, "Second Difference Mean-Std Curves", "Second difference value", (-10, 10)),
        (
            second_diff_filtered_dict,
            "Filtered Second Difference Mean-Std Curves",
            "Second difference value",
            (-1, 1),
        ),
    )

    for data_dict, title, ylabel, ylim in stats_to_plot:
        stats = Preprocessing.compute_mean_std_stats(data_dict)
        Plotting_Functions.plot_mean_std_curves(
            stats,
            class_order=class_order,
            title=title,
            ylabel=ylabel,
            ylim=ylim,
        )


def plot_random_sample_overview(
    class_order,
    random_seed,
    sliced_dict,
    sliced_filtered_dict,
    original_filtered_dict,
    original_envelope_dict,
    original_residual_dict,
    central_diff_dict,
    central_diff_filtered_dict,
    second_diff_dict,
    second_diff_filtered_dict,
    classes=None,
    n_samples=30,
    ncols=6,
):
    datasets = (
        (sliced_dict, "Raw Signals Before Filtering"),
        (sliced_filtered_dict, "Raw Signals After Hampel Filtering"),
        (original_filtered_dict, "Raw Signals After SG Filtering"),
        (original_envelope_dict, "Original Envelope Signals"),
        (original_residual_dict, "Original Residual Signals"),
        (central_diff_dict, "Raw Central Difference Signals"),
        (central_diff_filtered_dict, "Filtered Central Difference Signals"),
        (second_diff_dict, "Raw Second Difference Signals"),
        (second_diff_filtered_dict, "Filtered Second Difference Signals"),
    )

    for data_dict, title_prefix in datasets:
        Plotting_Functions.plot_random_class_samples_subplots(
            data_dict,
            class_order=class_order,
            selected_classes=classes,
            n_samples=n_samples,
            ncols=ncols,
            title_prefix=title_prefix,
            ylabel="Raw value",
            random_seed=random_seed,
        )


def inspect_normalized_data(
    cv_folds,
    class_order,
    selected_value_types,
    fold_id=0,
    n_samples=5,
):
    fold_data = cv_folds[fold_id]
    x_train_norm = fold_data["X_train"]
    y_train_norm = fold_data["y_train"]

    for channel_idx, value_type_name in enumerate(selected_value_types):
        normalized_channel_dict = {}
        for cls in class_order:
            class_mask = y_train_norm == cls
            if class_mask.any():
                normalized_channel_dict[cls] = pd.DataFrame(x_train_norm[class_mask, channel_idx, :])

        normalized_channel_stats = Preprocessing.compute_mean_std_stats(normalized_channel_dict)
        pretty_name = value_type_name.replace("_", " ").title()

        Plotting_Functions.plot_mean_std_curves(
            normalized_channel_stats,
            class_order=class_order,
            title=f"Normalized {pretty_name} Mean-Std Curves (Fold {fold_id} Train)",
            ylabel=f"Normalized {value_type_name} value",
        )
        Plotting_Functions.plot_class_samples_vertical(
            normalized_channel_dict,
            class_order=class_order,
            n_samples=n_samples,
            title=f"Normalized {pretty_name} Samples by Class (Fold {fold_id} Train)",
            ylabel=f"Normalized {value_type_name} value",
        )
