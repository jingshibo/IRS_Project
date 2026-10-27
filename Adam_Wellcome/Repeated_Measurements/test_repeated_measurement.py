##
from __future__ import annotations

import importlib
import sys
from pathlib import Path


ADAM_WELLCOME_DIR = Path(__file__).resolve().parents[1]
if str(ADAM_WELLCOME_DIR) not in sys.path:
    sys.path.insert(0, str(ADAM_WELLCOME_DIR))

from Functions import (  # noqa: E402
    append_pbs_complex_difference,
    append_pbs_reference_difference,
    apply_savgol_filter_nested,
    apply_savgol_to_re_im_and_recompute,
    downsample_nested_data,
    load_liquid_folder,
)
from Repeated_Measurements import dataset, training  # noqa: E402
from Repeated_Measurements.evaluation import (  # noqa: E402
    plot_grouped_recall_matrix,
    plot_grouped_recall_matrix_with_numbers,
    print_repeated_measurement_summary,
    summarize_repeated_measurement_results,
)

##
DATA_DIR = Path(r"/home/shibojing/data/adam wellcome")
raw_data = load_liquid_folder(DATA_DIR)
filtered_data = apply_savgol_to_re_im_and_recompute(
    raw_data,
    window_length=301,
    polyorder=3,
)

##
difference_data = append_pbs_reference_difference(filtered_data)
difference_data = append_pbs_complex_difference(difference_data)
filtered_difference_data = apply_savgol_filter_nested(
    difference_data,
    window_length=301,
    polyorder=3,
)
downsampled_difference_data = downsample_nested_data(
    filtered_difference_data,
    ratio=100,
)

##
data = dataset.build_repeated_measurement_dataset(
    downsampled_difference_data,
    channel_indices=(9, ),
    label_mode="liquid", # Use "joint" for liquid-concentration classes.
    include_reference=False,
)
cv_splits = dataset.build_leave_one_sample_out_splits(data)

print(
    f"Built {len(cv_splits)} leave-one-sample-out folds with "
    f"{len(data.sample_groups)} independent samples and {len(data.x)} repeated measurements."
)
for split in cv_splits:
    print(
        f"Fold {split.fold + 1}: "
        f"train measurements={len(split.x_train)}, "
        f"test measurements={len(split.x_test)}, "
        f"train samples={len(set(split.train_group_ids.tolist()))}, "
        f"test samples={len(set(split.test_group_ids.tolist()))}"
    )

##
importlib.reload(training)
trainer = training.RepeatedMeasurementTrainer(
    training.TrainerConfig(
        epochs=1000,
        batch_size=128,
        verbose=True,
        lr_scheduler_name="cosine",
        early_stopping_metric="sample_probability_test_acc",
        early_stopping_patience=300,
    )
)

fold_results = []
for split in cv_splits:
    print(f"Fold {split.fold + 1}/{len(cv_splits)}")
    fold_results.append(trainer.fit(split))

result_summary = summarize_repeated_measurement_results(fold_results)
print_repeated_measurement_summary(result_summary)
plot_grouped_recall_matrix(result_summary, matrix_name="sample_probability")
plot_grouped_recall_matrix_with_numbers(result_summary, matrix_name="sample_probability")
