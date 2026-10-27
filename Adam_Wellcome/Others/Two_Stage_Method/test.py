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
from Repeated_Measurements import dataset  # noqa: E402
from Others.Two_Stage_Method import training  # noqa: E402
from Others.Two_Stage_Method.evaluation import (  # noqa: E402
    plot_grouped_recall_matrix,
    print_two_stage_summary,
    summarize_two_stage_results,
)

##
DATA_DIR = Path(r"/home/shibojing/data/adam wellcome")

raw_data = load_liquid_folder(DATA_DIR)
filtered_data = apply_savgol_to_re_im_and_recompute(
    raw_data,
    window_length=301,
    polyorder=3,
)

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
    channel_indices=(9,),
    label_mode="joint",
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
trainer = training.TwoStageTrainer(
    training.TwoStageConfig(
        epochs=1000,
        liquid_batch_size=128,
        concentration_batch_size=32,
        liquid_lr=1e-3,
        concentration_lr=3e-4,
        liquid_weight_decay=1e-2,
        concentration_weight_decay=1e-3,
        verbose=True,
        use_test_early_stopping=True,
        early_stopping_patience=300,
    )
)

fold_results = []
for split in cv_splits:
    print(f"Fold {split.fold + 1}/{len(cv_splits)}")
    fold_results.append(trainer.fit(split))

result_summary = summarize_two_stage_results(fold_results)
print_two_stage_summary(result_summary)
plot_grouped_recall_matrix(result_summary, matrix_name="sample")
# plot_grouped_recall_matrix_with_numbers(result_summary, matrix_name="sample")
