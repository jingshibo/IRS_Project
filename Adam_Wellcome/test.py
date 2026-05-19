from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from Functions import (
    append_pbs_reference_difference,
    apply_savgol_filter_nested,
    downsample_nested_data,
    load_liquid_folder,
    plot_all_liquids,
)
from Model import (
    AdamWellcomeTrainer,
    TrainerConfig,
    build_classification_dataset,
    build_stratified_cv_splits,
)

##
DATA_DIR = Path(r"/home/shibojing/data/adam wellcome")

data = load_liquid_folder(DATA_DIR)
difference_data = append_pbs_reference_difference(data)
plot_all_liquids(difference_data, channel_idx=6)

##
filtered_difference_data = apply_savgol_filter_nested(
    difference_data,
    window_length=31,
    polyorder=3,
)
downsample_ratio = 50
filtered_difference_data = downsample_nested_data(
    filtered_difference_data,
    ratio=downsample_ratio,
)
plot_all_liquids(filtered_difference_data, channel_idx=6)

##
RANDOM_SEED = 42
dataset = build_classification_dataset(
    filtered_difference_data,
    channel_indices=(6,), ## how about only amplitude? actually worse than phase and both amp and phase.
    label_mode="joint",
    include_reference=False,
)
cv_splits = build_stratified_cv_splits(
    dataset,
    test_ratio=0.15,
    n_splits=5,
    random_seed=RANDOM_SEED,
)

##
trainer = AdamWellcomeTrainer(
    TrainerConfig(
        epochs=100,
        batch_size=32,
        patience=20,
        verbose=True,
    )
)

fold_results = []
for split in cv_splits:
    print(f"Fold {split.fold + 1}/{len(cv_splits)}")
    fold_results.append(trainer.fit(split))

mean_val_acc = float(np.mean([result.best_val_acc for result in fold_results]))
mean_test_acc = float(np.mean([result.test_acc for result in fold_results]))
average_val_confusion_matrix = np.mean(
    [result.val_confusion_matrix.astype(np.float64) for result in fold_results],
    axis=0,
)
average_val_recall_matrix = np.divide(
    average_val_confusion_matrix,
    average_val_confusion_matrix.sum(axis=1, keepdims=True),
    out=np.zeros_like(average_val_confusion_matrix, dtype=np.float64),
    where=average_val_confusion_matrix.sum(axis=1, keepdims=True) != 0,
)

print(
    f"Mean best val acc={mean_val_acc:.4f}, "
    f"mean test acc={mean_test_acc:.4f}"
)
