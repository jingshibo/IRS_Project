##
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Model_Structure as UrgentModelStructure
from IRS_Insecticide_Residual.Raw_Data_Implementation.Models import Model_Training
from IRS_Insecticide_Residual.Utility_Functions import Preprocessing
from IRS_Insecticide_Residual.Raw_Data_Implementation.Functions import Viewing


##
DATA_DIR = Path(r"/home/shibojing/data/urgent_report")
DATA_FILES = (
    # "Mozambique.csv",
    "N1_RED_Ghana.csv", ## the best one
    # "N3_RED_Ghana.csv",
)


def load_urgent_report_data(
    data_dir: Path,
    file_names: tuple[str, ...],
) -> tuple[str, tuple[str, ...], list[str], dict[str, pd.DataFrame]]:
    dataframes = []
    reference_sensor_columns = None
    label_col = "Class"

    for file_name in file_names:
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        if label_col not in df.columns:
            raise ValueError(f"'{label_col}' column not found in {file_path}")

        class_index = df.columns.get_loc(label_col)
        sensor_columns = df.columns[class_index + 1 :].tolist()
        if not sensor_columns:
            raise ValueError(f"No sensor-reading columns found after 'Class' in {file_path}")

        if reference_sensor_columns is None:
            reference_sensor_columns = sensor_columns
        elif sensor_columns != reference_sensor_columns:
            raise ValueError(f"Sensor column mismatch in {file_path}")

        dataframes.append(df[[label_col, *sensor_columns]])

    if not dataframes:
        raise ValueError("No data files were loaded")

    df = pd.concat(dataframes, ignore_index=True)
    sensor_columns = reference_sensor_columns

    class_order = ("Low", "Target", "High")
    categorized_dict = {
        key: group[sensor_columns].reset_index(drop=True)
        for key, group in df.groupby(label_col)
    }

    return label_col, class_order, sensor_columns, categorized_dict

label_col, class_order, sensor_columns, categorized_dict = load_urgent_report_data(DATA_DIR, DATA_FILES)


## preprocessing
signal_segments = ((0, 4096), )
sliced_dict = Preprocessing.slice_dict_signal_segments(categorized_dict, segments=signal_segments)
sliced_filtered_dict = Preprocessing.fast_spike_filter_dict(
    sliced_dict,
    radius=3,
    transform="sqrt",
    method="fast",
    n_sigmas=3.0,
    k=4.0,
    min_threshold=1000.0,
)
original_filtered_dict = Preprocessing.apply_savgol_filter_dict(
    sliced_filtered_dict,
    window_length=31,
    polyorder=3,
    deriv=0,
    mode="mirror",
)
original_filtered_dict = Preprocessing.downsample_dict_signals(original_filtered_dict, step=10, offset=0)
central_diff_dict = Preprocessing.compute_central_diff_dict(original_filtered_dict)
central_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(
    central_diff_dict,
    window_length=31,
    polyorder=3,
    deriv=0,
    mode="mirror",
)
second_diff_dict = Preprocessing.compute_second_central_diff_dict(original_filtered_dict)
second_diff_filtered_dict = Preprocessing.apply_savgol_filter_dict(
    second_diff_dict,
    window_length=31,
    polyorder=3,
    deriv=0,
    mode="mirror",
)

rolling_variance_dict = Preprocessing.compute_rolling_variance_dict(original_filtered_dict, window_size=31)
derivative_energy_dict = Preprocessing.compute_derivative_energy_dict(central_diff_filtered_dict, window_size=31)
original_envelope_dict = Preprocessing.apply_savgol_filter_dict(original_filtered_dict, window_length=31, polyorder=3, deriv=0)
original_residual_dict = Preprocessing.calculate_residual_dict(original_filtered_dict, original_envelope_dict)


## create dataset
random_seed = 42
value_type_dicts = {
    "original": original_filtered_dict,
    "first_diff_filtered": central_diff_filtered_dict,
    "second_diff_filtered": second_diff_filtered_dict,
    "rolling_variance": rolling_variance_dict,
    "derivative_energy": derivative_energy_dict,
    "residual": original_residual_dict,
}
selected_value_types = ["original", "first_diff_filtered", "second_diff_filtered"]

x_all, y_all = Preprocessing.build_multi_channel_dataset(
    data_dict_map=value_type_dicts,
    selected_types=selected_value_types,
)

x_trainval, x_test, y_trainval, y_test = Preprocessing.split_holdout(
    x_all,
    y_all,
    test_size=0.02,
    random_seed=random_seed,
)

cv_folds = Preprocessing.build_normalized_cv_folds(
    x_trainval,
    y_trainval,
    n_splits=5,
    random_seed=random_seed,
    clip_max_value=None,
)


## model training
importlib.reload(UrgentModelStructure)
importlib.reload(Model_Training)

cnn_layers = 4
cnn_channels = (32, 64, 128, 256)
cnn_kernel_sizes = (5, 5, 3, 3)
imbalance_strategy = "weighted_sampler"  # "none", "class_weight", "soft_class_weight", "soft_class_weight_focal_loss", "manual_class_weight", "weighted_sampler", "class_weight_and_sampler", "focal_loss", or "class_weight_focal_loss"
focal_gamma = 0.5 ## 0.5 = mild focal effect, 1.0 = moderate focal effect, 2.0 = strong focal effect
class_weight_beta = 1 # beta=1.0 is full class weighting. beta=0.0 is no weighting. So 0.25 or 0.5 gives a middle ground.
manual_class_weights = {
    "Low": 10,
    "Target": 5,
    "High": 2,
}
train_class_sample_limits = {
    "High": 150,
}
train_class_sample_seed = random_seed

def build_urgent_cnn(in_channels: int, num_classes: int):
    return UrgentModelStructure.FlexibleOneDCNNClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        cnn_layers=cnn_layers,
        channels=cnn_channels,
        kernel_sizes=cnn_kernel_sizes,
    )


Model_Training.OneDCNNClassifier = build_urgent_cnn
trainer_model_name = "shared_backbone_2ch"
model_name = f"urgent_flexible_cnn_{cnn_layers}layers_{imbalance_strategy}"
train_out: Model_Training.TrainOutput = Model_Training.train_1d_cnn_cv(
    cv_folds=cv_folds,
    class_order=class_order,
    model_name=trainer_model_name,
    epochs=100,
    batch_size=30,
    lr=1e-4,
    weight_decay=1e-4,
    label_smoothing=0.05,
    patience=40,
    random_shift_max_points=5,
    random_shift_fill_mode="wrap",
    tensorboard_log_dir="runs/1d_cnn_cv",
    imbalance_strategy=imbalance_strategy,
    focal_gamma=focal_gamma,
    class_weight_beta=class_weight_beta,
    manual_class_weights=manual_class_weights,
    train_class_sample_limits=train_class_sample_limits,
    train_class_sample_seed=train_class_sample_seed,
)
print("Mean best val acc:", train_out["mean_best_val_acc"])
print("Class index mapping:", train_out["label_to_idx"])
print("Model name:", model_name)


# confusion recall matrix plotting
manual_confusion_recall = None
# Example manual inputs:
# manual_confusion_recall = np.array([
#     [0.697, 0.105, 0.197],
#     [0.515, 0.273, 0.212],
#     [0.215, 0.076, 0.709],
# ])
# manual_confusion_recall = np.array([
#     [0.76, 0.04, 0.20],
#     [0.246, 0.6, 0.154],
#     [0.215, 0.051, 0.734],
# ])


def plot_confusion_matrix(ax, matrix: np.ndarray, labels: tuple[str, ...], title: str, value_format: str) -> None:
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "white" if value > threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{format(value * 100.0, value_format)}%",
                ha="center",
                va="center",
                color=text_color,
            )


confusion_recall = (
    np.asarray(manual_confusion_recall)
    if manual_confusion_recall is not None
    else train_out["overall_confusion_recall"]
)

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
plot_confusion_matrix(ax, confusion_recall, class_order, f"B Ghana: Acc=72.6%, BalAcc=72.0%", ".1f")
plt.show()

##

