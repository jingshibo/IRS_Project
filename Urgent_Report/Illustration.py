from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(r"/home/shibojing/data/urgent_report")
DATA_FILES = (
    "Mozambique.csv",
)
CLASS_ORDER = ("Low", "Target", "High")
OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "illustrations"

CLASS_COLORS = {
    "Low": "#2f6fbb",
    "Target": "#c9822a",
    "High": "#2f8f5b",
}


def load_urgent_report_data(
    data_dir: Path,
    file_names: tuple[str, ...],
) -> tuple[list[str], dict[str, pd.DataFrame]]:
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
            raise ValueError(f"No sensor columns found after '{label_col}' in {file_path}")

        if reference_sensor_columns is None:
            reference_sensor_columns = sensor_columns
        elif sensor_columns != reference_sensor_columns:
            raise ValueError(f"Sensor column mismatch in {file_path}")

        dataframes.append(df[[label_col, *sensor_columns]])

    if not dataframes:
        raise ValueError("No data files were loaded")

    df = pd.concat(dataframes, ignore_index=True)
    categorized_dict = {
        label: group[reference_sensor_columns].reset_index(drop=True)
        for label, group in df.groupby(label_col)
    }
    return reference_sensor_columns, categorized_dict


def plot_mean_signal_sem(
    categorized_dict: dict[str, pd.DataFrame],
    sensor_columns: list[str],
    output_dir: Path,
) -> None:
    x_axis = np.arange(len(sensor_columns))
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    for label in CLASS_ORDER:
        if label not in categorized_dict:
            continue
        values = categorized_dict[label][sensor_columns].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])

        color = CLASS_COLORS[label]
        ax.plot(x_axis, mean, label=f"{label} (n={values.shape[0]})", color=color, linewidth=1.8)
        ax.fill_between(x_axis, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)

    ax.set_title("Mozambique Mean Sensor Signal +/- SEM")
    ax.set_xlabel("Sensor index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "mean_signal_sem.png", dpi=300)
    plt.show()

def plot_pca_projection(
    categorized_dict: dict[str, pd.DataFrame],
    sensor_columns: list[str],
    output_dir: Path,
) -> None:
    values = []
    labels = []
    for label in CLASS_ORDER:
        if label not in categorized_dict:
            continue
        class_values = categorized_dict[label][sensor_columns].to_numpy(dtype=np.float32)
        values.append(class_values)
        labels.extend([label] * class_values.shape[0])

    x = np.vstack(values)
    labels = np.asarray(labels)
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(x_scaled)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for label in CLASS_ORDER:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=28,
            alpha=0.72,
            color=CLASS_COLORS[label],
            label=f"{label} (n={mask.sum()})",
            edgecolors="none",
        )

    explained = pca.explained_variance_ratio_ * 100.0
    ax.set_title("PCA Projection of Sensor Signals")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "pca_projection.png", dpi=300)
    plt.show()

def plot_pca_projection_3d(
    categorized_dict: dict[str, pd.DataFrame],
    sensor_columns: list[str],
    output_dir: Path,
) -> None:
    values = []
    labels = []
    for label in CLASS_ORDER:
        if label not in categorized_dict:
            continue
        class_values = categorized_dict[label][sensor_columns].to_numpy(dtype=np.float32)
        values.append(class_values)
        labels.extend([label] * class_values.shape[0])

    x = np.vstack(values)
    labels = np.asarray(labels)
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3, random_state=42)
    points = pca.fit_transform(x_scaled)

    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for label in CLASS_ORDER:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            points[mask, 2],
            s=26,
            alpha=0.72,
            color=CLASS_COLORS[label],
            label=f"{label} (n={mask.sum()})",
            depthshade=False,
        )

    explained = pca.explained_variance_ratio_ * 100.0
    ax.set_title("3D PCA Projection of Sensor Signals")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained[2]:.1f}%)")
    ax.view_init(elev=22, azim=42)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "pca_projection_3d.png", dpi=300)
    plt.show()

def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n_a = a.shape[0]
    n_b = b.shape[0]
    mean_diff = a.mean(axis=0) - b.mean(axis=0)
    pooled_var = (
        ((n_a - 1) * a.var(axis=0, ddof=1) + (n_b - 1) * b.var(axis=0, ddof=1))
        / max(n_a + n_b - 2, 1)
    )
    return mean_diff / np.sqrt(pooled_var + 1e-8)


def plot_pairwise_effect_sizes(
    categorized_dict: dict[str, pd.DataFrame],
    sensor_columns: list[str],
    output_dir: Path,
) -> None:
    pairs = (
        ("Low", "High"),
        ("Target", "High"),
        ("Low", "Target"),
    )
    x_axis = np.arange(len(sensor_columns))

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for label_a, label_b in pairs:
        if label_a not in categorized_dict or label_b not in categorized_dict:
            continue
        values_a = categorized_dict[label_a][sensor_columns].to_numpy(dtype=np.float32)
        values_b = categorized_dict[label_b][sensor_columns].to_numpy(dtype=np.float32)
        effect = cohens_d(values_a, values_b)
        ax.plot(x_axis, effect, linewidth=1.5, label=f"{label_a} vs {label_b}")

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.5)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.45)
    ax.axhline(-0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.45)
    ax.set_title("Pairwise Effect Size Across Sensor Index")
    ax.set_xlabel("Sensor index")
    ax.set_ylabel("Cohen's d")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "pairwise_effect_sizes.png", dpi=300)
    plt.show()

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sensor_columns, categorized_dict = load_urgent_report_data(DATA_DIR, DATA_FILES)

    plot_mean_signal_sem(categorized_dict, sensor_columns, OUTPUT_DIR)
    plot_pca_projection(categorized_dict, sensor_columns, OUTPUT_DIR)
    plot_pca_projection_3d(categorized_dict, sensor_columns, OUTPUT_DIR)
    plot_pairwise_effect_sizes(categorized_dict, sensor_columns, OUTPUT_DIR)

    print(f"Saved figures to: {OUTPUT_DIR}")



if __name__ == "__main__":
    main()
