##
from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "IRS_Insecticide_Residual").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find the IRS_Project root. Run this file from inside the project folder."
    )


if "__file__" in globals():
    SCHOOL_VISIT_DIR = Path(__file__).resolve().parent
else:
    PROJECT_ROOT_FROM_CWD = _find_project_root(Path.cwd().resolve())
    SCHOOL_VISIT_DIR = PROJECT_ROOT_FROM_CWD / "IRS_Insecticide_Residual" / "School Visit"

PROJECT_ROOT = _find_project_root(SCHOOL_VISIT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCHOOL_VISIT_DIR) not in sys.path:
    sys.path.insert(0, str(SCHOOL_VISIT_DIR))

from school_visit_demo.classifier import train_demo_classifier, train_pca_demo_classifier
from school_visit_demo.config import CLASS_ORDER, RANDOM_SEED
from school_visit_demo.cnn_classifier import train_cnn_demo_classifier
from school_visit_demo.data_pipeline import (
    build_demo_signal_data_from_grouped,
    load_grouped_insecticide_data,
)
from school_visit_demo.features import (
    extract_complex_demo_features,
    extract_simple_demo_features,
)
from school_visit_demo.plots import (
    display_class_label,
    find_abnormal_spike_examples,
    plot_classifier_result_comparison,
    plot_cnn_feature_learning_comparison,
    plot_cnn_feature_learning_comparison_3d,
    plot_cnn_feature_learning_comparison_3d_interactive_html,
    plot_individual_raw_signal_subplots,
    plot_processed_class_average_before_after_slicing,
    plot_raw_signals,
    plot_spike_removal_examples,
    plot_unknown_classification_game_html,
    select_raw_sample_indices,
    plot_unknown_prediction,
)


def log(message: str = "") -> None:
    print(message, flush=True)


def log_saved(path: Path) -> None:
    log(f"Saved: {path}")


## editable demo settings
excel_path = None  # use None to check the default lab paths
sheet_name = 0
label_col = None  # use None to treat the first Excel column as the label column
output_dir = SCHOOL_VISIT_DIR / "outputs"
test_size = 0.20
unknown_test_position = None  # use None to choose a confident correct holdout example
random_seed = RANDOM_SEED

cnn_model_name = "shared_backbone_2ch"
cnn_epochs = 30
cnn_batch_size = 32
cnn_lr = 1e-4
cnn_weight_decay = 1e-4
cnn_label_smoothing = 0.3
cnn_patience = 10
cnn_n_splits = 5
cnn_verbose = True


## load insecticide residual data
output_dir.mkdir(parents=True, exist_ok=True)
log(f"Output folder: {output_dir}")

log("Loading insecticide residual data...")
log("Reading the Excel file can take 20-60 seconds on some machines...")
(
    data_path,
    resolved_label_col,
    class_order,
    raw_by_class,
    removed_zero_sample_indices,
) = load_grouped_insecticide_data(
    data_path=excel_path,
    sheet_name=sheet_name,
    label_col=label_col,
    class_order=CLASS_ORDER,
)

raw_sample_count = sum(len(data) for data in raw_by_class.values())
log(f"Data file: {data_path}")
log(f"Label column: {resolved_label_col}")
log(f"Display classes: {', '.join(display_class_label(label) for label in class_order)}")
log(f"Raw samples: {raw_sample_count}")
if removed_zero_sample_indices:
    log(f"Removed all-zero rows: {removed_zero_sample_indices}")


## show individual raw Purified Water / Tap Water / Dirty Water measurements separately
raw_sample_indices = select_raw_sample_indices(
    raw_by_class,
    class_order,
)

individual_raw_path = plot_individual_raw_signal_subplots(
    raw_by_class,
    class_order,
    output_dir / "01_individual_raw_measurements.png",
    sample_indices_by_class=raw_sample_indices,
)
log_saved(individual_raw_path)


## overlap raw Purified Water / Tap Water / Dirty Water signals
raw_signal_path = plot_raw_signals(
    raw_by_class,
    class_order,
    output_dir / "02_raw_signal_overlay.png",
    sample_indices_by_class=raw_sample_indices,
)
log_saved(raw_signal_path)


## show abnormal spikes and the same signals after spike removal
spike_examples = find_abnormal_spike_examples(
    raw_by_class,
    class_order,
    n_examples=3,
)
class_rank = {label: index for index, label in enumerate(class_order)}
spike_examples = sorted(
    spike_examples,
    key=lambda spike_example: class_rank.get(str(spike_example["label"]), len(class_order)),
)
spike_removal_path = plot_spike_removal_examples(
    spike_examples,
    output_dir / "03_spike_removal_example.png",
)
log_saved(spike_removal_path)
for spike_example in spike_examples:
    log(
        "Spike example: "
        f"{display_class_label(str(spike_example['label']))} sample {spike_example['sample_idx']}, "
        f"measurement point {spike_example['point_idx']}, "
        f"raw {spike_example['raw_value']:.1f} -> "
        f"after removal {spike_example['despiked_value']:.1f}"
    )


## clean/process the signals and show clearer class differences
log("Processing signals: slicing, despiking, smoothing, downsampling, and derivatives...")
signal_data = build_demo_signal_data_from_grouped(
    data_path=data_path,
    label_col=resolved_label_col,
    class_order=class_order,
    raw_by_class=raw_by_class,
    removed_zero_sample_indices=removed_zero_sample_indices,
)

log(f"Processed samples: {len(signal_data.y_all)}")
log(f"Processed tensor shape: {signal_data.x_all.shape}")

processed_slicing_path = plot_processed_class_average_before_after_slicing(
    signal_data.full_smoothed_no_sqrt_by_class,
    signal_data.full_smoothed_by_class,
    signal_data.smoothed_by_class,
    signal_data.class_order,
    output_dir / "04_processed_class_average_before_after_slicing.png",
)
log_saved(processed_slicing_path)


## train classifiers and reduce features to 2D/3D maps for display
feature_table_path = None
stale_feature_table_path = output_dir / "feature_examples.csv"
if stale_feature_table_path.exists():
    stale_feature_table_path.unlink()

log("Extracting simple manually designed features...")
simple_feature_data = extract_simple_demo_features(
    signal_data.x_all,
    signal_data.y_all,
    channel_names=signal_data.selected_value_types,
)

log("Extracting complex manually designed features...")
complex_feature_data = extract_complex_demo_features(
    signal_data.x_all,
    signal_data.y_all,
    channel_names=signal_data.selected_value_types,
)

processed_signal_features = signal_data.x_all.reshape(len(signal_data.x_all), -1)

log("Training Simple Feature classifier...")
simple_feature_classification = train_demo_classifier(
    simple_feature_data.x_features,
    signal_data.y_all,
    class_order=signal_data.class_order,
    random_seed=random_seed,
    test_size=test_size,
    unknown_test_position=unknown_test_position,
    method_name="Simple Feature",
    input_description="compact manually designed signal features",
)

log("Training Complex Feature classifier...")
complex_feature_classification = train_demo_classifier(
    complex_feature_data.x_features,
    signal_data.y_all,
    class_order=signal_data.class_order,
    random_seed=random_seed,
    test_size=test_size,
    unknown_test_position=unknown_test_position,
    method_name="Complex Feature",
    input_description="full manually designed signal features",
)

log("Training PCA-space classifier...")
pca_classification = train_pca_demo_classifier(
    processed_signal_features,
    signal_data.y_all,
    class_order=signal_data.class_order,
    random_seed=random_seed,
    test_size=test_size,
    unknown_test_position=unknown_test_position,
)

log("Training CNN demonstration classifier...")
cnn_classification = train_cnn_demo_classifier(
    signal_data.x_all,
    signal_data.y_all,
    class_order=signal_data.class_order,
    random_seed=random_seed,
    test_size=test_size,
    unknown_test_position=unknown_test_position,
    model_name=cnn_model_name,
    epochs=cnn_epochs,
    batch_size=cnn_batch_size,
    lr=cnn_lr,
    weight_decay=cnn_weight_decay,
    label_smoothing=cnn_label_smoothing,
    patience=cnn_patience,
    n_splits=cnn_n_splits,
    verbose=cnn_verbose,
)

classification = cnn_classification

feature_map_path = plot_cnn_feature_learning_comparison(
    classification,
    output_dir / "05_cnn_feature_learning_comparison.png",
    show_unknown=False,
)
log_saved(feature_map_path)
feature_map_3d_path = plot_cnn_feature_learning_comparison_3d(
    classification,
    output_dir / "06_cnn_feature_learning_comparison_3d.png",
    show_unknown=False,
)
log_saved(feature_map_3d_path)
feature_map_3d_interactive_path = plot_cnn_feature_learning_comparison_3d_interactive_html(
    classification,
    output_dir / "06_cnn_feature_learning_comparison_3d_interactive.html",
    show_unknown=False,
)
log_saved(feature_map_3d_interactive_path)
classifier_comparison_path = plot_classifier_result_comparison(
    pca_classification,
    simple_feature_classification,
    complex_feature_classification,
    cnn_classification,
    output_dir / "07_classifier_result_comparison.png",
)
log_saved(classifier_comparison_path)


## classify an unknown example
unknown_prediction_path = plot_unknown_prediction(
    signal_data.processed_by_class,
    classification,
    output_dir / "08_unknown_prediction.png",
)
log_saved(unknown_prediction_path)
unknown_game_path = plot_unknown_classification_game_html(
    signal_data.raw_by_class,
    signal_data.processed_by_class,
    classification,
    output_dir / "08_unknown_classification_game.html",
    simple_feature_result=simple_feature_classification,
    complex_feature_result=complex_feature_classification,
    pca_result=pca_classification,
)
log_saved(unknown_game_path)


## print demo summary
figure_paths = [
    individual_raw_path,
    raw_signal_path,
    spike_removal_path,
    processed_slicing_path,
    feature_map_path,
    feature_map_3d_path,
    feature_map_3d_interactive_path,
    classifier_comparison_path,
    unknown_prediction_path,
    unknown_game_path,
]

log("\nDemo summary")
log("------------")
log(f"Classifier: {classification.method_name}")
log(f"Classifier input: {classification.input_description}")
log(f"Holdout accuracy: {classification.test_accuracy:.3f}")
log(f"PCA Feature holdout accuracy: {pca_classification.test_accuracy:.3f}")
log(f"Simple Feature holdout accuracy: {simple_feature_classification.test_accuracy:.3f}")
log(f"Complex Feature holdout accuracy: {complex_feature_classification.test_accuracy:.3f}")
log(f"CNN Feature holdout accuracy: {cnn_classification.test_accuracy:.3f}")
log(f"Unknown example true label: {display_class_label(classification.unknown_true_label)}")
log(f"Unknown example prediction: {display_class_label(classification.unknown_pred_label)}")
log("Unknown example confidence:")
for label, probability in zip(classification.class_order, classification.unknown_prob):
    log(f"  {display_class_label(label)}: {probability:.3f}")

log("\nSaved outputs:")
for path in figure_paths:
    log(f"  {path}")
if feature_table_path is not None:
    log(f"  {feature_table_path}")
