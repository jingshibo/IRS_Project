# School Visit Insecticide Residual Demo

This folder contains a simplified, visual demonstration built from the
`IRS_Insecticide_Residual` data pipeline.

The demonstration story is:

```text
load insecticide residual data
-> show multiple individual raw LOW / TARGET / HIGH measurements
-> overlap the same raw samples to show small class differences and noise
-> show abnormal spike examples and the same signals after spike removal
-> clean/process the signals
-> compare processed class averages before and after slicing
-> show PCA feature maps before and after CNN feature learning
-> compare Original and CNN classification results
-> classify a randomly selected unknown example
```

The main script is:

```bash
python "IRS_Insecticide_Residual/School Visit/run_demo.py"
```

The script follows the same top-level style as the main project examples. It
can also be run from an interactive console because it does not depend on
`__file__` being defined.

Edit these variables near the top of `run_demo.py` when needed:

```python
excel_path = None
sheet_name = 0
label_col = None
output_dir = SCHOOL_VISIT_DIR / "outputs"
test_size = 0.20
unknown_test_position = None
random_seed = RANDOM_SEED
```

`excel_path = None` checks the usual lab data path and the
`IRS_SCHOOL_VISIT_DATA_PATH` environment variable.

The script trains both classifiers. The original manually designed feature
classifier is used for comparison in figure `07`; the CNN from
`Raw_Data_Implementation` is used for the feature maps and the final
unknown-sample prediction.

The CNN settings are also editable near the top of `run_demo.py`, for example
`cnn_epochs`, `cnn_batch_size`, `cnn_model_name`, and `cnn_n_splits`.

Outputs are written to `outputs/`:

- `01_individual_raw_measurements.png`
- `02_raw_signal_overlay.png`
- `03_spike_removal_example.png`
- `04_processed_class_average_before_after_slicing.png`
- `05_cnn_feature_learning_comparison.png`
- `06_cnn_feature_learning_comparison_3d.png`
- `06_cnn_feature_learning_comparison_3d_interactive.html`
- `07_classifier_result_comparison.png`
- `08_unknown_prediction.png`
- `08_unknown_classification_game.html`

Figure `07` reports row-normalized recall percentages in the confusion
matrices. The label `Original` refers to the manually designed feature
classifier. The label `CNN` refers to the learned signal-feature classifier.

The interactive unknown-sample game embeds a small balanced pool of unknown
holdout samples. Each time `Load Unknown Sample` is clicked, the page randomly
selects a candidate, displays its raw signal, then shows the processed signal,
2D and 3D CNN feature-map positions, class confidence, and finally the true
label.
