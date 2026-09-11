# School Visit Insecticide Residual Demo

This folder contains a simplified, visual demonstration built from the
`IRS_Insecticide_Residual` data pipeline.

The demonstration story is:

```text
load insecticide residual data
-> show multiple individual raw Purified Water / Tap Water / Dirty Water measurements
-> overlap the same raw samples to show small class differences and noise
-> show abnormal spike examples and the same signals after spike removal
-> clean/process the signals
-> compare processed class averages before and after slicing
-> show PCA feature maps before and after CNN feature learning
-> compare PCA, simple-feature, complex-feature, and CNN-feature classification results
-> classify a student-selected unknown example
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

The script trains four demonstration classifiers: PCA Feature, Simple Feature,
Complex Feature, and CNN Feature. The CNN from `Raw_Data_Implementation` is
used for the learned feature maps and the final static unknown-sample
prediction.

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
matrices for the four feature views: PCA Feature, Simple Feature, Complex
Feature, and CNN Feature.

For the school-visit display, the original dataset labels are renamed:
`LOW` is shown as `Purified Water`, `TARGET` is shown as `Tap Water`, and
`HIGH` is shown as `Dirty Water`. This is only a display change; the underlying
data loading and model training still use the original labels.

The interactive unknown-sample game embeds a balanced pool of 18 mystery
holdout samples. The mystery set is chosen for teaching value: it prefers
examples where feature views disagree, where a weaker transformation makes a
wrong prediction, or where the 3D point sits near the wrong class group.
Students choose an anonymous item from the mystery sample list, then guess
Purified Water, Tap Water, or Dirty Water from the raw signal. `Process Sample`
shows the cleaned signal and a known-pattern comparison, using class average
curves and usual-range bands. Students can then guess again from the cleaner
evidence.
`Transform Sample` lets students choose
between four views of the same sample: `PCA Feature`, `Simple Feature`,
`Complex Feature`, and `CNN Feature`. Each method button gives one intuitive
sentence and one technical sentence. The map-based views use a 3D similarity
map so students can compare whether different transformations make the classes
cluster more clearly. `Simple Feature` uses compact whole-curve summary
measurements. `Complex Feature` uses the full manual feature extractor from
`Feature_Implementation`: whole-curve summaries, peak-and-dip shape
measurements, area measurements, and derivative-change measurements. A
selected set of 150 known map points is clickable; when students click one of
these points, the corresponding cleaned signal curve is shown beside the map.
The mystery sample marker is clickable too. After that, students can choose a
method, ask the classifier, switch to another method, and ask again. The
confidence chart only shows methods already tested in that round; its legend
shows the class predicted by each tested method. Students can build the
comparison step by step before revealing the true label. After the true label is
revealed, the transform method buttons remain available so students can keep
inspecting why different feature views look easier or harder to separate. They
can also continue pressing `Ask Classifier` to add remaining untested methods to
the confidence chart until all four methods have been compared. The `Choose
Another Sample` button is available throughout the activity and clears
the current sample display without changing completed scores. `Reset Score`
clears the score counters. The page keeps a simple student-versus-classifier
score across rounds.
