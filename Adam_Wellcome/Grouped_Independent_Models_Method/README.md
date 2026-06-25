# Grouped Independent Models Method

This method trains two separate CNN models instead of a shared two-head model:

- a liquid model for liquid identity classification
- a concentration model for grouped concentration prediction

The final joint prediction is formed by combining the two independent predictions:

```text
predicted_liquid + "__" + predicted_concentration_group
```

So a joint prediction is correct only when both the liquid and concentration group are correct.

## Files

- `model.py`: defines `IndependentAdamWellcomeCNN1D`, the CNN used by both independent models.
- `training.py`: trains the liquid and concentration models separately and reports combined results.
- `evaluation.py`: summarizes fold results and plots grouped recall matrices.
- `test_group_independent_models.py`: runnable experiment script.

## Data Setup

The script builds the dataset with:

```python
label_mode="joint"
```

The joint labels are then split into:

```text
liquid label
concentration group label
```

The default grouped concentration labels are:

```text
high   = 10^-1 to 10^-3
medium = 10^-4 to 10^-7
low    = 10^-8 to 10^-10
```

These ranges can be changed in `test_group_independent_models.py`:

```python
concentration_group_ranges = (
    ("high", 1, 3),
    ("medium", 4, 7),
    ("low", 8, 10),
)
```

## Model Training

The two models are trained one by one.

First, the liquid model is trained and its best checkpoint is selected using:

```python
liquid_early_stopping_metric="sample_liquid_acc"
```

Then, the concentration model is trained and its best checkpoint is selected using:

```python
concentration_early_stopping_metric="sample_concentration_acc"
```

After both best models are restored, the final liquid, concentration, and joint accuracies are calculated.

## Liquid Model

The liquid model is a regular multiclass classifier.

For five liquids, it outputs five logits and is trained with:

```python
nn.CrossEntropyLoss(label_smoothing=liquid_label_smoothing)
```

Sample-level liquid prediction averages softmax probabilities across repeated measurements, then takes `argmax`.

## Concentration Model

The concentration model uses a strict ordinal formulation.

For three ordered concentration groups:

```text
high -> medium -> low
```

the model learns:

```text
one concentration score
ordered cutpoints
```

The ordinal logits are calculated as:

```python
ordinal_logits = score - ordered_cutpoints
```

For three concentration groups, the model outputs two threshold logits. Targets are cumulative:

```text
high   -> [0, 0]
medium -> [1, 0]
low    -> [1, 1]
```

Training uses:

```python
nn.BCEWithLogitsLoss()
```

Prediction decodes the class by counting how many threshold logits are positive:

```python
predicted_group_index = sum(ordinal_logits > 0)
```

Sample-level concentration prediction averages ordinal logits across repeated measurements, then decodes the averaged logits.

## Final Joint Accuracy

For each held-out sample:

1. Average liquid probabilities across repeats.
2. Predict the liquid with `argmax`.
3. Average concentration ordinal logits across repeats.
4. Decode the concentration group.
5. Combine them into a joint label.

Example:

```text
ECOLI + medium -> ECOLI__medium
```

Sample joint accuracy is:

```text
number of samples where both liquid and concentration group are correct
/
number of held-out samples
```

## Reported Metrics

The trainer reports:

```text
measurement_joint_acc
sample_joint_acc
measurement_liquid_acc
sample_liquid_acc
measurement_concentration_acc
sample_concentration_acc
measurement_confusion_matrix
sample_confusion_matrix
```

The evaluation script averages these metrics across folds and can plot grouped recall matrices.

## Important Caveats

The current training code uses the held-out test fold for early stopping. This is useful for experimentation, but it can make the reported test accuracy optimistic. For a cleaner estimate, use an inner validation split from the training samples, or use fixed training epochs and reserve the test fold only for final reporting.

The label mappings are currently built from the training fold. This assumes every training fold contains all liquid classes and all concentration groups. If a fold is missing a class, the fold may fail or produce inconsistent confusion matrix dimensions.
