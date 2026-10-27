# Ordinal Method

This folder implements a direct ordinal concentration model on the repeated-measurement
leave-one-sample-out folds.

The model has two heads:

- a liquid classification head trained with cross-entropy;
- a concentration ordinal head trained with ordered thresholds, so adjacent
  concentrations are treated as closer than distant ones.

The final joint prediction is reconstructed as:

```text
predicted_liquid__predicted_concentration
```

The example script uses exploratory test-fold early stopping for preliminary
upper-bound experiments. Do not treat these results as an unbiased final test
estimate.
