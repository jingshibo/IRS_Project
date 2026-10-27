# Regression Method

This folder implements a direct liquid-plus-concentration model for the Adam Wellcome repeated-measurement data.

The split is the same leave-one-independent-sample-out split used by `Repeated_Measurements` and `Ordinal_Method`. All five repeated measurements from one physical sample stay together in either training or testing.

The model has one shared 1D CNN feature extractor and two heads:

- liquid head: cross-entropy classification over the five liquids
- concentration head: scalar regression to the ordered concentration index

The concentration target is normalized from class index `0..N-1` to `0..1`. At evaluation time, the scalar output is converted back to concentration class by multiplying by `N-1`, rounding to the nearest class, and clipping to the valid range.

Sample-level evaluation averages the five repeated measurements before deciding the liquid and concentration:

- liquid: average predicted probabilities, then argmax
- concentration: average scalar regression output, then round to the nearest concentration class

Important parameters:

- `concentration_loss_weight`: weights the regression loss relative to the liquid classification loss
- `early_stopping_metric`: defaults to `sample_joint_acc`, matching the grouped sample-level goal
- `early_stopping_patience`: number of epochs to wait without improvement

