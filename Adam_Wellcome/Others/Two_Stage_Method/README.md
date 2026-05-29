# Two-Stage Method

This folder implements hierarchical liquid-concentration classification for the
Adam Wellcome repeated-measurement data.

The split is reused from `Repeated_Measurements`: each fold holds out one
independent sample from every `(liquid, concentration)` block, while all five
repeated measurements from that sample stay together.

The model is trained in two stages:

- stage 1: one classifier predicts the liquid;
- stage 2: one concentration classifier is trained separately for each liquid.

At inference time, the predicted liquid selects the corresponding concentration
classifier. The final prediction is reconstructed as:

```text
predicted_liquid__predicted_concentration
```

The example script uses test-fold early stopping for preliminary upper-bound
experiments. Do not treat these numbers as an unbiased final test estimate.
