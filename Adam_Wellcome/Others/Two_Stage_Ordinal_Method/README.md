# Two-Stage Ordinal Method

This folder combines the two-stage hierarchy with ordinal concentration modeling.

The split is reused from `Repeated_Measurements`: each fold holds out one
independent sample from every `(liquid, concentration)` block, while all five
repeated measurements from that sample stay together.

The method trains:

- stage 1: one liquid classifier;
- stage 2: one ordinal concentration model for each liquid, initialized from
  the pretrained liquid classifier feature extractor and projection layer.

At inference time, the predicted liquid selects the matching ordinal
concentration model, and the final joint prediction is reconstructed as:

```text
predicted_liquid__predicted_concentration
```

The example script uses exploratory test-fold early stopping for preliminary
upper-bound experiments. Do not treat these results as an unbiased final test
estimate.

`use_liquid_pretraining=True` enables pretrained initialization for stage 2.
`freeze_pretrained_features=False` means the copied feature layers are fine-tuned
instead of frozen.

The reported concentration accuracy is conditional on the liquid route being
correct. This avoids counting predictions such as `ECOLI__10-5` as concentration
correct for a true `Abau__10-5` sample.
