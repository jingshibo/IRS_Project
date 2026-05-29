# Repeated-Measurement Experiment

This folder implements sample-grouped splitting and sample-level evaluation for
the Adam Wellcome repeated-measurement data.

The independent experimental unit is:

```text
(liquid, concentration, sample)
```

Each independent sample has repeated measurements identified by `rep`. All
repeats from the same independent sample are kept in the same split.

The split strategy is leave-one-sample-out within each liquid-concentration
block. With three independent samples and five repeats per sample, each fold is:

```text
training: 50 liquid-concentration blocks x 2 samples x 5 repeats = 500 measurements
testing:  50 liquid-concentration blocks x 1 sample  x 5 repeats = 250 measurements
```

For `label_mode="liquid"`, the same grouped split is used, but the target label
has five liquid classes instead of fifty liquid-concentration classes.

Test accuracy is reported at both levels:

- measurement-level accuracy: each repeated measurement is counted separately;
- sample-level accuracy: the five repeated measurements of one sample are
  aggregated into one final prediction using probability averaging or majority
  vote.

The example trainer in [test.py](C:/Users/Shibo.Jing/PycharmProjects/IRS_Project/Adam_Wellcome/Repeated_Measurements/test.py:1)
uses exploratory early stopping on the held-out test fold with
`early_stopping_metric="sample_probability_test_acc"`. This is suitable for a
preliminary upper-bound experiment, but it should not be reported as an
unbiased final test estimate.
