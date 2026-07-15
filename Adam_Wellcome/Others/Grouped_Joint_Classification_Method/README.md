# Grouped Concentration Joint Classification Method

This folder implements direct classification over grouped joint labels.

The concentration grouping is:

- `high`: `10^-1` to `10^-3`
- `medium`: `10^-4` to `10^-7`
- `low`: `10^-8` to `10^-10`

Unlike the two-head grouped classification and ordinal methods, this model has
one output head over the full grouped joint label space:

```text
5 liquids x 3 concentration groups = 15 classes
```

Examples of direct target labels are:

```text
Abau__high
Abau__medium
Abau__low
ECOLI__high
```

The dataset split remains the same 3-fold leave-one-independent-sample-out
split, and sample-level evaluation averages the repeated-measurement class
probabilities before choosing the final joint label.
