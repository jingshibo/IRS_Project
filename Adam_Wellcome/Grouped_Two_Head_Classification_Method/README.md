# Grouped Concentration Classification Method

This folder implements plain classification with grouped concentration labels instead of the original 10 dilution levels.

Default concentration grouping:

- `high`: `10^-1` to `10^-3`
- `medium`: `10^-4` to `10^-7`
- `low`: `10^-8` to `10^-10`

The model predicts:

- liquid identity with a classification head
- grouped concentration with a plain cross-entropy classification head

So the effective joint label space becomes `5 liquids x 3 concentration groups = 15` grouped joint classes.

This is useful as a preliminary-data test: if grouped concentration accuracy improves substantially relative to exact dilution-level prediction, that supports the claim that the sensing system is concentration-sensitive even if exact 10-level classification is still difficult.

The dataset split remains the same 3-fold leave-one-independent-sample-out split, and sample-level evaluation still averages the 5 repeated measurements before decoding.
