# Ordinal 50-Class Method

This folder implements direct ordinal classification over the 50 joint labels `liquid__concentration`.

Unlike `Ordinal_Method`, this model does not have separate liquid and concentration heads. It trains one ordinal head over a fixed global order of the 50 classes.

The global class order is:

- sort by liquid name
- within each liquid, sort by concentration order

This gives an ordinal structure to the full 50-class task, but the ordering is only scientifically natural within each liquid. Across different liquids, the ordinal relation is mainly a modeling convenience rather than a true physical order.

The dataset split remains the same 3-fold leave-one-independent-sample-out split, and sample-level evaluation averages the 5 repeated measurements by averaging the ordinal logits before decoding.

The summary still reports:

- joint 50-class accuracy
- derived liquid accuracy from the decoded joint labels
- derived concentration accuracy from the decoded joint labels

