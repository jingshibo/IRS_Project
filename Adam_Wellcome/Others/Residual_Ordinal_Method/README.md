# Residual Ordinal Method

This folder implements a single-stage ordinal model with liquid-reference residual preprocessing.

For each cross-validation fold:

- compute one mean reference signal per liquid using the training measurements in that fold only
- build training residuals by subtracting the liquid-specific training mean using the known training liquid labels
- fit an auxiliary liquid classifier on the raw training signals only
- predict test liquid identities with that auxiliary classifier
- build test residuals by subtracting the liquid-specific training mean chosen by the predicted test liquid, not the true test liquid
- train the ordinal liquid-plus-concentration model on either:
  - `residual_only`: residual channels only
  - `concat`: concatenate raw channels and residual channels

The default configuration uses `residual_mode="concat"` because the model still predicts liquid and concentration jointly. Using residuals alone can suppress the liquid identity signal that the liquid head needs.

Important leakage rule:

- liquid reference means are computed from `x_train` only inside each fold
- test fold measurements never contribute to the liquid reference
- true test liquid labels are not used for residual preprocessing

The dataset split remains the same 3-fold leave-one-independent-sample-out split used elsewhere in the project.
