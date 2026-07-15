# Two-Stage Residual Ordinal Method

This folder implements a clean two-stage version of the liquid-reference residual idea.

Stage 1:

- train a liquid classifier on the raw signals
- predict liquid identity at measurement level and sample level

Stage 2:

- train one ordinal concentration model per liquid
- for each liquid, compute a liquid-specific reference mean from that liquid's training measurements only
- subtract that mean before concentration modeling
- optionally use:
  - `residual_only`: only the residual channels
  - `concat`: raw channels concatenated with residual channels

This is cleaner than the single-stage residual ordinal method because the residual preprocessing is only used after liquid routing. That matches the intended use of liquid-specific baseline removal.

Important leakage rule:

- each liquid reference mean is computed from that liquid's training fold measurements only
- test measurements never contribute to the residual reference
- stage 2 uses the liquid predicted by stage 1 during evaluation, not the true test liquid

The default configuration uses `residual_mode="residual_only"` and keeps the same 3-fold leave-one-independent-sample-out split used elsewhere in the project.

Stage-specific optimization can be tuned independently:

- `liquid_batch_size`, `liquid_lr`, `liquid_weight_decay` for stage 1 liquid classification
- `concentration_batch_size`, `concentration_lr`, `concentration_weight_decay` for stage 2 concentration fine-tuning

This is useful because each liquid-specific stage-2 model is trained on much less data than stage 1.
