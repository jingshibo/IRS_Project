from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Functions.config import (
    EmbeddedPipelineConfig,
)

DEFAULT_OUTPUT_DIR = (
    "IRS_Insecticide_Residual/Raw_Data_Implementation/"
    "Embedded_Implementation/Retrain_Keras_Method/artifacts/final_model"
)


@dataclass(frozen=True, kw_only=True)
class FinalTrainingConfig(EmbeddedPipelineConfig):
    """Configuration for the final Keras deployment-training pipeline."""

    excel_path: str
    output_dir: str = DEFAULT_OUTPUT_DIR
    final_epochs: Optional[int] = None
    cv_folds_for_epoch_selection: int = 5
    run_cv_for_epoch_selection: bool = True
    max_cv_epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.3
    patience: int = 25
    use_lr_scheduler: bool = True
    scheduler_factor: float = 0.7
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    random_shift_max_points: int = 5
    random_shift_fill_mode: str = "wrap"
    optimizer_beta_1: float = 0.9
    optimizer_beta_2: float = 0.999
    optimizer_epsilon: float = 1e-8
    optimizer_amsgrad: bool = False
