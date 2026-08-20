from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from IRS_Insecticide_Residual.Raw_Data_Implementation.Embedded_Implementation.Shared_Functions.config import (
    EmbeddedPipelineConfig,
)

METHOD_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = str(METHOD_DIR / "Results")


@dataclass(frozen=True, kw_only=True)
class PytorchToKerasConfig(EmbeddedPipelineConfig):
    """Configuration for final PyTorch training followed by Keras transfer."""

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
    device: Optional[str] = None
    num_workers: int = 0
    tensorboard_log_dir: Optional[str] = None
    tensorboard_write_every_n: int = 10
    parity_warning_threshold: float = 1e-4
    tflite_variants: tuple[str, ...] = ()
