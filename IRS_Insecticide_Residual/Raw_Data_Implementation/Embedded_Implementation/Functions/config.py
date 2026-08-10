from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True, kw_only=True)
class EmbeddedPipelineConfig:
    """Shared data/model configuration used by all embedded generation paths."""

    class_order: Sequence[str] = ("LOW", "TARGET", "HIGH")
    model_name: str = "shared_backbone_2ch"
    selected_value_types: Sequence[str] = ("original", "first_diff_filtered", "second_diff_filtered")
    signal_segments: Sequence[tuple[int, int]] = ((0, 1000), (1800, 3500))
    spike_radius: int = 3
    spike_transform: str = "sqrt"
    spike_method: str = "fast"
    spike_n_sigmas: float = 3.0
    spike_k: float = 4.0
    spike_min_threshold: float = 1000.0
    savgol_window_length: int = 31
    savgol_polyorder: int = 3
    savgol_deriv: int = 0
    savgol_mode: str = "mirror"
    downsample_step: int = 5
    downsample_offset: int = 0
    rolling_window_size: int = 31
    random_seed: int = 42
    test_size: float = 0.15
    representative_count: int = 128
    clip_max_value: Optional[float] = None
    match_pytorch_flatten: bool = True
