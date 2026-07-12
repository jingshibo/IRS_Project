from typing import List, Optional, Sequence, Tuple

import numpy as np

from IRS_Insecticide_Residual.Feature_Implementation.Functions.Global_Features import calculate_global_features


def _validate_signal_dataset(x: np.ndarray) -> np.ndarray:
    """Validate that x has shape [N, C, L] and signal length >= 2."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"x must have shape [N, C, L], got {x.shape}")
    if x.shape[2] < 2:
        raise ValueError(f"Signal length must be >= 2, got {x.shape[2]}")
    return x


def _channel_feature_block(channel_values: np.ndarray, channel_name: str) -> Tuple[np.ndarray, List[str]]:
    """Extract one channel's global feature block of shape [N, F]."""
    channel_values = np.asarray(channel_values, dtype=np.float32)
    if channel_values.ndim != 2:
        raise ValueError(f"channel_values must have shape [N, L], got {channel_values.shape}")

    return calculate_global_features(channel_values, channel_name=channel_name)


def extract_feature_matrix(
    x: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract handcrafted tabular features from a signal tensor of shape [N, C, L]."""
    x = _validate_signal_dataset(x)
    n_channels = x.shape[1]

    if channel_names is None:
        resolved_channel_names = [f"channel_{idx}" for idx in range(n_channels)]
    else:
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names length must match channel count {n_channels}, got {len(channel_names)}"
            )
        resolved_channel_names = [str(name) for name in channel_names]

    feature_blocks = []
    feature_names: List[str] = []
    for channel_idx, channel_name in enumerate(resolved_channel_names):
        channel_block, channel_feature_names = _channel_feature_block(x[:, channel_idx, :], channel_name=channel_name)
        feature_blocks.append(channel_block)
        feature_names.extend(channel_feature_names)

    feature_matrix = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    return feature_matrix, feature_names
