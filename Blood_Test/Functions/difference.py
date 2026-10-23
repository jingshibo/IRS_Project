from __future__ import annotations

from typing import Dict

import numpy as np

from .data_loader import IM_IDX, RE_IDX


def _get_set_specific_pbs_reference(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    pbs_sample_type: str,
    pbs_concentration: str,
) -> np.ndarray:
    if (
        pbs_sample_type not in data
        or pbs_concentration not in data[pbs_sample_type]
    ):
        raise KeyError(
            f"PBS reference block not found: {pbs_sample_type} {pbs_concentration}"
        )

    pbs_block = np.asarray(
        data[pbs_sample_type][pbs_concentration],
        dtype=np.float32,
    )
    if pbs_block.ndim != 5:
        raise ValueError(
            "Expected PBS block with shape "
            f"[set, sample, repetition, frequency, channel], got {pbs_block.shape}"
        )

    # Preserve the set axis and average PBS samples/repetitions within each set.
    return np.nanmean(pbs_block, axis=(1, 2), dtype=np.float32)


def append_pbs_reference_difference(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    pbs_sample_type: str = "PBS",
    pbs_concentration: str = "reference",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Append pointwise differences from each set's mean PBS reference trace.

    Input channels::

        [real, imaginary, amplitude, phase]

    Output channels::

        [real, imaginary, amplitude, phase,
         delta_real, delta_imaginary, delta_amplitude, delta_phase]
    """
    pbs_reference = _get_set_specific_pbs_reference(
        data,
        pbs_sample_type=pbs_sample_type,
        pbs_concentration=pbs_concentration,
    )

    referenced_data: Dict[str, Dict[str, np.ndarray]] = {}
    for sample_type, concentration_map in data.items():
        referenced_data[sample_type] = {}
        for concentration, block in concentration_map.items():
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 5 or block.shape[-1] != 4:
                raise ValueError(
                    "Expected block with shape "
                    "[set, sample, repetition, frequency, 4], "
                    f"got {block.shape}"
                )
            if block.shape[0] != pbs_reference.shape[0]:
                raise ValueError(
                    f"Set count mismatch for {sample_type} {concentration}: "
                    f"data has {block.shape[0]}, PBS has {pbs_reference.shape[0]}"
                )

            deltas = block - pbs_reference[:, np.newaxis, np.newaxis, :, :]
            referenced_data[sample_type][concentration] = np.concatenate(
                (block, deltas.astype(np.float32, copy=False)),
                axis=-1,
            )

    return referenced_data


def append_pbs_complex_difference(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    pbs_sample_type: str = "PBS",
    pbs_concentration: str = "reference",
    eps: float = 1e-12,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Append set-specific complex PBS-reference features.

    Two channels are appended to the existing channels:

    - ``complex_ratio_phase``: phase of ``signal / mean_PBS``.
    - ``complex_subtraction_magnitude``: magnitude of ``signal - mean_PBS``.
    """
    pbs_reference = _get_set_specific_pbs_reference(
        data,
        pbs_sample_type=pbs_sample_type,
        pbs_concentration=pbs_concentration,
    )
    if pbs_reference.shape[-1] < 2:
        raise ValueError(
            f"Expected at least real/imaginary PBS channels, got {pbs_reference.shape}"
        )

    real_base = pbs_reference[..., RE_IDX]
    imaginary_base = pbs_reference[..., IM_IDX]
    complex_base = real_base + 1j * imaginary_base

    ratio_data: Dict[str, Dict[str, np.ndarray]] = {}
    for sample_type, concentration_map in data.items():
        ratio_data[sample_type] = {}
        for concentration, block in concentration_map.items():
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 5 or block.shape[-1] < 2:
                raise ValueError(
                    "Expected block with shape "
                    f"[set, sample, repetition, frequency, channel>=2], got {block.shape}"
                )
            if block.shape[0] != complex_base.shape[0]:
                raise ValueError(
                    f"Set count mismatch for {sample_type} {concentration}: "
                    f"data has {block.shape[0]}, PBS has {complex_base.shape[0]}"
                )

            real_values = block[..., RE_IDX]
            imaginary_values = block[..., IM_IDX]
            complex_values = real_values + 1j * imaginary_values
            expanded_base = complex_base[:, np.newaxis, np.newaxis, :]

            complex_ratio_phase = np.angle(
                complex_values / (expanded_base + eps)
            ).astype(np.float32, copy=False)
            complex_subtraction_magnitude = np.abs(
                complex_values - expanded_base
            ).astype(np.float32, copy=False)

            appended = np.stack(
                (complex_ratio_phase, complex_subtraction_magnitude),
                axis=-1,
            ).astype(np.float32, copy=False)
            ratio_data[sample_type][concentration] = np.concatenate(
                (block, appended),
                axis=-1,
            )

    return ratio_data

