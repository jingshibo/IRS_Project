from __future__ import annotations

from typing import Dict

import numpy as np

from .data_loader import AMP_IDX, IM_IDX, PHASE_IDX, RE_IDX


PBS_REFERENCE_MODES = {"own_set", "other_sets_mean"}


def _get_pbs_reference(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    pbs_sample_type: str,
    pbs_concentration: str,
    reference_mode: str,
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
    set_means = np.nanmean(pbs_block, axis=(1, 2), dtype=np.float32)
    if reference_mode == "own_set":
        return set_means
    if reference_mode != "other_sets_mean":
        raise ValueError(
            f"reference_mode must be one of {sorted(PBS_REFERENCE_MODES)}, "
            f"got {reference_mode!r}"
        )
    if set_means.shape[0] < 2:
        raise ValueError(
            "reference_mode='other_sets_mean' requires at least two PBS sets"
        )

    references = np.empty_like(set_means)
    for target_set in range(set_means.shape[0]):
        other_sets = np.delete(set_means, target_set, axis=0)
        references[target_set] = np.nanmean(
            other_sets,
            axis=0,
            dtype=np.float32,
        )

    # Average cross-set references in the complex plane, then derive amplitude
    # and phase from the averaged real and imaginary components.
    if references.shape[-1] > PHASE_IDX:
        complex_reference = (
            references[..., RE_IDX] + 1j * references[..., IM_IDX]
        )
        references[..., AMP_IDX] = np.abs(complex_reference)
        references[..., PHASE_IDX] = np.angle(complex_reference)
    return references


def append_pbs_plain_difference(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    pbs_sample_type: str = "PBS",
    pbs_concentration: str = "reference",
    reference_mode: str = "own_set",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Append pointwise differences from the selected mean PBS reference trace.

    ``reference_mode='own_set'`` uses each set's own mean PBS. With
    ``reference_mode='other_sets_mean'``, each target set uses the pointwise
    mean PBS from all other sets.

    Input channels::

        [real, imaginary, amplitude, phase]

    Output channels::

        [real, imaginary, amplitude, phase,
         delta_real, delta_imaginary, delta_amplitude, delta_phase]
    """
    pbs_reference = _get_pbs_reference(
        data,
        pbs_sample_type=pbs_sample_type,
        pbs_concentration=pbs_concentration,
        reference_mode=reference_mode,
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
    reference_mode: str = "own_set",
    eps: float = 1e-12,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Append complex features using the selected PBS-reference mode.

    Five channels are appended to the existing channels:

    - ``complex_ratio_phase``: phase of ``signal / mean_PBS``.
    - ``complex_subtraction_magnitude``: magnitude of ``signal - mean_PBS``.
    - ``phase_conjugate_difference``: phase of ``signal * conj(mean_PBS)``.
    - cosine and sine of the conjugate-product phase difference.
    """
    pbs_reference = _get_pbs_reference(
        data,
        pbs_sample_type=pbs_sample_type,
        pbs_concentration=pbs_concentration,
        reference_mode=reference_mode,
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

            complex_ratio_phase = np.angle(complex_values / (expanded_base + eps)).astype(np.float32, copy=False)
            complex_subtraction_magnitude = np.abs(complex_values - expanded_base).astype(np.float32, copy=False)
            # Phase difference using complex conjugate
            phase_conjugate_difference = np.angle(complex_values * np.conj(expanded_base)).astype(np.float32, copy=False)
            cos_phase_difference = np.cos(phase_conjugate_difference).astype(np.float32, copy=False)
            sin_phase_difference = np.sin(phase_conjugate_difference).astype(np.float32, copy=False)

            appended = np.stack(
                (complex_ratio_phase, complex_subtraction_magnitude, phase_conjugate_difference, cos_phase_difference, sin_phase_difference),
                axis=-1,
            ).astype(np.float32, copy=False)
            ratio_data[sample_type][concentration] = np.concatenate(
                (block, appended),
                axis=-1,
            )

    return ratio_data

