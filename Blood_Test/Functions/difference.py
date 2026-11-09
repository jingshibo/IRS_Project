from __future__ import annotations

from typing import Dict

import numpy as np

from .data_loader import AMP_IDX, IM_IDX, PHASE_IDX, RE_IDX


PBS_REFERENCE_MODES = {"own_set", "shared_baseline"}


def _get_pbs_reference(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    pbs_sample_type: str,
    pbs_concentration: str,
    reference_mode: str,
) -> np.ndarray:
    """Return own-set means or the same PBS-2/PBS-3 baseline for every set."""
    if reference_mode not in PBS_REFERENCE_MODES:
        raise ValueError(
            f"reference_mode must be one of {sorted(PBS_REFERENCE_MODES)}, "
            f"got {reference_mode!r}"
        )
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
    if pbs_block.ndim != 5 or pbs_block.shape[-1] < 2:
        raise ValueError(
            "Expected PBS block with shape "
            f"[set, sample, repetition, frequency, channel>=2], got {pbs_block.shape}"
        )

    if reference_mode == "own_set":
        # Preserve the original own-set calculation.
        return np.nanmean(pbs_block, axis=(1, 2), dtype=np.float32)
    if pbs_block.shape[0] < 3:
        raise ValueError(
            "reference_mode='shared_baseline' requires PBS Sets 2 and 3"
        )

    # Average repetitions within PBS Sets 2 and 3 first, then give their means
    # equal weight. PBS Set 1 never contributes, including for target Sets 2/3.
    baseline_set_means = np.nanmean(pbs_block[1:3], axis=(1, 2), dtype=np.float32)
    if not np.isfinite(baseline_set_means[..., :2]).all():
        raise ValueError(
            "shared_baseline requires valid real/imaginary PBS means in both "
            "Sets 2 and 3 at every frequency"
        )
    shared_reference = baseline_set_means.mean(axis=0, dtype=np.float32)

    # Derive amplitude and phase from the shared complex baseline rather than
    # averaging the source amplitudes or wrapped phase angles.
    if shared_reference.shape[-1] > PHASE_IDX:
        complex_reference = (
            shared_reference[..., RE_IDX] + 1j * shared_reference[..., IM_IDX]
        )
        shared_reference[..., AMP_IDX] = np.abs(complex_reference)
        shared_reference[..., PHASE_IDX] = np.angle(complex_reference)
    return np.broadcast_to(
        shared_reference,
        (pbs_block.shape[0], *shared_reference.shape),
    ).copy()


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
    ``reference_mode='shared_baseline'``, EVERY set uses the same pointwise
    (mean_PBS_Set_2 + mean_PBS_Set_3) / 2. Each set mean averages its PBS
    repetitions first; amplitude and phase come from the shared complex mean.
    PBS Set 1 is excluded from this shared reference.

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

    ``own_set`` uses each set's own mean PBS. ``shared_baseline`` uses the same
    (mean_PBS_Set_2 + mean_PBS_Set_3) / 2 for ALL sets, excluding PBS Set 1.

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

