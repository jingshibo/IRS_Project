from __future__ import annotations

from typing import Dict

import numpy as np

from .data_loader import IM_IDX, PHASE_IDX, RE_IDX


def append_pbs_reference_difference(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    pbs_liquid: str = "PBS",
    pbs_concentration: str = "10-0",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Append pointwise differences from the mean PBS reference trace.

    Input:
        data[liquid][concentration] -> [sample, rep, L, 4]
        channels = [re, im, amplitude, phase]

    Output:
        referenced_data[liquid][concentration] -> [sample, rep, L, 8]
        channels = [
            re, im, amplitude, phase,
            delta_re, delta_im, delta_amplitude, delta_phase
        ]
    """
    if pbs_liquid not in data or pbs_concentration not in data[pbs_liquid]:
        raise KeyError(
            f"PBS reference block not found: {pbs_liquid}_{pbs_concentration}"
        )

    pbs_block = data[pbs_liquid][pbs_concentration]
    pbs_reference = np.nanmean(pbs_block, axis=(0, 1), dtype=pbs_block.dtype)

    referenced_data: Dict[str, Dict[str, np.ndarray]] = {}
    for liquid, concentration_map in data.items():
        referenced_data[liquid] = {}
        for concentration, block in concentration_map.items():
            if block.shape[-1] != 4:
                raise ValueError(
                    f"Expected 4 input channels [re, im, amplitude, phase], "
                    f"got shape {block.shape}"
                )

            deltas = block - pbs_reference[np.newaxis, np.newaxis, :, :]
            referenced_data[liquid][concentration] = np.concatenate(
                (block, deltas.astype(block.dtype, copy=False)),
                axis=-1,
            )

    return referenced_data


def append_pbs_complex_difference(
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    pbs_liquid: str = "PBS",
    pbs_concentration: str = "10-0",
    eps: float = 1e-12,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Append PBS-reference complex-ratio features to each block.

    Input:
        data[liquid][concentration] -> [sample, rep, L, 4]
        channels = [re, im, amplitude, phase]

    Output:
        ratio_data[liquid][concentration] -> [sample, rep, L, C + 4]
        channels = [
            ...existing input channels...,
            x_phase, x_mag_sub, wrapped_delta_phase, x_phase_conj
        ]
    """
    if pbs_liquid not in data or pbs_concentration not in data[pbs_liquid]:
        raise KeyError(f"PBS reference block not found: {pbs_liquid}_{pbs_concentration}")

    pbs_block = np.asarray(data[pbs_liquid][pbs_concentration], dtype=np.float32)
    if pbs_block.shape[-1] < 2:
        raise ValueError(f"Expected at least re/im channels in PBS block, got shape {pbs_block.shape}")

    re_base = np.nanmean(pbs_block[..., RE_IDX], axis=(0, 1), dtype=np.float32)
    im_base = np.nanmean(pbs_block[..., IM_IDX], axis=(0, 1), dtype=np.float32)
    pbs_phase = np.nanmean(pbs_block[..., PHASE_IDX], axis=(0, 1), dtype=np.float32)
    z_base = re_base + 1j * im_base

    ratio_data: Dict[str, Dict[str, np.ndarray]] = {}
    for liquid, concentration_map in data.items():
        ratio_data[liquid] = {}
        for concentration, block in concentration_map.items():
            block = np.asarray(block, dtype=np.float32)
            if block.ndim != 4 or block.shape[-1] < 2:
                raise ValueError(
                    f"Expected block with shape [sample, rep, L, C>=2], got {block.shape}"
                )

            re_values = block[..., RE_IDX]
            im_values = block[..., IM_IDX]
            z = re_values + 1j * im_values
            z_corr = z / (z_base[np.newaxis, np.newaxis, :] + eps)
            z_sub = z - z_base[np.newaxis, np.newaxis, :]
            phase = block[..., PHASE_IDX]

            x_phase = np.angle(z_corr).astype(np.float32, copy=False)
            # x_phase_conj = np.angle(
            #     z * np.conj(z_base[np.newaxis, np.newaxis, :]) # the value is same as x_phase except for a small eps effect
            # ).astype(np.float32, copy=False)
            # x_mag = np.abs(z_corr).astype(np.float32, copy=False)
            # x_mag_db = np.log1p(x_mag).astype(np.float32, copy=False)
            # x_phase_sin = np.sin(x_phase).astype(np.float32, copy=False)
            # x_phase_cos = np.cos(x_phase).astype(np.float32, copy=False)
            x_mag_sub = np.abs(z_sub).astype(np.float32, copy=False)
            # x_phase_sub = np.angle(z_sub).astype(np.float32, copy=False)

            appended = np.stack(
                (x_phase, x_mag_sub),
                axis=-1,
            ).astype(np.float32, copy=False)

            ratio_data[liquid][concentration] = np.concatenate((block, appended), axis=-1)

    return ratio_data
