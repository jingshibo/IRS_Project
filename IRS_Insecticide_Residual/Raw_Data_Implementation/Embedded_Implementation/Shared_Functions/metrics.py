from __future__ import annotations

import numpy as np


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Convert logits to softmax probabilities using NumPy."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32, copy=False)


def compute_logit_parity(reference_logits: np.ndarray, exported_logits: np.ndarray) -> dict[str, float]:
    abs_diff = np.abs(reference_logits - exported_logits)
    return {
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "p95_abs_diff": float(np.percentile(abs_diff, 95)) if abs_diff.size else 0.0,
    }
