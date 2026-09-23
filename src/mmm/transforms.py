"""Media response transforms: geometric adstock and Hill saturation.

Both are written to work on NumPy arrays (simulation, plotting) and on PyTensor
tensors (inside the PyMC model), so the same functional form is used to
generate and to fit — the recovery test is only meaningful if that holds.
"""

from __future__ import annotations

import numpy as np


def geometric_adstock_np(x: np.ndarray, decay: float, max_lag: int) -> np.ndarray:
    """y_t = sum_{l=0}^{L} decay^l * x_{t-l}, normalised so weights sum to 1."""
    w = decay ** np.arange(max_lag + 1)
    w = w / w.sum()
    out = np.zeros_like(x, dtype=float)
    for l, wl in enumerate(w):
        out[l:] += wl * x[: len(x) - l] if l else wl * x
    return out


def hill_np(x: np.ndarray, half_sat: float, slope: float) -> np.ndarray:
    """Hill saturation in [0, 1): x^s / (K^s + x^s)."""
    xs = np.power(np.clip(x, 0, None), slope)
    return xs / (np.power(half_sat, slope) + xs)


def geometric_adstock_pt(x, decay, max_lag: int):
    """PyTensor version. x: (T,) tensor; decay: scalar RV."""
    import pytensor.tensor as pt
    lags = pt.arange(max_lag + 1)
    w = decay ** lags
    w = w / pt.sum(w)
    T = x.shape[0]
    out = pt.zeros_like(x)
    for l in range(max_lag + 1):
        shifted = pt.concatenate([pt.zeros(l), x[: T - l]]) if l else x
        out = out + w[l] * shifted
    return out


def hill_pt(x, half_sat, slope):
    import pytensor.tensor as pt
    xs = pt.power(pt.maximum(x, 0.0), slope)
    return xs / (pt.power(half_sat, slope) + xs)
