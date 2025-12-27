from __future__ import annotations
import numpy as np

def nonconvex_f(x: np.ndarray) -> np.ndarray:
    """f(x) = (x^2 + x(1+sin x) + 2) * exp(-|x|/3). Vectorized."""
    x = np.asarray(x)
    return (x**2 + x * (1.0 + np.sin(x)) + 2.0) * np.exp(-np.abs(x) / 3.0)
