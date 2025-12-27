from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    alpha: float,
    niter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x0, float).copy()
    d = x.size
    X = np.zeros((d, niter), float)
    fvals = np.zeros(niter, float)
    for k in range(niter):
        X[:, k] = x
        fvals[k] = float(f(x))
        x = x - alpha * grad(x)
    return X, fvals
