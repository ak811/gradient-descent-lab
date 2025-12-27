from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

def gd_backtracking_armijo(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    alpha0: float,
    c1: float,
    beta: float,
    niter: int,
    max_backtracks: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x0, float).copy()
    d = x.size
    X = np.zeros((d, niter), float)
    fvals = np.zeros(niter, float)
    alphas = np.zeros(niter, float)

    for k in range(niter):
        X[:, k] = x
        fx = float(f(x))
        gx = grad(x)
        ddir = -gx
        alpha = alpha0
        for _ in range(max_backtracks):
            if float(f(x + alpha * ddir)) <= fx + c1 * alpha * float(gx @ ddir):
                break
            alpha *= beta
        alphas[k] = alpha
        fvals[k] = fx
        x = x + alpha * ddir
    return X, fvals, alphas
