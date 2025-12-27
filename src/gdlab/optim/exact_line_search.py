from __future__ import annotations
import numpy as np
from typing import Tuple

def alpha_exact_ls(A: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    """Exact line search for f(x)=1/m||Ax-y||^2 along steepest descent direction."""
    m = A.shape[0]
    r = A @ x - y
    z = A @ (A.T @ r)
    num = 0.5 * m * float(r @ z)
    den = float(z @ z)
    return 0.0 if den <= 0 else num / den

def gd_exact_line_search_ls(A: np.ndarray, y: np.ndarray, x0: np.ndarray, niter: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x0, float).copy()
    d = x.size
    X = np.zeros((d, niter), float)
    fvals = np.zeros(niter, float)
    alphas = np.zeros(niter, float)
    m = A.shape[0]
    for k in range(niter):
        X[:, k] = x
        r = A @ x - y
        fvals[k] = float((r @ r) / m)
        a = alpha_exact_ls(A, y, x)
        alphas[k] = a
        g = (2.0 / m) * (A.T @ r)
        x = x - a * g
    return X, fvals, alphas
