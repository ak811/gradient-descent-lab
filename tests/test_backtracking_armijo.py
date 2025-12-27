from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gdlab.objectives.quadratic2d import Quadratic2D
from gdlab.optim.backtracking import gd_backtracking_armijo

def test_backtracking_satisfies_armijo() -> None:
    quad = Quadratic2D(1.75)
    x0 = np.array([1.0, 4.0])
    X, fvals, alphas = gd_backtracking_armijo(quad.f, quad.grad, x0, alpha0=10.0, c1=1e-4, beta=0.5, niter=25)
    for k in range(24):
        x = X[:, k]
        gx = quad.grad(x)
        d = -gx
        lhs = quad.f(x + alphas[k] * d)
        rhs = quad.f(x) + 1e-4 * alphas[k] * float(gx @ d)
        assert float(lhs) <= float(rhs) + 1e-10
