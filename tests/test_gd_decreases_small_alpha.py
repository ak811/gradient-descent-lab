from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gdlab.objectives.quadratic2d import Quadratic2D
from gdlab.optim.gd import gradient_descent

def test_gd_decreases_for_safe_step() -> None:
    quad = Quadratic2D(1.75)
    x0 = np.array([1.0, 4.0])
    alpha = quad.theorem_step()
    _, fvals = gradient_descent(quad.f, quad.grad, x0, alpha, niter=30)
    assert fvals[-1] < fvals[0]
    assert np.all(np.diff(fvals[5:]) <= 1e-9 + 0.0)  # eventually monotone (numerical tolerance)
