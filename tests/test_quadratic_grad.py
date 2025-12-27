from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gdlab.objectives.quadratic2d import Quadratic2D

def test_quadratic_grad_matches_fd() -> None:
    quad = Quadratic2D(1.75)
    x = np.array([0.3, -1.1])
    g = quad.grad(x)
    eps = 1e-6
    gfd = np.zeros_like(g)
    for i in range(2):
        e = np.zeros(2); e[i] = 1.0
        gfd[i] = (quad.f(x + eps * e) - quad.f(x - eps * e)) / (2 * eps)
    assert np.allclose(g, gfd, atol=1e-5, rtol=1e-5)
