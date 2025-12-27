from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gdlab.objectives.least_squares import load_diabetes_data, LeastSquares
from gdlab.optim.exact_line_search import alpha_exact_ls

def test_exact_line_search_decreases_ls() -> None:
    A, y = load_diabetes_data(scaled=True)
    ls = LeastSquares(A, y)
    x = np.zeros(A.shape[1])
    a = alpha_exact_ls(A, y, x)
    assert a >= 0.0
    g = ls.grad(x)
    assert ls.f(x - a * g) <= ls.f(x) + 1e-12
