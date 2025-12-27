from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gdlab.config import ensure_figures_dir, DEFAULT_BETA
from gdlab.objectives.quadratic2d import Quadratic2D
from gdlab.optim.gd import gradient_descent
from gdlab.plotting.lines import semilog_lines

def main() -> None:
    figs = ensure_figures_dir()
    quad = Quadratic2D(DEFAULT_BETA)
    x0 = np.array([1.0, 4.0])

    res = minimize(lambda x: float(quad.f(x)), x0, jac=quad.grad, method="BFGS", options={"return_all": True})
    allvecs = np.array(res.allvecs)  # (k,2)
    Xs = allvecs.T

    Xb, fb = gradient_descent(quad.f, quad.grad, x0, alpha=0.5, niter=300)
    Xd, fd = gradient_descent(quad.f, quad.grad, x0, alpha=quad.theorem_step(), niter=300)

    fstar = quad.f_star()
    gaps = {
        "SciPy(BFGS)": quad.f(Xs) - fstar,
        "GD alpha=0.5": fb - fstar,
        "GD alpha=1/L": fd - fstar,
    }
    semilog_lines(gaps, figs / "scipy_vs_gd_gaps.png", ylabel="Optimality gap")

if __name__ == "__main__":
    main()
