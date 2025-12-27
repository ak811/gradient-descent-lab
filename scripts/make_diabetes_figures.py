from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gdlab.config import ensure_figures_dir
from gdlab.objectives.least_squares import load_diabetes_data, LeastSquares
from gdlab.optim.gd import gradient_descent
from gdlab.optim.exact_line_search import gd_exact_line_search_ls
from gdlab.optim.backtracking import gd_backtracking_armijo
from gdlab.plotting.lines import semilog_lines, line_plot

def main() -> None:
    figs = ensure_figures_dir()
    A, y = load_diabetes_data(scaled=True)
    ls = LeastSquares(A, y)
    xstar = ls.normal_eq_solution()
    fstar = ls.f(xstar)
    L, mu = ls.L_mu()

    x0 = np.zeros(A.shape[1])
    T = 10000
    runs = {
        "1/L": gradient_descent(ls.f, ls.grad, x0, 1.0 / L, T)[1] - fstar,
        "1/(L+mu)": gradient_descent(ls.f, ls.grad, x0, 1.0 / (L + mu), T)[1] - fstar,
        "2/L": gradient_descent(ls.f, ls.grad, x0, 2.0 / L, T)[1] - fstar,
        "2/(L+mu)": gradient_descent(ls.f, ls.grad, x0, 2.0 / (L + mu), T)[1] - fstar,
    }
    semilog_lines(runs, figs / "diabetes_gd_gaps.png", ylabel="Optimality gap")

    Xe, fe, ae = gd_exact_line_search_ls(A, y, x0, niter=T)
    Xb, fb, ab = gd_backtracking_armijo(ls.f, ls.grad, x0, alpha0=1000.0, c1=0.9, beta=0.9, niter=1000)

    combined = dict(runs)
    combined["Exact line search"] = fe - fstar
    combined["Backtracking"] = fb - fstar
    semilog_lines(combined, figs / "diabetes_exact_vs_fixed_vs_bt.png", ylabel="Optimality gap")

    line_plot({"Exact line search": ae[:500], "Backtracking": ab[:500]}, figs / "stepsizes_exact_vs_bt.png", ylabel="Step size")

if __name__ == "__main__":
    main()
