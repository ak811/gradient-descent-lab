from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gdlab.config import ensure_figures_dir
from gdlab.utils.seeds import set_global_seed
from gdlab.objectives.nonconvex1d import nonconvex_f
from gdlab.plotting.hist import histogram

def main(n_samples: int = 10000, seed: int = 0) -> None:
    figs = ensure_figures_dir()
    set_global_seed(seed)

    x0s = np.random.uniform(-10.0, 10.0, size=n_samples)
    sols = np.empty(n_samples, float)
    for i, x0 in enumerate(x0s):
        res = minimize(lambda t: float(nonconvex_f(t[0])), x0=np.array([x0]), method="BFGS", options={"maxiter": 50})
        sols[i] = float(res.x[0])

    histogram(sols, figs / "nonconvex_hist_minimizers.png", bins=100, xlabel="Estimated minimizer x")
    histogram(nonconvex_f(sols), figs / "nonconvex_hist_objvals.png", bins=100, xlabel="Objective value f(x)")

if __name__ == "__main__":
    main()
