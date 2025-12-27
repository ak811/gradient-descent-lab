from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gdlab.config import ensure_figures_dir, DEFAULT_BETA, DEFAULT_GRID_SIZE, DEFAULT_PLOT_RANGE, DEFAULT_CENTER
from gdlab.objectives.quadratic2d import Quadratic2D
from gdlab.optim.gd import gradient_descent
from gdlab.plotting.contours import contour_trajectory
from gdlab.plotting.lines import semilog_lines

def main() -> None:
    figs = ensure_figures_dir()
    quad = Quadratic2D(DEFAULT_BETA)

    cx, cy = DEFAULT_CENTER
    r, n = DEFAULT_PLOT_RANGE, DEFAULT_GRID_SIZE
    tx = np.linspace(cx - r, cx + r, n)
    ty = np.linspace(cy - r, cy + r, n)
    xg, yg = np.meshgrid(tx, ty)
    z = quad.f(np.array([xg, yg]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xg, yg, z, cmap="viridis")
    fig.colorbar(surf)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
    fig.savefig(figs / "quadratic_surface.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.contourf(tx, ty, z, 20)
    plt.colorbar().set_label("f(x,y)", rotation=270)
    plt.xlabel("x"); plt.ylabel("y")
    fig.savefig(figs / "quadratic_contours.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    x0 = np.array([1.0, 4.0])
    Xb, fb = gradient_descent(quad.f, quad.grad, x0, alpha=0.5, niter=300)
    contour_trajectory(quad.f, Xb, figs / "gd_traj_alpha_0p5.png")

    Xc, fc = gradient_descent(quad.f, quad.grad, x0, alpha=1.0, niter=300)
    contour_trajectory(quad.f, Xc, figs / "gd_traj_alpha_1p0.png")

    ad = quad.theorem_step()
    Xd, fd = gradient_descent(quad.f, quad.grad, x0, alpha=ad, niter=300)
    contour_trajectory(quad.f, Xd, figs / "gd_traj_alpha_theorem.png")

    fstar = quad.f_star()
    gaps = {"alpha=0.5": fb - fstar, "alpha=1/L": fd - fstar}
    semilog_lines(gaps, figs / "gap_quadratic_b_vs_d.png", ylabel="Optimality gap")

if __name__ == "__main__":
    main()
