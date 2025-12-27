from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def contour_trajectory(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    outpath: Path,
    levels: int = 20,
    gridsize: int = 400,
    pad: float = 0.5,
) -> None:
    xmin, xmax = float(X[0].min()), float(X[0].max())
    ymin, ymax = float(X[1].min()), float(X[1].max())
    tx = np.linspace(xmin - pad, xmax + pad, gridsize)
    ty = np.linspace(ymin - pad, ymax + pad, gridsize)
    xg, yg = np.meshgrid(tx, ty)
    z = f(np.array([xg, yg]))

    fig = plt.figure()
    plt.contourf(xg, yg, z, levels=levels)
    plt.axis("equal")
    plt.plot(X[0, :], X[1, :], "c.-")
    plt.xlabel("x")
    plt.ylabel("y")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
