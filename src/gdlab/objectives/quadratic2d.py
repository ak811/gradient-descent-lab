from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gdlab.utils.linalg import sym_eig_minmax

@dataclass(frozen=True)
class Quadratic2D:
    beta: float = 1.75

    def hessian(self) -> np.ndarray:
        return np.array([[2.0, self.beta], [self.beta, 2.0]])

    def f(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return x[0] ** 2 + x[1] ** 2 + self.beta * x[0] * x[1] + x[0] + 2.0 * x[1]

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.array([2.0 * x[0] + self.beta * x[1] + 1.0, 2.0 * x[1] + self.beta * x[0] + 2.0])

    def minimizer(self) -> np.ndarray:
        H = self.hessian()
        c = np.array([1.0, 2.0])
        return -np.linalg.solve(H, c)

    def f_star(self) -> float:
        xstar = self.minimizer()
        return float(self.f(xstar))

    def lipschitz_L(self) -> float:
        _, mx = sym_eig_minmax(self.hessian())
        return mx

    def theorem_step(self) -> float:
        return 1.0 / self.lipschitz_L()
