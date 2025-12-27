from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_diabetes
from gdlab.utils.linalg import sym_eig_minmax

def load_diabetes_data(scaled: bool = True) -> tuple[np.ndarray, np.ndarray]:
    A, y = load_diabetes(return_X_y=True, scaled=scaled)
    return np.asarray(A, float), np.asarray(y, float)

@dataclass(frozen=True)
class LeastSquares:
    A: np.ndarray
    y: np.ndarray

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    def f(self, x: np.ndarray) -> float:
        r = self.A @ x - self.y
        return float((r @ r) / self.m)

    def grad(self, x: np.ndarray) -> np.ndarray:
        r = self.A @ x - self.y
        return (2.0 / self.m) * (self.A.T @ r)

    def normal_eq_solution(self) -> np.ndarray:
        return np.linalg.lstsq(self.A, self.y, rcond=None)[0]

    def L_mu(self) -> tuple[float, float]:
        H = (2.0 / self.m) * (self.A.T @ self.A)
        mn, mx = sym_eig_minmax(H)
        return mx, mn
