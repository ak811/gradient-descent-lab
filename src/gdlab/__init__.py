from .config import FIGURES_DIR
from .objectives.quadratic2d import Quadratic2D
from .objectives.least_squares import load_diabetes_data, LeastSquares
from .optim.gd import gradient_descent

__all__ = ["FIGURES_DIR", "Quadratic2D", "load_diabetes_data", "LeastSquares", "gradient_descent"]
