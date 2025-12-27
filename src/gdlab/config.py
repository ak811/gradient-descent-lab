from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "Figures"

DEFAULT_BETA = 1.75
DEFAULT_GRID_SIZE = 400
DEFAULT_PLOT_RANGE = 4.0
DEFAULT_CENTER = (0.5, -1.5)
DEFAULT_DPI = 200

def ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR
