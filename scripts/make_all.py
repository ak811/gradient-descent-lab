from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from scripts.make_quadratic_figures import main as quad_main
from scripts.make_scipy_comparison import main as scipy_main
from scripts.make_nonconvex_figures import main as nonconvex_main
from scripts.make_diabetes_figures import main as diabetes_main

def main() -> None:
    quad_main()
    scipy_main()
    nonconvex_main()
    diabetes_main()

if __name__ == "__main__":
    main()
