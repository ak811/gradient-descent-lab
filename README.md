# Gradient Descent Lab (Scientific Project Repo)

This repository is a modular, reproducible implementation of gradient-based optimization methods across three classic problem families:

- **2D quadratic** objective (smooth, strongly convex)  
- **1D nonconvex** objective (multi-start behavior and local minima)  
- **Least-squares linear regression** (Diabetes dataset)

It includes:

- fixed-step **Gradient Descent**
- theorem-based stepsize choice via smoothness constant (**1/L**)
- **SciPy** baseline solver (**BFGS**)
- **Exact line search** for least squares
- **Backtracking Armijo line search**

All scripts generate deterministic plots saved into `Figures/`.

---

## Install

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproduce all figures

Run the full pipeline:

```bash
python scripts/make_all.py
```

All plots are saved into:

- `Figures/`

> Note: The README embeds images by filename. If you already have images saved locally, keep the same names in `Figures/`.
> The scripts are written to generate the same filenames so the gallery stays consistent.

---

## Results Gallery (from `Figures/`)

### Quadratic (2D)

![Quadratic surface](Figures/quadratic_surface.png)

![Quadratic contours](Figures/quadratic_contours.png)

![GD trajectory alpha=0.5](Figures/gd_traj_alpha_0p5.png)

![GD trajectory alpha=1.0](Figures/gd_traj_alpha_1p0.png)

![GD trajectory theorem alpha](Figures/gd_traj_alpha_theorem.png)

![Optimality gaps b vs d](Figures/gap_quadratic_b_vs_d.png)

![SciPy vs GD gaps](Figures/scipy_vs_gd_gaps.png)

### Nonconvex (1D)

![Histogram of minimizers](Figures/nonconvex_hist_minimizers.png)

![Histogram of objective values](Figures/nonconvex_hist_objvals.png)

### Least Squares (Diabetes dataset)

![Fixed-stepsize GD gaps](Figures/diabetes_gd_gaps.png)

![Fixed vs exact vs backtracking](Figures/diabetes_exact_vs_fixed_vs_bt.png)

![Stepsizes exact vs backtracking](Figures/stepsizes_exact_vs_bt.png)

---

## Project layout

```text
gradient-descent-lab/
├─ README.md
├─ requirements.txt
├─ Figures/                  # all saved plots used by README
├─ src/gdlab/                # reusable library code (objectives, optimizers, plotting utils)
├─ scripts/                  # reproducible figure-generation scripts
└─ tests/                    # quick correctness + sanity tests
```

### Key modules

- `src/gdlab/objectives/`
  - `quadratic2d.py`: quadratic objective, gradient, Hessian, closed-form minimizer
  - `nonconvex1d.py`: nonconvex objective used for multi-start experiments
  - `least_squares.py`: Diabetes dataset + least squares objective/gradient and (L, μ)

- `src/gdlab/optim/`
  - `gd.py`: fixed-step gradient descent
  - `exact_line_search.py`: exact stepsize for least squares
  - `backtracking.py`: Armijo backtracking GD

- `src/gdlab/plotting/`
  - `contours.py`: contour + trajectory plots
  - `lines.py`: semilog gap plots, step-size plots
  - `hist.py`: histograms

---

## Reproducibility notes

- Randomness is seeded in nonconvex experiments.
- Scripts create `Figures/` if it does not exist.
- Figures are saved with stable filenames used by this README.
- The codebase is split into small modules (each < 100 lines) by design.

---

## Tests

Run the test suite:

```bash
pytest -q
```

Tests include:
- finite-difference gradient checks for the quadratic
- basic descent/convergence sanity checks
- Armijo condition verification for backtracking
- decrease guarantee check for exact line search on least squares

---

## Citation

If you use this repository as a reference, please cite via `CITATION.cff`.
