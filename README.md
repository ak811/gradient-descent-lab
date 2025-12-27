# Gradient Descent Lab (Scientific Project Repo)

This repository is an implementation of gradient-based optimization methods across three classic problem families:

- **2D quadratic** objective (smooth, strongly convex)  
- **1D nonconvex** objective (multi-start behavior and local minima)  
- **Least-squares linear regression** (Diabetes dataset)

Implemented methods:

- fixed-step **Gradient Descent**
- theorem-based stepsize choice via smoothness constant (**1/L**)
- **SciPy** baseline solver (**BFGS**)
- **Exact line search** for least squares
- **Backtracking Armijo line search**

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproduce all figures

```bash
python scripts/make_all.py
```

All plots are saved into:

- `figures/`

---

## Results Gallery (15 figures)

### Quadratic (2D)

1. **Objective surface**  
![Quadratic surface](figures/quadratic_surface.png)

2. **Objective contours**  
![Quadratic contours](figures/quadratic_contours.png)

3. **Contour plot with a single reference point**  
![Single-point trajectory](figures/quadratic_traj_xempty.png)

4. **Gradient Descent trajectory (α = 0.5)**  
![GD trajectory alpha=0.5](figures/gd_traj_alpha_0p5.png)

5. **Gradient Descent trajectory (α = 1.0)**  
![GD trajectory alpha=1.0](figures/gd_traj_alpha_1p0.png)

6. **Gradient Descent trajectory (α = 1/L)**  
![GD trajectory theorem alpha](figures/gd_traj_alpha_theorem.png)

7. **Optimality gap (α = 0.5 vs α = 1/L)**  
![Optimality gaps b vs d](figures/gap_quadratic_b_vs_d.png)

8. **SciPy solver iterates on contour plot**  
![SciPy iterates contour](figures/scipy_iterates_contour.png)

9. **Optimality gaps: SciPy vs GD**  
![SciPy vs GD gaps](figures/scipy_vs_gd_gaps.png)

### Nonconvex (1D)

10. **Multi-start histogram summary (minimizers and objective values)**  
![Nonconvex histograms](figures/nonconvex_histograms.png)

11. **Objective function with detected local minima**  
![Nonconvex function minima](figures/nonconvex_function_minima.png)

### Least Squares (Diabetes dataset)

12. **Fixed-stepsize GD: optimality gaps**  
![Fixed-stepsize GD gaps](figures/diabetes_gd_gaps.png)

13. **Fixed steps vs Exact line search**  
![Fixed vs exact](figures/diabetes_fixed_vs_exact.png)

14. **Fixed steps vs Exact line search vs Backtracking**  
![Fixed vs exact vs backtracking](figures/diabetes_fixed_vs_exact_vs_bt.png)

15. **Stepsizes over iterations (Exact vs Backtracking)**  
![Stepsizes exact vs backtracking](figures/stepsizes_exact_vs_bt.png)

---

## Tests

```bash
pytest -q
```

The test suite includes:
- finite-difference gradient checks for the quadratic
- descent sanity checks
- Armijo condition verification for backtracking
- decrease check for exact line search on least squares

---

## Citation

If you use this repository as a reference, please cite via `CITATION.cff`.
