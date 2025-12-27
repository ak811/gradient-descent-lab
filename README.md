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

> **Note:** GitHub Markdown does not support consistent image sizing with `![...](...)`.  
> The images below use HTML `<img>` tags so every figure is displayed at the same width.

### Quadratic (2D)

1. **Objective surface**  
<p align="center">
  <img src="figures/quadratic_surface.png" alt="Quadratic surface" width="380" />
</p>

2. **Objective contours**  
<p align="center">
  <img src="figures/quadratic_contours.png" alt="Quadratic contours" width="380" />
</p>

3. **Contour plot with a single reference point**  
<p align="center">
  <img src="figures/quadratic_traj_xempty.png" alt="Single-point trajectory" width="380" />
</p>

4. **Gradient Descent trajectory (α = 0.5)**  
<p align="center">
  <img src="figures/gd_traj_alpha_0p5.png" alt="GD trajectory alpha=0.5" width="380" />
</p>

5. **Gradient Descent trajectory (α = 1.0)**  
<p align="center">
  <img src="figures/gd_traj_alpha_1p0.png" alt="GD trajectory alpha=1.0" width="380" />
</p>

6. **Gradient Descent trajectory (α = 1/L)**  
<p align="center">
  <img src="figures/gd_traj_alpha_theorem.png" alt="GD trajectory theorem alpha" width="380" />
</p>

7. **Optimality gap (α = 0.5 vs α = 1/L)**  
<p align="center">
  <img src="figures/gap_quadratic_b_vs_d.png" alt="Optimality gaps b vs d" width="380" />
</p>

8. **SciPy solver iterates on contour plot**  
<p align="center">
  <img src="figures/scipy_iterates_contour.png" alt="SciPy iterates contour" width="380" />
</p>

9. **Optimality gaps: SciPy vs GD**  
<p align="center">
  <img src="figures/scipy_vs_gd_gaps.png" alt="SciPy vs GD gaps" width="380" />
</p>

### Nonconvex (1D)

10. **Multi-start histogram summary (minimizers and objective values)**  
<p align="center">
  <img src="figures/nonconvex_histograms.png" alt="Nonconvex histograms" width="380" />
</p>

11. **Objective function with detected local minima**  
<p align="center">
  <img src="figures/nonconvex_function_minima.png" alt="Nonconvex function minima" width="380" />
</p>

### Least Squares (Diabetes dataset)

12. **Fixed-stepsize GD: optimality gaps**  
<p align="center">
  <img src="figures/diabetes_gd_gaps.png" alt="Fixed-stepsize GD gaps" width="380" />
</p>

13. **Fixed steps vs Exact line search**  
<p align="center">
  <img src="figures/diabetes_fixed_vs_exact.png" alt="Fixed vs exact" width="380" />
</p>

14. **Fixed steps vs Exact line search vs Backtracking**  
<p align="center">
  <img src="figures/diabetes_fixed_vs_exact_vs_bt.png" alt="Fixed vs exact vs backtracking" width="380" />
</p>

15. **Stepsizes over iterations (Exact vs Backtracking)**  
<p align="center">
  <img src="figures/stepsizes_exact_vs_bt.png" alt="Stepsizes exact vs backtracking" width="380" />
</p>

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
