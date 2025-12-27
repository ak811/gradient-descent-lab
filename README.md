# Gradient Descent Lab: Methods and Empirical Behavior

This repository implements gradient-based optimization methods across three classic problem families:

- **2D quadratic** objective (smooth, strongly convex)
- **1D nonconvex** objective (multi-start behavior and local minima)
- **Least-squares linear regression** (Diabetes dataset)

Implemented methods:

- fixed-step **Gradient Descent (GD)**
- theorem-based stepsize choice via smoothness constant (**1/L**)
- **SciPy** baseline solver (**BFGS**)
- **Exact line search** for least squares
- **Backtracking Armijo line search**

---

## Install

```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Reproduce all figures

```bash
python scripts/make_all.py
```

### Where figures are saved

figures are saved to **`figures/`** (capital F) via `src/gdlab/config.py`:

```python
figures_DIR = PROJECT_ROOT / "figures"
```

The image links below assume figures live in **`figures/`**.

---

## Main math formulas

### Gradient Descent (GD)

For a differentiable objective $f:\mathbb{R}^d\to\mathbb{R}$, GD iterates as:
$$
x_{k+1} = x_k - \alpha_k \nabla f(x_k).
$$

**Optimality gap** (used in many plots):
$$
\mathrm{gap}_k := f(x_k) - f^\star,\qquad f^\star = \min_x f(x).
$$

---

## Objective 1: 2D quadratic

### Definition

For parameter $\beta \in \mathbb{R}$,
$$
f(x,y) = x^2 + y^2 + \beta xy + x + 2y.
$$

### Gradient and Hessian

$$
\nabla f(x,y)=
\begin{pmatrix}
2x + \beta y + 1\\
2y + \beta x + 2
\end{pmatrix},
\qquad
\nabla^2 f =
\begin{pmatrix}
2 & \beta\\
\beta & 2
\end{pmatrix}.
$$

### Minimizer

The minimizer satisfies $\nabla f(x^\star)=0$, equivalently:
$$
\nabla^2 f\,x^\star + \begin{pmatrix}1\\2\end{pmatrix}=0
\quad\Rightarrow\quad
x^\star = -(\nabla^2 f)^{-1}\begin{pmatrix}1\\2\end{pmatrix}.
$$

### Smoothness constant $L$ and theorem step size

For this quadratic, $\nabla f$ is $L$-Lipschitz with
$$
L = \lambda_{\max}(\nabla^2 f).
$$
A standard safe fixed step size is
$$
\alpha = \frac{1}{L}.
$$

---

## Objective 2: 1D nonconvex

### Definition

$$
f(x) = \left(x^2 + x(1+\sin x) + 2\right)\exp\left(-\frac{|x|}{3}\right).
$$

This objective has multiple local minima, so local optimizers can converge to different solutions depending on initialization.

---

## Objective 3: least squares (Diabetes dataset)

### Definition

Given $A \in \mathbb{R}^{m\times n}$, $y\in\mathbb{R}^m$,
$$
f(x) = \frac{1}{m}\|Ax-y\|_2^2.
$$

### Gradient and Hessian

$$
\nabla f(x) = \frac{2}{m}A^\top(Ax-y),
\qquad
\nabla^2 f(x) = \frac{2}{m}A^\top A.
$$

### Smoothness $L$ and strong convexity $\mu$

$$
L = \lambda_{\max}\!\left(\frac{2}{m}A^\top A\right),\quad
\mu = \lambda_{\min}\!\left(\frac{2}{m}A^\top A\right).
$$

### Exact line search (least squares)

Along steepest descent direction $d_k=-\nabla f(x_k)$,
$$
\alpha_k = \arg\min_{\alpha\ge 0} f(x_k+\alpha d_k).
$$
For least squares (quadratic), this has a closed form:
$$
\alpha_k = \frac{\|\nabla f(x_k)\|_2^2}{\nabla f(x_k)^\top \nabla^2 f \,\nabla f(x_k)},
\qquad
\nabla^2 f=\frac{2}{m}A^\top A.
$$

### Backtracking Armijo line search

Starting from $\alpha_0$, shrink $\alpha \leftarrow \beta\alpha$ until:
$$
f(x_k + \alpha d_k) \le f(x_k) + c_1\alpha \nabla f(x_k)^\top d_k,
\quad d_k=-\nabla f(x_k),
\quad 0<c_1<1,\; 0<\beta<1.
$$

---

## Results

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

4. **Gradient Descent trajectory ($\alpha = 0.5$)**  
<p align="center">
  <img src="figures/gd_traj_alpha_0p5.png" alt="GD trajectory alpha=0.5" width="380" />
</p>

5. **Gradient Descent trajectory ($\alpha = 1.0$)**  
<p align="center">
  <img src="figures/gd_traj_alpha_1p0.png" alt="GD trajectory alpha=1.0" width="380" />
</p>

6. **Gradient Descent trajectory ($\alpha = 1/L$)**  
<p align="center">
  <img src="figures/gd_traj_alpha_theorem.png" alt="GD trajectory theorem alpha" width="380" />
</p>

7. **Optimality gap ($\alpha = 0.5$ vs $\alpha = 1/L$)**  
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

---

## License

MIT (see `LICENSE`).
