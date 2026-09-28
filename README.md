# CS164 — Optimization Methods

Course notes, numerical experiments, and Python implementations for CS164 Optimization Methods.

## Repository structure

- [`notebooks/`](notebooks/) — Jupyter notebooks with explanations, exercises, and visualizations.
- [`src/`](src/) — Python scripts and reusable implementations corresponding to the sessions.
- [`requirements.txt`](requirements.txt) — Python dependencies.

## Sessions

### Session 1 — What and Why to Optimize

**Main contents**
- Objective functions, design variables, and feasible sets.
- Formulating a simple optimization problem using a thumbnail-brightness example.
- Plotting univariate functions and bivariate contour maps.

**Files**
- [Jupyter notebook](notebooks/01_what_why_optimize.ipynb)
- [Python script](src/why_optimize.py)

### Session 2 — Taylor Series and Numerical Approximation

**Main contents**
- Univariate Taylor series and first-, second-, and third-order Taylor polynomials.
- Approximating `sin(x)` around different expansion points.
- Multivariate linear and quadratic approximations using gradients and Hessians.
- Comparing original functions with their Taylor approximations in plots.

**Files**
- [Jupyter notebook](notebooks/02_taylor_series_numerical_approximation.ipynb)
- [Python script](src/taylor_series.py)

### Session 3 — Quadratic Forms

**Main contents**
- Reviewing stationary points, second-derivative tests, and unconstrained optimization.
- Quadratic approximations of multivariable functions and their Hessians.
- Comparing nonlinear surfaces with their quadratic Taylor approximations.
- Connecting minimization to the least-squares objective in linear regression.

**Files**
- [Jupyter notebook](notebooks/03_quadratic_forms.ipynb)
- [Python script](src/quadratic_forms.py)

### Session 4 — Tests for Positive Definiteness

**Main contents**
- Symmetric matrices, real eigenvalues, and the connection to positive definiteness.
- Two-dimensional positive-definiteness criteria and determinant counterexamples.
- Quadratic forms, bowls, saddles, and contour/level-set visualizations.
- Gradients of linear and quadratic matrix expressions.

**Files**
- [Jupyter notebook](notebooks/04_positive_definiteness.ipynb)
- [Python script](src/positive_definiteness.py)

### Session 5 — Bracketing Local Minima

**Main contents**
- Local minima of one-variable functions and derivative-free bracketing.
- Central finite differences for approximating derivatives.
- Expanding intervals to bracket a minimum.
- Using derivative signs and bisection to narrow a bracket.

**Files**
- [Jupyter notebook](notebooks/05_bracketing.ipynb)
- [Python script](src/bracketing.py)

### Session 6 — Descent and Exact Line Search

**Main contents**
- Descent directions, step sizes, and iterative updates.
- Visualizing descent steps on contour plots.
- Reducing multivariable optimization to a one-dimensional line-search problem.
- Exact line search using numerical derivatives, interval expansion, and bisection.

**Files**
- [Jupyter notebook](notebooks/06_descent_line_search.ipynb)
- [Python script](src/line_search.py)

### Session 7 — Gradient Descent and Approximate Line Search

**Main contents**
- Exact versus approximate line search and the Wolfe conditions.
- Armijo backtracking and sufficient-decrease tests.
- Absolute and relative improvement, gradient-norm, and maximum-iteration stopping conditions.
- Normalized gradient descent with central finite differences.
- Testing on a quadratic bowl and a saddle; examining exact line search on the saddle.

**Files**
- [Jupyter notebook](notebooks/07_gradient_descent.ipynb)
- [Python script](src/gradient_descent.py)

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

To run a session's Python script directly, for example:

```bash
python src/gradient_descent.py
```
