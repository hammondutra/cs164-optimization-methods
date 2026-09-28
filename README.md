# CS164 - Optimization Methods

A learning repository for numerical optimization methods studied in CS164.

## Structure

- `notebooks/` - concept explanations, derivations, experiments, plots, and reflections.
- `src/` - reusable Python implementations extracted from the notebooks.

## Current Topics

### Session 1 — What and Why to Optimize

Covers objectives, design variables, feasible sets, and Python/SageMath plotting warmups.
Notebook: [`notebooks/01_what_why_optimize.ipynb`](notebooks/01_what_why_optimize.ipynb) · Python: [`src/01_what_why_optimize.py`](src/01_what_why_optimize.py).

### Session 2 — Taylor Series and Numerical Approximation

Covers univariate and multivariate Taylor polynomials, first- through third-order expansions and Python translations of the original plots.
Notebook: [`notebooks/02_taylor_series_numerical_approximation.ipynb`](notebooks/02_taylor_series_numerical_approximation.ipynb) · Python: [`src/02_taylor_series_numerical_approximation.py`](src/02_taylor_series_numerical_approximation.py).

### Session 3 — Quadratic Forms

Covers Hessian-based quadratic approximations, second-derivative tests, the linear-regression objective and surface comparisons. The supplied PDF shows Q1–Q6 of nine questions; missing questions are not reconstructed.
Notebook: [`notebooks/03_quadratic_forms.ipynb`](notebooks/03_quadratic_forms.ipynb) · Python: [`src/03_quadratic_forms.py`](src/03_quadratic_forms.py).

### Session 4 — Tests for Positive Definiteness

Covers symmetric matrices, eigenvalues, Sylvester's criterion, determinant counterexamples, contour and level-set plots, and matrix-polynomial gradients. The last source response was truncated in the PDF.
Notebook: [`notebooks/04_positive_definiteness.ipynb`](notebooks/04_positive_definiteness.ipynb) · Python: [`src/04_positive_definiteness.py`](src/04_positive_definiteness.py).

### Session 5 — Bracketing Local Minima

This notebook covers:

- local minima of univariate functions,
- derivative-free bracketing,
- the Intermediate Value Theorem,
- central finite differences,
- interval expansion,
- bisection on the derivative.

Open [`notebooks/05_bracketing.ipynb`](notebooks/05_bracketing.ipynb) for the full walkthrough.

Reusable implementations are available in [`src/bracketing.py`](src/bracketing.py).

### Session 6 — Descent and Exact Line Search

This notebook covers:

- descent-direction iteration,
- direction vectors and step sizes,
- contour visualization of descent steps,
- reducing a multivariable objective to a one-dimensional line-search problem,
- exact line search,
- numerical derivatives using central differences,
- bisection for finding a line-search minimum,
- automatic interval expansion,
- exact line search on a quadratic bowl.

Open [`notebooks/06_descent_line_search.ipynb`](notebooks/06_descent_line_search.ipynb) for the full walkthrough.

Reusable line-search implementations are available in [`src/line_search.py`](src/line_search.py).

### Session 7 — Gradient Descent and Approximate Line Search

This notebook covers:

- exact versus approximate line search and the Wolfe conditions,
- Armijo backtracking with three simple numerical tests,
- absolute and relative improvement, gradient-norm and iteration-limit stopping rules,
- central finite differences and normalized gradient descent,
- numerical-versus-exact comparison for the quadratic bowl,
- saddle-point behavior under backtracking and analytical exact line search.

Open [`notebooks/07_gradient_descent.ipynb`](notebooks/07_gradient_descent.ipynb) for the complete Session 7 PCW.

Reusable implementations and executable tests are in [`src/gradient_descent.py`](src/gradient_descent.py).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
