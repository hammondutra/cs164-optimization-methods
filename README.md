# CS164 — Optimization Methods

Course notes, numerical experiments, and Python/R implementations for CS164 Optimization Methods.

## Repository structure

- [`notebooks/`](notebooks/) — Jupyter notebooks with explanations, exercises, and visualizations.
- [`src/`](src/) — Python and R scripts with implementations corresponding to the sessions.
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

### Session 15 — Applying & Interpreting the KKT Conditions

**Main contents**
- Formulating nonnegative least-squares fitting for three drinks and six flavor dimensions.
- Constructing the Lagrangian and applying the four KKT conditions.
- Enumerating all eight active sets with an R solver, including the nonnegativity and sum-to-one constraints.
- Interpreting binding constraints and nonzero inequality multipliers as sensitivity information.
- Comparing the physically feasible optimal blend with the equality-only least-squares fit.

**Files**
- [Jupyter notebook (R / IRkernel)](notebooks/15_applying_interpreting_kkt.ipynb)
- [R script](src/kkt_applications.R)

### Session 16 — Linear Programming

**Main contents**
- Matrix-form LP constraints, affine functions, convex objectives, and convex feasible sets.
- Graphical feasible regions, vertices, linear and nonlinear level curves, and unbounded optimization.
- Industrial profit maximization: \`(x,y)=(1,1)\`, maximum profit \`14\`; optional KKT certificate.
- A greedy vertex-walking algorithm as an introduction to simplex.
- Numerical verification with Python/CVXPY and review of L1, L2, and L-infinity norms.

**Files**
- [R notebook — Questions 1–6 and 8](notebooks/16_linear_programming_R.ipynb)
- [Python notebook — Question 7](notebooks/16_linear_programming_Python.ipynb)
- [R script — graphical experiments](src/linear_programming.R)
- [Python script — CVXPY solver](src/linear_programming.py)

Separate kernels keep the examples reproducible: R (IRkernel) for plots, Python (ipykernel) for CVXPY. While R can call Python through \`reticulate\`, that introduces another runtime dependency for a single exercise.

\`\`\`bash
Rscript src/linear_programming.R   # saves session16_plots.pdf
python src/linear_programming.py
\`\`\`

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


For the Session 15 R notebook, install R and register its Jupyter kernel (IRkernel). For example, in an R console:

```r
install.packages("IRkernel")
IRkernel::installspec(user = TRUE)
```

You can run the standalone session script with:

```bash
Rscript src/kkt_applications.R
```
