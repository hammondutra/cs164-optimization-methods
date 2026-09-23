# CS164 - Optimization Methods

A learning repository for numerical optimization methods studied in CS164.

## Structure

- `notebooks/` - concept explanations, derivations, experiments, plots, and reflections.
- `src/` - reusable Python implementations extracted from the notebooks.

## Current Topics

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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt