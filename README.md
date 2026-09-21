# CS164 - Optimization Methods

A learning repository for numerical optimization methods studied in CS164.

## Structure

- `notebooks/` - concept explanations, derivations, experiments, plots, and reflections.
- `src/` - reusable Python implementations extracted from the notebooks.

## Current topics

### Bracketing local minima

The first notebook covers:

- local minima of univariate functions,
- derivative-free bracketing,
- the Intermediate Value Theorem,
- central finite differences,
- interval expansion,
- bisection on the derivative.

Open [`notebooks/01_bracketing.ipynb`](notebooks/01_bracketing.ipynb) for the full walkthrough.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then select the `.venv` Python interpreter/kernel in VS Code or Jupyter.
