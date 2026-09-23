"""Exact line-search utilities for multivariate optimization."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def central_difference(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    """Approximate f'(x) using a centered finite difference."""
    return (f(x + h) - f(x - h)) / (2 * h)


def expand_interval(
    f: Callable[[float], float],
    a: float = 0.0,
    b: float = 0.25,
    max_iter: int = 100,
) -> tuple[float, float]:
    """
    Expand the upper bound until a local minimum is bracketed.

    Requires f'(a) < 0 and seeks b such that f'(b) > 0.
    """
    if central_difference(f, a) >= 0:
        raise ValueError("The chosen direction is not a descent direction.")

    for _ in range(max_iter):
        if central_difference(f, b) > 0:
            return a, b

        b *= 2

    raise RuntimeError("Could not bracket a local minimum.")


def bisection_minimum(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
) -> float:
    """Approximate a local minimum by applying bisection to f'."""
    if central_difference(f, a) >= 0 or central_difference(f, b) <= 0:
        raise ValueError("Need f'(a) < 0 and f'(b) > 0.")

    for _ in range(max_iter):
        m = (a + b) / 2
        dm = central_difference(f, m)

        if abs(dm) < 1e-12:
            return m

        if dm > 0:
            b = m
        else:
            a = m

        if abs(b - a) < epsilon:
            return (a + b) / 2

    raise RuntimeError("Maximum number of iterations reached.")


def exact_line_search(
    f: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    initial_interval: tuple[float, float] = (0.0, 0.25),
) -> tuple[float, NDArray[np.float64]]:
    """
    Find a step size alpha minimizing f(x + alpha*d).

    Returns
    -------
    alpha_star:
        Approximate optimal step size along d.
    x_next:
        New design point x + alpha_star*d.
    """
    phi = lambda alpha: f(x + alpha * d)

    a, b = expand_interval(
        phi,
        initial_interval[0],
        initial_interval[1],
    )

    alpha_star = bisection_minimum(phi, a, b)
    x_next = x + alpha_star * d

    return alpha_star, x_next