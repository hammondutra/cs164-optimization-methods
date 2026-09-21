"""Bracketing and bisection utilities for one-dimensional optimization."""

from collections.abc import Callable


def bracket_minimum(
    f: Callable[[float], float],
    x0: float,
    h: float,
    c: float,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Find an interval that brackets a local minimum without using derivatives."""
    if h == 0:
        raise ValueError("h must be nonzero")
    if c <= 1:
        raise ValueError("c must be greater than 1")

    a = x0
    fa = f(a)
    b = a + h
    fb = f(b)

    if fb > fa:
        a, b = b, a
        fa, fb = fb, fa
        h = -h

    for _ in range(max_iter):
        x_next = b + h
        f_next = f(x_next)

        if f_next > fb:
            return min(a, x_next), max(a, x_next)

        a, fa = b, fb
        b, fb = x_next, f_next
        h *= c

    raise RuntimeError("Maximum number of iterations reached")


def central_difference(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    """Approximate f'(x) with a centered finite difference."""
    return (f(x + h) - f(x - h)) / (2 * h)


def expand_interval(
    f: Callable[[float], float],
    a: float,
    b: float,
    h: float = 1e-5,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Expand [a,b] until f'(a) < 0 and f'(b) > 0."""
    for _ in range(max_iter):
        if central_difference(f, a, h) < 0 and central_difference(f, b, h) > 0:
            return a, b

        w = abs((b - a) / 2)
        a -= w
        b += w

    raise RuntimeError("Could not find a suitable interval.")


def bisection_minimum(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsilon: float = 1e-8,
    h: float = 1e-5,
    max_iter: int = 1000,
) -> tuple[float, float]:
    """Approximate a local minimum by applying bisection to f'."""
    if central_difference(f, a, h) >= 0 or central_difference(f, b, h) <= 0:
        raise ValueError("Need f'(a) < 0 and f'(b) > 0.")

    for _ in range(max_iter):
        m = (a + b) / 2
        dm = central_difference(f, m, h)

        if abs(dm) < 1e-12:
            return m, m

        if dm > 0:
            b = m
        else:
            a = m

        if abs(b - a) < epsilon:
            return a, b

    raise RuntimeError("Maximum number of iterations reached.")
