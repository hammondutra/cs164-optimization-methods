"""CS164 Session 1: What and Why to Optimize — Python port of visible PCW plots.

The screenshot contains SageMath and Python plotting warmups. This module uses
NumPy and Matplotlib to reproduce the Python plots and toy functions.
The thumbnail example is a hypothetical *linear* model from the student's answer,
not measured YouTube data.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def thumbnail_views(brightness):
    """Student's illustrative model: V = 1_000_000 b, with 0 <= b <= 1."""
    b = np.asarray(brightness)
    if np.any((b < 0) | (b > 1)):
        raise ValueError('Thumbnail brightness must lie in [0, 1].')
    return 1_000_000 * b


def f_univariate(x):
    return (2 * x + 1) ** 2


def g_bivariate(x, y):
    return 2 * x - 3 * y + y**2


def plot_univariate():
    x = np.linspace(-2, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, f_univariate(x), label=r'$f(x)=(2x+1)^2$')
    ax.set(xlabel='x', ylabel='f(x)', title='Session 1: univariate plotting warmup')
    ax.legend(); ax.grid(True)
    return fig


def plot_contour():
    axis = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(axis, axis)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(X, Y, g_bivariate(X, Y), levels=20, cmap='Blues')
    fig.colorbar(cs, ax=ax, label='g(x,y)')
    ax.set(xlabel='x', ylabel='y', title=r'$g(x,y)=2x-3y+y^2$')
    return fig


def main():
    plot_univariate(); plot_contour(); plt.show()

if __name__ == '__main__':
    main()
