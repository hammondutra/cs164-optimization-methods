"""CS164 Session 2: Taylor series and numerical approximation.

Portable NumPy/Matplotlib translations of the visible SageMath plots.
The source screenshot includes univariate approximations of sin(x) at pi and 1,
plus multivariate linear/quadratic approximations of x**3+y and sin(x)+cos(y).
"""
import math
import numpy as np
import matplotlib.pyplot as plt


def sin_taylor(x, x0, order):
    """Taylor polynomial of sin(x) at x0 through the given order."""
    x = np.asarray(x)
    return sum(math.sin(x0 + k*math.pi/2) / math.factorial(k) * (x-x0)**k
               for k in range(order+1))


def plot_univariate(x0):
    x = np.linspace(x0-2, x0+2, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, np.sin(x), label=r'$f(x)=\sin x$', linewidth=2.5)
    for order in (1, 2, 3):
        ax.plot(x, sin_taylor(x, x0, order), linestyle='--',
                label=f'Order {order} Taylor')
    ax.set(xlabel='x', ylabel='y', title=f'Taylor approximations at x0={x0:.5g}')
    ax.legend(); ax.grid(True)
    return fig


def cubic_original(x, y):
    return x**3 + y


def cubic_linear(x, y):
    return y


def cubic_quadratic(x, y):
    return y


def trig_original(x, y):
    return np.sin(x) + np.cos(y)


def trig_linear(x, y, x0, y0):
    return (np.sin(x0) + np.cos(y0)
            + np.cos(x0)*(x-x0) - np.sin(y0)*(y-y0))


def trig_quadratic(x, y, x0, y0):
    return trig_linear(x, y, x0, y0) - 0.5*np.sin(x0)*(x-x0)**2 - 0.5*np.cos(y0)*(y-y0)**2


def plot_surfaces(original, linear, quadratic, xlim, ylim, name):
    x = np.linspace(*xlim, 55)
    y = np.linspace(*ylim, 55)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection='3d')
    for f, label, color in ((original, 'Original', 'C0'), (linear, 'Linear', 'C1'),
                            (quadratic, 'Quadratic', 'C2')):
        ax.plot_surface(X, Y, f(X, Y), color=color, alpha=.42, linewidth=0, label=label)
    ax.set(xlabel='x', ylabel='y', zlabel='f(x,y)', title=name)
    # Artist-based legend is more portable than surface proxy legend across versions.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=color, label=label, alpha=.5)
                       for label, color in [('Original','C0'),('Linear','C1'),('Quadratic','C2')]])
    return fig


def main():
    plot_univariate(np.pi)
    plot_univariate(1.0)
    plot_surfaces(cubic_original, cubic_linear, cubic_quadratic,
                  (-1, 1), (-1, 1), r'$x^3+y$ near $(0,0)$')
    for x0, y0 in ((0.0, 0.0), (np.pi/2, np.pi/2)):
        plot_surfaces(trig_original,
                      lambda x,y,x0=x0,y0=y0: trig_linear(x,y,x0,y0),
                      lambda x,y,x0=x0,y0=y0: trig_quadratic(x,y,x0,y0),
                      (x0-2,x0+2), (y0-2,y0+2),
                      f'sin(x)+cos(y) near ({x0:.3f},{y0:.3f})')
    plt.show()

if __name__ == '__main__':
    main()
