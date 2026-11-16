"""CS164 Session 20 — Quadratic Programming (PCW Question 4).

Reconstructed from the five-page submitted Session 20 pre-class work.
Companion notebook: notebooks/20_quadratic_programming.ipynb

Run from the repository root:
    python src/quadratic_programming.py
    python src/quadratic_programming.py --save-dir session20_figures
    python src/quadratic_programming.py --show

The numerical solver uses CVXPY; GLPK and CVXOPT are not needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 0.5 * x.T @ Q @ x + c.T @ x + d
Q = np.array([[6.0, 2.0], [2.0, 2.0]])
C = np.array([1.0, 6.0])
D = 2.0

# A @ x <= b is equivalent to 2*x1 + 3*x2 >= 4, x1 >= 0, x2 >= 0.
A = np.array([[-2.0, -3.0], [-1.0, 0.0], [0.0, -1.0]])
B = np.array([-4.0, 0.0, 0.0])


def objective(x1, x2):
    """Evaluate f(x1, x2), including on NumPy meshgrids."""
    return 3 * x1**2 + x2**2 + 2 * x1 * x2 + x1 + 6 * x2 + D


def solve_quadratic_program():
    """Solve the strictly convex QP and verify the submitted result."""
    import cvxpy as cp

    x = cp.Variable(2)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, Q) + C @ x + D),
        [A @ x <= B],
    )
    problem.solve()

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Solver did not find an optimum: {problem.status}")

    solution = np.asarray(x.value, dtype=float).reshape(2)
    value = float(problem.value)
    slack = B - A @ solution

    if not np.all(slack >= -1e-5):
        raise AssertionError(f"Infeasible computed solution: {solution}")
    if not np.allclose(solution, [0.5, 1.0], atol=1e-4):
        raise AssertionError(f"Unexpected minimizer: {solution}")
    if not np.isclose(value, 11.25, atol=1e-4):
        raise AssertionError(f"Unexpected minimum value: {value}")

    return {
        "status": problem.status,
        "x": solution,
        "value": value,
        "slack": slack,
        "active_constraints": np.isclose(slack, 0.0, atol=1e-4),
    }


def plot_problem(optimum=None):
    """Plot contours, feasible region, and optionally the optimal point."""
    x1 = np.linspace(0, 2.5, 400)
    x2 = np.linspace(0, 2.5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = objective(X1, X2)

    # Feasibility includes the nonnegative orthant (the plot starts at 0).
    feasible = (2 * X1 + 3 * X2 >= 4) & (X1 >= 0) & (X2 >= 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    contours = ax.contour(X1, X2, Z, levels=15, cmap="viridis")
    ax.clabel(contours, inline=True, fontsize=8)
    ax.contourf(X1, X2, feasible.astype(float), levels=[0.5, 1.5],
                colors=["lightgreen"], alpha=0.35)

    t = np.linspace(0, 2, 200)
    ax.plot(t, (4 - 2 * t) / 3, "k--", label="2x₁ + 3x₂ = 4")

    if optimum is not None:
        point = np.asarray(optimum, dtype=float).reshape(2)
        ax.scatter(point[0], point[1], color="red", s=100, zorder=5,
                   label=f"Optimum ({point[0]:.2f}, {point[1]:.2f})")

    ax.set(xlim=(0, 2.5), ylim=(0, 2.5), xlabel="x₁", ylabel="x₂",
           title="Quadratic program: contours and feasible region")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Directory for the contour and solution PNG figures.")
    parser.add_argument("--show", action="store_true",
                        help="Display the figures interactively.")
    args = parser.parse_args(argv)

    result = solve_quadratic_program()
    print("SESSION 20 — QUADRATIC PROGRAMMING")
    print("Status:", result["status"])
    print("Optimal solution [x1, x2]:", np.round(result["x"], 6))
    print("Minimum objective value:", round(result["value"], 6))
    print("Slack b - Ax:", np.round(result["slack"], 6))
    print("Active constraints:", result["active_constraints"])

    if args.save_dir is not None or args.show:
        figures = {
            "qp_feasible_region.png": plot_problem(),
            "qp_optimal_solution.png": plot_problem(result["x"]),
        }
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                dest = args.save_dir / name
                fig.savefig(dest, dpi=160, bbox_inches="tight")
                print("Saved:", dest)
        if args.show:
            plt.show()
        else:
            for fig in figures.values():
                plt.close(fig)


if __name__ == "__main__":
    main()
