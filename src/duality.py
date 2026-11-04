"""CS164 Session 17 — Duality, reconstructed from the submitted PCW.

Companion notebook: notebooks/17_duality.ipynb

Q1: Plot the concrete-mixture primal and dual feasible regions.
Q2: Solve the two LPs independently using CVXPY.
Q3: Plot L(x, mu) for mu = 1, 10, 100.
Q4: Plot a finite-step approximation to the infinite penalty.

The 10-page PDF ends at the opening of Q5; no unseen Q5/Q6 solution
is attributed to the submission.

Run from repository root:
    python src/duality.py
    python src/duality.py --save-dir session17_figures
    python src/duality.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Composition rows: cement, gravel, sand; columns: types 1 and 2.
A = np.array([[0.3, 0.1],
              [0.4, 0.2],
              [0.3, 0.7]], dtype=float)
B = np.array([5.0, 3.0, 4.0], dtype=float)
COST = np.array([5.0, 1.0], dtype=float)


def solve_concrete_primal_dual():
    """Solve both LPs independently; do not use CVXPY dual_value."""
    x = cp.Variable(2, nonneg=True)
    primal = cp.Problem(cp.Minimize(COST @ x), [A @ x >= B])
    primal.solve()

    mu = cp.Variable(3, nonneg=True)
    dual = cp.Problem(cp.Maximize(B @ mu), [A.T @ mu <= COST])
    dual.solve()

    good = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if primal.status not in good or dual.status not in good:
        raise RuntimeError(f"Unexpected statuses: {primal.status}, {dual.status}")

    x_star = np.asarray(x.value, dtype=float).reshape(2)
    mu_star = np.asarray(mu.value, dtype=float).reshape(3)
    p_star = float(primal.value)
    d_star = float(dual.value)
    gap = p_star - d_star

    # Expected from the submitted PCW: p*=d*=50, x*=(0,50), mu*=(10,0,0).
    if not np.allclose(x_star, [0, 50], atol=1e-3):
        raise AssertionError(f"Unexpected primal optimizer: {x_star}")
    if not np.allclose(mu_star, [10, 0, 0], atol=1e-3):
        raise AssertionError(f"Unexpected dual optimizer: {mu_star}")
    if not (np.isclose(p_star, 50, atol=1e-3)
            and np.isclose(d_star, 50, atol=1e-3)):
        raise AssertionError(f"Unexpected optimal values: {p_star}, {d_star}")
    if not np.all(A @ x_star >= B - 1e-4):
        raise AssertionError("Primal infeasibility")
    if not np.all(A.T @ mu_star <= COST + 1e-4):
        raise AssertionError("Dual infeasibility")
    if abs(gap) > 1e-3:
        raise AssertionError(f"Unexpected duality gap: {gap}")

    return {
        "primal_status": primal.status,
        "dual_status": dual.status,
        "x": x_star,
        "mu": mu_star,
        "primal_value": p_star,
        "dual_value": d_star,
        "gap": gap,
    }


def plot_concrete_feasible_sets():
    """Q1: 2D primal feasible set and 3D dual feasible polytope."""
    fig = plt.figure(figsize=(14, 5.5))
    ax1 = fig.add_subplot(121)
    x1 = np.linspace(0, 20, 500)
    cement = (B[0] - A[0, 0] * x1) / A[0, 1]
    gravel = (B[1] - A[1, 0] * x1) / A[1, 1]
    sand = (B[2] - A[2, 0] * x1) / A[2, 1]
    lower = np.maximum.reduce([cement, gravel, sand, np.zeros_like(x1)])

    ax1.fill_between(x1, lower, 60, where=lower <= 60,
                     color="tab:green", alpha=0.2, label="Feasible region")
    ax1.plot(x1, cement, label="Cement constraint")
    ax1.plot(x1, gravel, label="Gravel constraint")
    ax1.plot(x1, sand, label="Sand constraint")
    ax1.scatter([0], [50], color="red", s=75, zorder=5)
    ax1.annotate("Optimal (0, 50)\nCost = $50", (0, 50),
                 xytext=(3, 53), arrowprops={"arrowstyle": "->"})
    ax1.set(xlim=(0, 20), ylim=(0, 60), xlabel="x₁ (Type 1, lb)",
            ylabel="x₂ (Type 2, lb)", title="Primal feasible set")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(122, projection="3d")
    # In the nonnegative orthant, the type-1 dual inequality is redundant.
    # The remaining inequality is 0.1*mu1 + 0.2*mu2 + 0.7*mu3 <= 1.
    vertices = np.array([
        [0, 0, 0], [10, 0, 0], [0, 5, 0], [0, 0, 1 / 0.7],
    ], dtype=float)
    faces = [
        vertices[list(ids)] for ids in
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    ]
    poly = Poly3DCollection(faces, alpha=0.23, facecolor="tab:blue",
                            edgecolor="black", linewidth=0.8)
    ax2.add_collection3d(poly)
    ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                color="tab:blue", s=28)
    ax2.scatter([10], [0], [0], color="red", s=80, depthshade=False)
    ax2.text(10, 0, 0.3, "Optimal (10, 0, 0)", color="red")
    ax2.set(xlim=(0, 11), ylim=(0, 6), zlim=(0, 2),
            xlabel="μ₁", ylabel="μ₂", zlabel="μ₃",
            title="Dual feasible set")
    ax2.view_init(elev=22, azim=35)
    fig.tight_layout()
    return fig


def lagrangian(x, mu):
    """Minimize x**2 subject to 1-x <= 0."""
    return x**2 + mu * (1 - x)


def plot_lagrangian_penalties(multipliers=(1, 10, 100)):
    """Q3: Compare the objective with its Lagrangian at three multipliers."""
    x = np.linspace(-1, 4, 500)
    fig, axes = plt.subplots(1, len(multipliers), figsize=(16, 5))
    axes = np.atleast_1d(axes)
    for ax, mu in zip(axes, multipliers):
        ax.axvspan(-1, 1, color="tab:red", alpha=0.07,
                   label="Infeasible (x < 1)")
        ax.axvspan(1, 4, color="tab:green", alpha=0.07,
                   label="Feasible (x ≥ 1)")
        ax.plot(x, x**2, "--", color="gray", linewidth=2, label="f(x) = x²")
        ax.plot(x, lagrangian(x, mu), linewidth=2, label=f"L(x, {mu})")
        ax.axvline(1, color="black", linestyle=":", label="x = 1")
        ax.scatter([1], [1], color="black", zorder=5)
        values = lagrangian(x, mu)
        ax.set(xlim=(-1, 4), ylim=(min(-5, values.min() - 5),
                                   max(20, values.max() + 5)),
               xlabel="x", ylabel="Function value",
               title=f"Lagrangian with μ = {mu}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def finite_step(x, penalty=20):
    """Finite-height illustration, not a literal infinite penalty."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 1, x**2, float(penalty))


def plot_finite_step(penalty=20):
    """Q4: Step penalty T(x)=20 for x<1, x**2 otherwise."""
    x = np.linspace(-4, 4, 1000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(-4, 1, color="tab:red", alpha=0.08,
               label="Infeasible region (x < 1)")
    ax.axvspan(1, 4, color="tab:green", alpha=0.08,
               label="Feasible region (x ≥ 1)")
    ax.plot(x, x**2, "--", color="gray", linewidth=2,
            label="Original f(x) = x²")
    ax.plot(x[x < 1], finite_step(x[x < 1], penalty),
            color="tab:red", linewidth=2.5, label=f"T(x) = {penalty} (x < 1)")
    ax.plot(x[x >= 1], finite_step(x[x >= 1], penalty),
            color="tab:green", linewidth=2.5, label="T(x) = x² (x ≥ 1)")
    ax.scatter([1], [penalty], facecolors="white", edgecolors="tab:red",
               s=80, linewidths=2, zorder=5)
    ax.scatter([1], [1], color="tab:green", s=75, zorder=6,
               label="Optimal (1, 1)")
    ax.axvline(1, linestyle=":", color="black", alpha=0.7)
    ax.set(xlim=(-4, 4), ylim=(-1, penalty + 3), xlabel="x",
           ylabel="Function value",
           title="Finite step approximation of the infinite penalty")
    ax.legend(loc="center left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Directory in which to save three PNG plots.")
    parser.add_argument("--show", action="store_true",
                        help="Display interactive plots.")
    args = parser.parse_args(argv)

    results = solve_concrete_primal_dual()
    print("SESSION 17 — CONCRETE MIXTURE LP")
    print("Primal:", results["primal_status"])
    print("Concrete quantities (lb):", np.round(results["x"], 6))
    print("Minimum cost ($):", round(results["primal_value"], 6))
    print("Dual:", results["dual_status"])
    print("Multipliers:", np.round(results["mu"], 6))
    print("Maximum dual value ($):", round(results["dual_value"], 6))
    print("Duality gap:", round(results["gap"], 8))

    if args.save_dir is not None or args.show:
        figures = {
            "concrete_feasible_sets.png": plot_concrete_feasible_sets(),
            "lagrangian_penalties.png": plot_lagrangian_penalties(),
            "finite_step_penalty.png": plot_finite_step(),
        }
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            for filename, fig in figures.items():
                fig.savefig(args.save_dir / filename, dpi=160, bbox_inches="tight")
                print("Saved:", args.save_dir / filename)
        if args.show:
            plt.show()
        else:
            for fig in figures.values():
                plt.close(fig)


if __name__ == "__main__":
    main()
