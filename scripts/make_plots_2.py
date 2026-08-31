"""Regenerate the Calc II and ML plots into plots/. Run from the project root:

    python scripts/make_plots_2.py
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calccode import regression, transforms
from calccode.integrals import convergence_study
from calccode.series import taylor_coefficients_from_expr, taylor_error, taylor_polynomial
from calccode.symbolic import Sin, Var

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")


def plot_quadrature_convergence() -> None:
    study = convergence_study(math.exp, 0.0, 1.0, math.e - 1.0)
    ns = study["ns"].astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ns, study["left_err"], "o-", label="left sum")
    ax.loglog(ns, study["trapezoid_err"], "s-", label="trapezoid")
    ax.loglog(ns, study["simpson_err"], "^-", label="simpson")
    ax.loglog(ns, 0.05 / ns, ":", color="gray", label="reference O(1/n)")
    ax.loglog(ns, 0.01 / ns**2, "--", color="gray", label="reference O(1/n^2)")
    ax.loglog(ns, 1e-3 / ns**4, "-.", color="gray", label="reference O(1/n^4)")
    ax.set_xlabel("panels n")
    ax.set_ylabel("absolute error")
    ax.set_title("Quadrature error on exp over [0, 1]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "quadrature_convergence.png"), dpi=120)
    plt.close(fig)


def plot_taylor_sin() -> None:
    xs = np.linspace(-2.0 * math.pi, 2.0 * math.pi, 400)
    degrees = [1, 3, 5, 11]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(xs, np.sin(xs), "k-", lw=2, label="sin(x)")
    for deg in degrees:
        coeffs = taylor_coefficients_from_expr(Sin(Var("x")), 0.0, deg)
        p = taylor_polynomial(coeffs, 0.0)
        ax1.plot(xs, [p(float(v)) for v in xs], "--", label=f"degree {deg}")
    ax1.set_ylim(-2.0, 2.0)
    ax1.set_title("Taylor approximations of sin around 0")
    ax1.legend()

    for deg in degrees:
        coeffs = taylor_coefficients_from_expr(Sin(Var("x")), 0.0, deg)
        p = taylor_polynomial(coeffs, 0.0)
        err = taylor_error(math.sin, p, xs)
        ax2.semilogy(xs, np.maximum(err, 1e-18), label=f"degree {deg}")
    ax2.set_title("Error grows away from the expansion point")
    ax2.set_xlabel("x")
    ax2.set_ylabel("|sin(x) - P(x)| (log scale)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "taylor_sin.png"), dpi=120)
    plt.close(fig)


def plot_arm() -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    configs = [(0.3, 0.6), (1.2, -0.8), (2.2, 0.9)]
    for t1, t2 in configs:
        joints = transforms.planar_arm_fk(t1, t2, l1=2.0, l2=1.0)
        ax.plot(joints[:, 0], joints[:, 1], "o-", lw=2, label=f"({t1}, {t2})")
        ax.plot(joints[-1, 0], joints[-1, 1], "s", markersize=8)
    ax.plot(0.0, 0.0, "k*", markersize=15, label="base")
    circle = plt.Circle((0, 0), 3.0, fill=False, linestyle=":", color="gray")
    ax.add_patch(circle)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_title("2-link planar arm poses (l1 = 2, l2 = 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "arm_poses.png"), dpi=120)
    plt.close(fig)


def plot_least_squares() -> None:
    rng = np.random.default_rng(7)
    X = rng.uniform(-2.0, 2.0, size=80)
    y = 2.5 * X - 1.0 + 0.3 * rng.normal(size=80)

    w_closed = regression.ols_closed_form(X, y)
    w_gd, history = regression.ols_gradient_descent(X, y, lr=0.5, n_iter=3000)

    grid = np.linspace(-2.2, 2.2, 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(X, y, "o", alpha=0.6, label="data")
    ax1.plot(grid, 2.5 * grid - 1.0, "k-", label="true line")
    ax1.plot(grid, w_closed[0] + w_closed[1] * grid, "--", label="closed form")
    ax1.plot(grid, w_gd[0] + w_gd[1] * grid, ":", lw=2, label="gradient descent")
    ax1.set_title("Least squares fit")
    ax1.legend()

    ax2.semilogy(history)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("MSE (log scale)")
    ax2.set_title("Gradient descent convergence")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "least_squares.png"), dpi=120)
    plt.close(fig)


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_quadrature_convergence()
    print("wrote quadrature_convergence.png")
    plot_taylor_sin()
    print("wrote taylor_sin.png")
    plot_arm()
    print("wrote arm_poses.png")
    plot_least_squares()
    print("wrote least_squares.png")


if __name__ == "__main__":
    main()
