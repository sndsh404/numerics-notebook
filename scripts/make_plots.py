"""Regenerate every plot into plots/ and docs/img/. Run from the project root:

    python scripts/make_plots.py

plots/ is the gitignored scratch dir; docs/img/ holds the tracked copies
that README.md embeds.
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

from calccode.autograd import MLP, fit
from calccode.derivatives import convergence_study
from calccode.gradient import gradient_descent
from calccode.integrators import euler, oscillator_energy, rk4, semi_implicit_euler
from calccode.optimize import adam, bfgs, momentum_gd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIRS = [os.path.join(ROOT, "plots"), os.path.join(ROOT, "docs", "img")]


def save(fig: plt.Figure, name: str) -> None:
    for out_dir in OUT_DIRS:
        fig.savefig(os.path.join(out_dir, name), dpi=120)
    plt.close(fig)


def plot_fd_convergence() -> None:
    hs = np.logspace(-1, -10, 19)
    study = convergence_study(math.sin, 1.0, math.cos(1.0), hs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(hs, study["forward_err"], "o-", label="forward difference")
    ax.loglog(hs, study["central_err"], "s-", label="central difference")
    ax.loglog(hs, 0.3 * hs, ":", color="gray", label="reference O(h)")
    ax.loglog(hs, 0.1 * hs**2, "--", color="gray", label="reference O(h^2)")
    ax.set_xlabel("step h")
    ax.set_ylabel("absolute error at x = 1")
    ax.set_title("Finite difference error vs step size")
    ax.legend()
    fig.tight_layout()
    save(fig, "fd_convergence.png")


def plot_integrators() -> None:
    k, h, n = 2.0, 0.05, 20
    decay = lambda t, y: -k * y
    ts_e, ys_e = euler(decay, 0.0, np.array([1.0]), h, n)
    ts_r, ys_r = rk4(decay, 0.0, np.array([1.0]), h, n)
    exact = np.exp(-k * ts_e)

    osc = lambda t, y: np.array([y[1], -y[0]])
    ts_o, ys_exp = euler(osc, 0.0, np.array([1.0, 0.0]), 0.05, 10000)
    _, ys_si = semi_implicit_euler(osc, 0.0, np.array([1.0, 0.0]), 0.05, 10000)
    _, ys_rk = rk4(osc, 0.0, np.array([1.0, 0.0]), 0.05, 10000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ts_e, exact, "k-", label="exact e^(-2t)")
    ax1.plot(ts_e, ys_e[:, 0], "o--", label="explicit Euler")
    ax1.plot(ts_r, ys_r[:, 0], "s--", label="RK4")
    ax1.set_xlabel("t")
    ax1.set_title("dx/dt = -2x, h = 0.05")
    ax1.legend()

    ax2.plot(ts_o, oscillator_energy(ys_exp), label="explicit Euler")
    ax2.plot(ts_o, oscillator_energy(ys_si), label="semi-implicit Euler")
    ax2.plot(ts_o, oscillator_energy(ys_rk), label="RK4")
    ax2.set_yscale("log")
    ax2.set_xlabel("t")
    ax2.set_ylabel("energy (log scale)")
    ax2.set_title("Harmonic oscillator energy, 10000 steps")
    ax2.legend()

    fig.tight_layout()
    save(fig, "integrators.png")


def plot_gradient_descent() -> None:
    bowl = lambda v: (v[0] - 1.0) ** 2 + 3.0 * (v[1] + 2.0) ** 2
    start = np.array([5.0, 5.0])
    runs = [
        ("lr = 0.01 (too small)", gradient_descent(bowl, start, 0.01, 100)),
        ("lr = 0.1 (good)", gradient_descent(bowl, start, 0.1, 100)),
        ("lr = 0.4 (diverging)", gradient_descent(bowl, start, 0.4, 30)),
    ]

    gx = np.linspace(-4.0, 6.0, 120)
    gy = np.linspace(-6.0, 6.0, 120)
    xx, yy = np.meshgrid(gx, gy)
    zz = (xx - 1.0) ** 2 + 3.0 * (yy + 2.0) ** 2

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contour(xx, yy, zz, levels=np.logspace(-1, 2.5, 16), cmap="viridis")
    for label, path in runs:
        clipped = np.clip(path, [-4.0, -6.0], [6.0, 6.0])
        ax.plot(clipped[:, 0], clipped[:, 1], "o-", markersize=3, label=label)
    ax.plot(1.0, -2.0, "r*", markersize=14, label="minimum (1, -2)")
    ax.set_title("Gradient descent paths on a 2D bowl")
    ax.legend()
    fig.tight_layout()
    save(fig, "gradient_descent.png")


def plot_optimizer_paths() -> None:
    rosen = lambda v: (1.0 - v[0]) ** 2 + 100.0 * (v[1] - v[0] ** 2) ** 2
    start = np.array([-1.2, 1.0])
    paths = [
        ("plain GD", gradient_descent(rosen, start, 0.001, 12000)),
        ("momentum", momentum_gd(rosen, start, lr=0.001, tol=0.0, max_iter=1200).history),
        ("Adam", adam(rosen, start, lr=0.1, tol=0.0, max_iter=1200).history),
        ("BFGS", bfgs(rosen, start).history),
    ]

    gx = np.linspace(-1.5, 1.6, 160)
    gy = np.linspace(-0.6, 1.8, 160)
    xx, yy = np.meshgrid(gx, gy)
    zz = (1.0 - xx) ** 2 + 100.0 * (yy - xx**2) ** 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, zz, levels=np.logspace(-1, 3.3, 20), cmap="viridis")
    for label, path in paths:
        ax.plot(path[:, 0], path[:, 1], ".-", markersize=2, lw=0.8, label=f"{label} ({len(path) - 1} steps)")
    ax.plot(1.0, 1.0, "r*", markersize=14, label="minimum (1, 1)")
    ax.plot(start[0], start[1], "ko", markersize=6)
    ax.set_title("Optimizer paths on the Rosenbrock valley")
    ax.legend()
    fig.tight_layout()
    save(fig, "optimizer_paths.png")


def plot_autograd_fit() -> None:
    xs = np.linspace(-math.pi, math.pi, 32)
    ys = np.sin(xs)
    mlp = MLP([1, 10, 10, 1], seed=3)
    history = fit(mlp, xs, ys, epochs=200, lr=0.05)

    grid = np.linspace(-math.pi, math.pi, 200)
    preds = np.array([mlp(float(g)).data for g in grid])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.semilogy(history)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("mean squared error (log scale)")
    ax1.set_title("MLP training loss")

    ax2.plot(grid, np.sin(grid), "k-", label="sin(x)")
    ax2.plot(grid, preds, "--", color="tab:orange", label="MLP after 200 epochs")
    ax2.plot(xs, ys, "o", color="tab:blue", markersize=4, label="training points")
    ax1_loss = history[-1]
    ax2.set_title(f"Fit to sin(x), final loss {ax1_loss:.2e}")
    ax2.legend()

    fig.tight_layout()
    save(fig, "autograd_fit.png")


def main() -> None:
    for out_dir in OUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
    plot_fd_convergence()
    print("wrote fd_convergence.png")
    plot_integrators()
    print("wrote integrators.png")
    plot_gradient_descent()
    print("wrote gradient_descent.png")
    plot_optimizer_paths()
    print("wrote optimizer_paths.png")
    plot_autograd_fit()
    print("wrote autograd_fit.png")


if __name__ == "__main__":
    main()
