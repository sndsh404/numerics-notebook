"""Regenerate the Calc II and ML plots into plots/ and docs/img/.

Run from the project root:

    python scripts/make_plots_2.py

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

from calccode import fourier, interpolation, kinematics, montecarlo, ode_systems, probability, regression, transforms
from calccode.integrals import convergence_study
from calccode.series import taylor_coefficients_from_expr, taylor_error, taylor_polynomial
from calccode.symbolic import Sin, Var

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIRS = [os.path.join(ROOT, "plots"), os.path.join(ROOT, "docs", "img")]


def save(fig: plt.Figure, name: str) -> None:
    for out_dir in OUT_DIRS:
        fig.savefig(os.path.join(out_dir, name), dpi=120)
    plt.close(fig)


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
    save(fig, "quadrature_convergence.png")


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
    save(fig, "taylor_sin.png")


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
    save(fig, "arm_poses.png")


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
    save(fig, "least_squares.png")


def plot_pendulum_phase() -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for theta0 in (0.3, 1.0, 2.4):
        ts, ys = ode_systems.rk4_system(
            ode_systems.pendulum_rhs(), 0.0, np.array([theta0, 0.0]), h=0.002, n_steps=6000
        )
        ax.plot(ys[:, 0], ys[:, 1], lw=1, label=f"theta0 = {theta0}")
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")
    ax.set_title("Nonlinear pendulum phase portrait")
    ax.legend()
    fig.tight_layout()
    save(fig, "pendulum_phase.png")


def plot_lorenz_divergence() -> None:
    f = ode_systems.lorenz_rhs()
    y0 = np.array([1.0, 1.0, 1.0])
    ts, a = ode_systems.rk4_system(f, 0.0, y0, h=0.005, n_steps=6000)
    _, b = ode_systems.rk4_system(f, 0.0, y0 + np.array([1e-6, 0.0, 0.0]), h=0.005, n_steps=6000)
    sep = np.linalg.norm(a - b, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(a[:, 0], a[:, 2], lw=0.4)
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.set_title("Lorenz attractor")
    ax2.semilogy(ts, sep)
    ax2.set_xlabel("t")
    ax2.set_ylabel("separation (log scale)")
    ax2.set_title("Two starts 1e-6 apart diverge")
    fig.tight_layout()
    save(fig, "lorenz_divergence.png")


def plot_fourier_spectrum() -> None:
    n = 512
    sample_rate = 512.0
    t = np.arange(n) / sample_rate
    signal = 1.5 * np.sin(2 * math.pi * 3 * t) + 0.7 * np.sin(2 * math.pi * 7 * t)
    amps = fourier.amplitude_spectrum(signal)
    freqs = np.arange(amps.size) * sample_rate / n

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t[:128], signal[:128])
    ax1.set_xlabel("t (s)")
    ax1.set_title("Signal: 1.5 sin(2 pi 3 t) + 0.7 sin(2 pi 7 t)")
    ax2.stem(freqs[:20], amps[:20])
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("amplitude")
    ax2.set_title("One-sided DFT spectrum")
    fig.tight_layout()
    save(fig, "fourier_spectrum.png")


def plot_runge_phenomenon() -> None:
    f = lambda x: 1.0 / (1.0 + 25.0 * x * x)
    n = 15
    grid = np.linspace(-1.0, 1.0, 1000)
    exact = np.array([f(float(v)) for v in grid])

    xs_eq = np.linspace(-1.0, 1.0, n)
    ys_eq = np.array([f(float(v)) for v in xs_eq])
    p_eq = np.array([interpolation.lagrange_eval(xs_eq, ys_eq, float(v)) for v in grid])

    xs_ch = interpolation.chebyshev_nodes(n)
    ys_ch = np.array([f(float(v)) for v in xs_ch])
    p_ch = np.array([interpolation.lagrange_eval(xs_ch, ys_ch, float(v)) for v in grid])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, exact, "k-", lw=2, label="Runge function 1 / (1 + 25 x^2)")
    ax.plot(grid, p_eq, "--", color="tab:red", label=f"equispaced nodes, n = {n}")
    ax.plot(grid, p_ch, "-", color="tab:blue", label=f"Chebyshev nodes, n = {n}")
    ax.plot(xs_eq, ys_eq, "o", color="tab:red", markersize=4)
    ax.plot(xs_ch, ys_ch, "s", color="tab:blue", markersize=4)
    ax.set_ylim(-1.5, 2.0)
    ax.set_xlabel("x")
    ax.set_title("Runge phenomenon: node placement beats node count")
    ax.legend()
    fig.tight_layout()
    save(fig, "runge_phenomenon.png")


def plot_monte_carlo_scaling() -> None:
    ns = np.array([100, 300, 1000, 3000, 10000, 30000, 100000])
    ns_out, stds = montecarlo.pi_error_scaling(ns, n_seeds=25)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ns_out, stds, "o-", label="std of pi estimates (25 seeds)")
    ax.loglog(ns_out, 1.7 / np.sqrt(ns_out), "--", color="gray", label="reference 1/sqrt(n)")
    ax.set_xlabel("samples n")
    ax.set_ylabel("standard deviation")
    ax.set_title("Monte Carlo error scaling")
    ax.legend()
    fig.tight_layout()
    save(fig, "monte_carlo_scaling.png")


def plot_clt_demo() -> None:
    lam = 1.0
    n = 30
    data = probability.clt_demo(lam=lam, sample_size=n, n_means=4000, seed=7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pop_grid = np.linspace(0.0, 6.0, 200)
    pop_pdf = np.array([probability.exponential_pdf(float(x), lam) for x in pop_grid])
    ax1.hist(data["population"], bins=80, range=(0.0, 6.0), density=True, color="tab:blue", alpha=0.7)
    ax1.plot(pop_grid, pop_pdf, "k-", lw=2, label=f"exponential pdf, lam = {lam}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("density")
    ax1.set_title("Population: exponential draws")
    ax1.legend()

    ax2.hist(data["means"], bins=60, density=True, color="tab:orange", alpha=0.7)
    ax2.plot(
        data["grid"],
        data["normal_curve"],
        "k-",
        lw=2,
        label=f"normal mu = {data['mu']:.2f}, sigma = {data['sigma']:.3f}",
    )
    ax2.set_xlabel("sample mean")
    ax2.set_title(f"Means of {n} draws approach a normal")
    ax2.legend()

    fig.tight_layout()
    save(fig, "clt_demo.png")


def plot_time_scaling() -> None:
    start = np.array([0.0])
    final = np.array([1.5])
    Tf = 2.0
    cubic = kinematics.joint_trajectory(start, final, Tf, 200, method="cubic")
    quintic = kinematics.joint_trajectory(start, final, Tf, 200, method="quintic")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in (
        (ax1, "positions", "position theta(t)"),
        (ax2, "velocities", "velocity"),
        (ax3, "accelerations", "acceleration"),
    ):
        ax.plot(cubic["times"], cubic[key][:, 0], "--", color="tab:red", label="cubic")
        ax.plot(quintic["times"], quintic[key][:, 0], "-", color="tab:blue", label="quintic")
        ax.set_xlabel("t (s)")
        ax.set_title(title)
    ax1.legend()
    fig.suptitle("Joint move of 1.5 rad in 2 s: cubic vs quintic time scaling")
    fig.tight_layout()
    save(fig, "time_scaling.png")


def main() -> None:
    for out_dir in OUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
    plot_quadrature_convergence()
    print("wrote quadrature_convergence.png")
    plot_taylor_sin()
    print("wrote taylor_sin.png")
    plot_arm()
    print("wrote arm_poses.png")
    plot_least_squares()
    print("wrote least_squares.png")
    plot_pendulum_phase()
    print("wrote pendulum_phase.png")
    plot_lorenz_divergence()
    print("wrote lorenz_divergence.png")
    plot_fourier_spectrum()
    print("wrote fourier_spectrum.png")
    plot_runge_phenomenon()
    print("wrote runge_phenomenon.png")
    plot_monte_carlo_scaling()
    print("wrote monte_carlo_scaling.png")
    plot_clt_demo()
    print("wrote clt_demo.png")
    plot_time_scaling()
    print("wrote time_scaling.png")


if __name__ == "__main__":
    main()
