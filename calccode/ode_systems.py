"""Coupled ODE systems integrated with the RK4 from integrators.py.

Three classic systems ship as examples. The nonlinear pendulum whose
period matches the small-angle theory only near zero. Lotka-Volterra
predator-prey, which conserves a known invariant along every orbit. The
Lorenz attractor, where two starts 1e-6 apart end up uncorrelated.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from calccode.integrators import rk4

RHS = Callable[[float, np.ndarray], np.ndarray]


def rk4_system(
    f: RHS, t0: float, y0: np.ndarray, h: float, n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """RK4 for y' = f(t, y) with vector y. Thin wrapper for readability."""
    return rk4(f, t0, np.asarray(y0, dtype=float), h, n_steps)


def pendulum_rhs(g: float = 9.81, length: float = 1.0) -> RHS:
    """Nonlinear pendulum. State is [theta, omega], theta from vertical."""

    def f(t: float, y: np.ndarray) -> np.ndarray:
        theta, omega = y
        return np.array([omega, -(g / length) * math.sin(theta)])

    return f


def pendulum_period_numerical(
    theta0: float, g: float = 9.81, length: float = 1.0, h: float = 5e-4, periods: int = 8
) -> float:
    """Measure the period by timing upward zero crossings of theta."""
    t_end = 4.0 * periods * 2.0 * math.pi * math.sqrt(length / g)
    n_steps = int(t_end / h)
    ts, ys = rk4_system(pendulum_rhs(g, length), 0.0, np.array([theta0, 0.0]), h, n_steps)
    thetas = ys[:, 0]
    crossings = []
    for i in range(1, len(ts)):
        if thetas[i - 1] < 0.0 <= thetas[i]:
            frac = -thetas[i - 1] / (thetas[i] - thetas[i - 1])
            crossings.append(ts[i - 1] + frac * h)
    if len(crossings) < 2:
        raise ValueError("not enough crossings; integrate longer")
    return float(np.mean(np.diff(crossings)))


def pendulum_period_small_angle(g: float = 9.81, length: float = 1.0) -> float:
    """Small-angle analytic period T = 2 pi sqrt(L / g)."""
    return 2.0 * math.pi * math.sqrt(length / g)


def lotka_volterra_rhs(
    alpha: float = 1.0, beta: float = 0.1, delta: float = 0.075, gamma: float = 1.5
) -> RHS:
    """Predator-prey. x is prey, y is predators."""

    def f(t: float, y: np.ndarray) -> np.ndarray:
        prey, pred = y
        return np.array(
            [alpha * prey - beta * prey * pred, delta * prey * pred - gamma * pred]
        )

    return f


def lotka_volterra_invariant(
    states: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.1,
    delta: float = 0.075,
    gamma: float = 1.5,
) -> np.ndarray:
    """H = delta x - gamma ln x + beta y - alpha ln y, constant on orbits."""
    prey, pred = states[:, 0], states[:, 1]
    return delta * prey - gamma * np.log(prey) + beta * pred - alpha * np.log(pred)


def lorenz_rhs(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> RHS:
    """The Lorenz system. State is [x, y, z]."""

    def f(t: float, y: np.ndarray) -> np.ndarray:
        x1, x2, x3 = y
        return np.array(
            [
                sigma * (x2 - x1),
                x1 * (rho - x3) - x2,
                x1 * x2 - beta * x3,
            ]
        )

    return f
