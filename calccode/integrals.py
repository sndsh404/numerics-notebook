"""Numerical integration: Riemann sums, trapezoid, and Simpson's rule.

Every rule is a hand-written loop over panel evaluations. The
convergence study measures error against interval count and reuses the
log-log order fit from derivatives.py: trapezoid should come out order
2, Simpson order 4.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from calccode.derivatives import fit_order


def left_sum(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Left endpoint rule. First order."""
    h = (b - a) / n
    return h * sum(f(a + i * h) for i in range(n))


def right_sum(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Right endpoint rule. First order."""
    h = (b - a) / n
    return h * sum(f(a + (i + 1) * h) for i in range(n))


def midpoint_sum(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Midpoint rule. Second order, and it never evaluates the endpoints."""
    h = (b - a) / n
    return h * sum(f(a + (i + 0.5) * h) for i in range(n))


def trapezoid(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Trapezoid rule. Second order."""
    h = (b - a) / n
    interior = sum(f(a + i * h) for i in range(1, n))
    return h * (0.5 * f(a) + interior + 0.5 * f(b))


def simpson(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Simpson's rule with n panels, n must be even. Fourth order."""
    if n % 2 != 0:
        raise ValueError("simpson needs an even number of panels")
    h = (b - a) / n
    odd = sum(f(a + i * h) for i in range(1, n, 2))
    even = sum(f(a + i * h) for i in range(2, n, 2))
    return h / 3.0 * (f(a) + 4.0 * odd + 2.0 * even + f(b))


def convergence_study(
    f: Callable[[float], float],
    a: float,
    b: float,
    exact: float,
    ns: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Absolute error of each rule over a range of panel counts."""
    if ns is None:
        ns = np.array([8, 16, 32, 64, 128, 256, 512])
    ns = np.asarray(ns, dtype=int)
    ns_even = ns + (ns % 2)  # simpson needs even counts
    return {
        "ns": ns,
        "left_err": np.array([abs(left_sum(f, a, b, n) - exact) for n in ns]),
        "midpoint_err": np.array([abs(midpoint_sum(f, a, b, n) - exact) for n in ns]),
        "trapezoid_err": np.array([abs(trapezoid(f, a, b, n) - exact) for n in ns]),
        "simpson_err": np.array([abs(simpson(f, a, b, n) - exact) for n in ns_even]),
    }


def orders(study: dict[str, np.ndarray], a: float, b: float) -> dict[str, float]:
    """Empirical order of each rule, fit from the log-log error slope."""
    hs = (b - a) / study["ns"].astype(float)
    return {
        name: fit_order(hs, errs)
        for name, errs in study.items()
        if name != "ns"
    }


def integrate_inv_sqrt(n: int) -> dict[str, float]:
    """Demo: integral of 1/sqrt(x) over (0, 1], exact value 2.

    Naive midpoint handles the open endpoint but converges like
    1/sqrt(n) because the singularity dominates the error. The
    substitution x = t^2 turns the integrand into the constant 2,
    which midpoint integrates to machine precision on the first try.
    """
    exact = 2.0
    naive = midpoint_sum(lambda x: 1.0 / math.sqrt(x), 0.0, 1.0, n)

    def substituted(t: float) -> float:
        # x = t^2, dx = 2t dt, so the new integrand is (1/t) * 2t = 2.
        if t == 0.0:
            return 2.0
        return (1.0 / math.sqrt(t * t)) * 2.0 * t

    sub = midpoint_sum(substituted, 0.0, 1.0, n)
    return {"naive_err": abs(naive - exact), "substituted_err": abs(sub - exact)}
