"""Interpolation: Lagrange (barycentric), Newton divided differences,
piecewise linear, and natural cubic splines.

The spline system is solved with the hand-written Gaussian elimination
from linalg.py. The Runge demo at the bottom shows why high-degree
polynomials on equally spaced nodes are a trap.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from calccode import linalg


def barycentric_weights(xs: np.ndarray) -> np.ndarray:
    """w_j = 1 / prod_{k != j} (x_j - x_k)."""
    xs = np.asarray(xs, dtype=float)
    n = xs.size
    w = np.empty(n)
    for j in range(n):
        prod = 1.0
        for k in range(n):
            if k != j:
                prod *= xs[j] - xs[k]
        w[j] = 1.0 / prod
    return w


def lagrange_eval(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """Barycentric form of the interpolating polynomial at x.

    Stable and O(n) per evaluation once the weights exist. Returns the
    node value exactly when x lands on a node.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    w = barycentric_weights(xs)
    num, den = 0.0, 0.0
    for j in range(xs.size):
        d = x - xs[j]
        if d == 0.0:
            return float(ys[j])
        term = w[j] / d
        num += term * ys[j]
        den += term
    return num / den


def divided_difference_coeffs(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Newton form coefficients by the divided difference table."""
    xs = np.asarray(xs, dtype=float)
    coef = np.asarray(ys, dtype=float).copy()
    n = xs.size
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j])
    return coef


def newton_eval(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """Evaluate the Newton form with nested multiplication."""
    xs = np.asarray(xs, dtype=float)
    coef = divided_difference_coeffs(xs, ys)
    total = coef[-1]
    for i in range(xs.size - 2, -1, -1):
        total = total * (x - xs[i]) + coef[i]
    return float(total)


def piecewise_linear(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """Linear interpolation on the interval containing x, clamped at the ends."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    i = int(np.searchsorted(xs, x) - 1)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return float(ys[i] * (1.0 - t) + ys[i + 1] * t)


def natural_cubic_spline(xs: np.ndarray, ys: np.ndarray) -> Callable[[float], float]:
    """Natural cubic spline: C2 through the nodes, zero curvature at the ends.

    Builds the tridiagonal system for the second derivatives M_i and
    solves it with linalg.solve. No numpy.linalg.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = xs.size - 1  # number of intervals
    h = np.diff(xs)

    # Rows 0 and n pin M_0 = M_n = 0 (natural boundary).
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    A[0, 0] = 1.0
    A[n, n] = 1.0
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    M = linalg.solve(A, rhs)

    def spline(x: float) -> float:
        if x <= xs[0]:
            i = 0
        elif x >= xs[-1]:
            i = n - 1
        else:
            i = int(np.searchsorted(xs, x) - 1)
            i = min(i, n - 1)
        hi = xs[i + 1] - xs[i]
        a = (xs[i + 1] - x) / hi
        b = (x - xs[i]) / hi
        return float(
            a * ys[i]
            + b * ys[i + 1]
            + ((a**3 - a) * M[i] + (b**3 - b) * M[i + 1]) * hi * hi / 6.0
        )

    return spline


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Chebyshev points on [a, b], clustered near the ends."""
    j = np.arange(n)
    return (a + b) / 2.0 + (b - a) / 2.0 * np.cos((2 * j + 1) * math.pi / (2 * n))


def runge_comparison(n: int = 15) -> dict[str, float]:
    """Max error of equispaced vs Chebyshev interpolation of 1 / (1 + 25 x^2).

    The equispaced interpolant oscillates near the ends and its error
    grows with n. Chebyshev nodes keep the same polynomial degree tame.
    """
    f = lambda x: 1.0 / (1.0 + 25.0 * x * x)
    grid = np.linspace(-1.0, 1.0, 2001)
    exact = np.array([f(v) for v in grid])

    xs_eq = np.linspace(-1.0, 1.0, n)
    ys_eq = np.array([f(v) for v in xs_eq])
    err_eq = max(abs(lagrange_eval(xs_eq, ys_eq, v) - exact[i]) for i, v in enumerate(grid))

    xs_ch = chebyshev_nodes(n)
    ys_ch = np.array([f(v) for v in xs_ch])
    err_ch = max(abs(lagrange_eval(xs_ch, ys_ch, v) - exact[i]) for i, v in enumerate(grid))

    return {"equispaced_err": float(err_eq), "chebyshev_err": float(err_ch)}
