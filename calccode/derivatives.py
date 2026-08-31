"""Finite difference derivatives, written by hand.

The interesting question is not how to subtract two function values but
how the error depends on the step h. ``convergence_study`` measures that
dependence and ``fit_order`` reads the order off a log-log slope.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULT_H = 1e-5


def forward_diff(f: Callable[[float], float], x: float, h: float = DEFAULT_H) -> float:
    """First order estimate: error is O(h)."""
    return (f(x + h) - f(x)) / h


def backward_diff(f: Callable[[float], float], x: float, h: float = DEFAULT_H) -> float:
    """First order estimate, mirrored to the left of x."""
    return (f(x) - f(x - h)) / h


def central_diff(f: Callable[[float], float], x: float, h: float = DEFAULT_H) -> float:
    """Second order estimate: the O(h) terms cancel, error is O(h^2)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def second_derivative(f: Callable[[float], float], x: float, h: float = 1e-4) -> float:
    """Central second difference, error O(h^2)."""
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def convergence_study(
    f: Callable[[float], float],
    x: float,
    exact: float,
    hs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Measure forward and central difference errors over a range of h."""
    if hs is None:
        hs = np.logspace(-1, -8, 15)
    hs = np.asarray(hs, dtype=float)
    fwd = np.array([abs(forward_diff(f, x, h) - exact) for h in hs])
    cen = np.array([abs(central_diff(f, x, h) - exact) for h in hs])
    return {"hs": hs, "forward_err": fwd, "central_err": cen}


def fit_order(hs: np.ndarray, errors: np.ndarray) -> float:
    """Slope of log(error) vs log(h): the empirical order of accuracy.

    Zero-error rows are dropped because log(0) is undefined and they
    carry no information about the trend.
    """
    hs = np.asarray(hs, dtype=float)
    errors = np.asarray(errors, dtype=float)
    mask = errors > 0.0
    slope, _ = np.polyfit(np.log(hs[mask]), np.log(errors[mask]), 1)
    return float(slope)
