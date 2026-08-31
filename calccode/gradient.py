"""Gradient descent on top of the hand-written central difference.

The gradient itself comes from derivatives.py, one coordinate at a time.
Every run returns its full history so scripts can plot the path and the
loss curve for different learning rates.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from calccode.derivatives import central_diff

ScalarFn = Callable[[np.ndarray], float]


def numerical_gradient(f: ScalarFn, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central difference gradient, one partial derivative per coordinate."""
    x = np.asarray(x, dtype=float)
    grad = np.empty_like(x)
    for i in range(x.size):
        partial = lambda xi: f(np.concatenate([x[:i], [xi], x[i + 1 :]]))  # noqa: B023
        grad[i] = central_diff(partial, float(x[i]), h)
    return grad


def gradient_descent(
    f: ScalarFn,
    x0: np.ndarray,
    lr: float,
    n_iter: int,
    h: float = 1e-6,
) -> np.ndarray:
    """Run n_iter steps and return the path, shape (n_iter + 1, dim).

    A learning rate above 2 / (largest curvature) diverges on a quadratic,
    which the tests and plots demonstrate on a known bowl.
    """
    x = np.asarray(x0, dtype=float).copy()
    history = np.empty((n_iter + 1, x.size))
    history[0] = x
    for i in range(1, n_iter + 1):
        x = x - lr * numerical_gradient(f, x, h)
        history[i] = x
    return history


def gradient_descent_1d(
    f: Callable[[float], float],
    x0: float,
    lr: float,
    n_iter: int,
    h: float = 1e-6,
) -> np.ndarray:
    """1D convenience wrapper around gradient_descent."""
    path = gradient_descent(lambda v: f(float(v[0])), np.array([x0]), lr, n_iter, h)
    return path[:, 0]
