"""Multivariable calculus: partials, gradients, Jacobians, Hessians.

Everything is a central difference over one coordinate at a time,
reusing the machinery from derivatives.py and gradient.py. The gradient
check compares finite differences against exact partials from the
symbolic.py trees, which is the same idea as gradient checking a neural
net by hand.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from calccode.derivatives import central_diff
from calccode.gradient import numerical_gradient
from calccode.symbolic import Expr, eval_multi, partial

ScalarFn = Callable[[np.ndarray], float]
VectorFn = Callable[[np.ndarray], np.ndarray]


def partial_diff(f: ScalarFn, x: np.ndarray, i: int, h: float = 1e-5) -> float:
    """Partial derivative of f with respect to coordinate i at x."""
    x = np.asarray(x, dtype=float)
    return central_diff(lambda xi: f(np.concatenate([x[:i], [xi], x[i + 1 :]])), float(x[i]), h)


def gradient(f: ScalarFn, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Gradient vector, one central difference per coordinate."""
    return numerical_gradient(f, x, h)


def jacobian(F: VectorFn, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Jacobian of a vector-valued F, shape (n_outputs, n_inputs)."""
    x = np.asarray(x, dtype=float)
    n_out = np.asarray(F(x)).size
    J = np.empty((n_out, x.size))
    for i in range(x.size):
        col = central_diff(lambda xi: F(np.concatenate([x[:i], [xi], x[i + 1 :]])), float(x[i]), h)
        J[:, i] = np.asarray(col, dtype=float).ravel()
    return J


def hessian(f: ScalarFn, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Hessian of a scalar f: diagonal terms plus mixed partials.

    Mixed terms use the second central difference in two directions,
    error O(h^2). The result is symmetrized because d2f/dxdy = d2f/dydx
    and averaging removes asymmetric roundoff.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.empty((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h
        H[i, i] = (f(x + ei) - 2.0 * f(x) + f(x - ei)) / (h * h)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h
            mixed = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4.0 * h * h)
            H[i, j] = H[j, i] = mixed
    return H


def directional_derivative(
    f: ScalarFn, x: np.ndarray, direction: np.ndarray, h: float = 1e-6
) -> float:
    """Derivative of f at x along a direction, unit-normalized first."""
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    u = direction / norm
    return float(gradient(f, x, h) @ u)


def gradient_check(
    expr: Expr,
    var_names: list[str],
    point: np.ndarray,
    tol: float = 1e-4,
) -> float:
    """Compare symbolic partials against finite differences.

    The symbolic partial of each variable is evaluated exactly with
    eval_multi; the numeric side re-evaluates the original tree through
    central differences. This is the same idea as gradient checking a
    neural net by hand. Returns the largest disagreement.
    """
    point = np.asarray(point, dtype=float)
    env0 = dict(zip(var_names, point))

    def f(x: np.ndarray) -> float:
        return eval_multi(expr, dict(zip(var_names, x)))

    numeric = numerical_gradient(f, point)
    symbolic = np.array(
        [eval_multi(partial(expr, name), env0) for name in var_names]
    )
    diff = np.abs(numeric - symbolic)
    return float(np.max(diff))
