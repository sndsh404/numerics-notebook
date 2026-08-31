"""Exercise: multivariable calculus.

Reference implementation: calccode/multivar.py.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ScalarFn = Callable[[np.ndarray], float]


def partial_diff(f: ScalarFn, x: np.ndarray, i: int, h: float = 1e-5) -> float:
    """Partial derivative of f with respect to coordinate i at x."""
    raise NotImplementedError


def gradient(f: ScalarFn, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Gradient vector at x: one partial derivative per coordinate."""
    raise NotImplementedError


def directional_derivative(
    f: ScalarFn, x: np.ndarray, direction: np.ndarray, h: float = 1e-6
) -> float:
    """Derivative of f at x along a direction, unit-normalized first."""
    raise NotImplementedError
