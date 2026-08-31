"""Exercise: interpolation.

Reference implementation: calccode/interpolation.py.
"""

from __future__ import annotations

import numpy as np


def piecewise_linear(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """Linear interpolation on the interval containing x, clamped at the ends."""
    raise NotImplementedError


def lagrange_basis(xs: np.ndarray, j: int, x: float) -> float:
    """Value of the j-th Lagrange basis polynomial at x.

    L_j(x) = prod_{k != j} (x - x_k) / (x_j - x_k). It equals 1 at x_j
    and 0 at every other node.
    """
    raise NotImplementedError
