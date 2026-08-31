"""Exercise: eigenvalues.

Reference implementation: calccode/eigen.py.
"""

from __future__ import annotations

import numpy as np


def power_step(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """One step of power iteration: multiply v by A and normalize."""
    raise NotImplementedError


def residual(A: np.ndarray, lam: float, v: np.ndarray) -> float:
    """||A v - lambda v||, the honest check for an eigenpair."""
    raise NotImplementedError
