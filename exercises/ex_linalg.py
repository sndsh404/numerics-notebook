"""Exercise: hand-written matrix algorithms.

numpy arrays store the numbers; write the loops yourself.
Reference implementation: calccode/linalg.py.
"""

import numpy as np


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Triple loop matrix product."""
    raise NotImplementedError


def transpose(A: np.ndarray) -> np.ndarray:
    """Swap rows and columns."""
    raise NotImplementedError


def determinant(A: np.ndarray) -> float:
    """Determinant via elimination: product of the diagonal, sign from row swaps."""
    raise NotImplementedError


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b by Gaussian elimination with partial pivoting."""
    raise NotImplementedError


def rank(A: np.ndarray) -> int:
    """Number of nonzero pivots after elimination."""
    raise NotImplementedError
