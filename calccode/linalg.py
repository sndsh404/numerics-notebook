"""Hand-written matrix algorithms on numpy arrays.

numpy stores the numbers and nothing more: every algorithm here is a
loop written by hand. No numpy.linalg anywhere in this module.
"""

from __future__ import annotations

import math

import numpy as np


def norm(v: np.ndarray) -> float:
    """Euclidean norm, sqrt of the sum of squares, written by hand."""
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _as_matrix(A: np.ndarray) -> np.ndarray:
    M = np.asarray(A, dtype=float)
    if M.ndim != 2:
        raise ValueError("expected a 2D matrix")
    return M


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Triple loop matrix product."""
    A, B = _as_matrix(A), _as_matrix(B)
    n, k = A.shape
    k2, m = B.shape
    if k != k2:
        raise ValueError(f"shape mismatch: {A.shape} times {B.shape}")
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            total = 0.0
            for p in range(k):
                total += A[i, p] * B[p, j]
            C[i, j] = total
    return C


def transpose(A: np.ndarray) -> np.ndarray:
    A = _as_matrix(A)
    n, m = A.shape
    T = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            T[j, i] = A[i, j]
    return T


def trace(A: np.ndarray) -> float:
    A = _as_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("trace needs a square matrix")
    return float(sum(A[i, i] for i in range(A.shape[0])))


def _eliminate(M: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, int, int]:
    """Forward elimination with partial pivoting.

    Returns the upper triangular result, the number of row swaps, and
    the number of nonzero pivots found. A column counts as a pivot when
    its largest entry is at least tol times the largest entry of the
    input. Rows are only swapped and reduced by row_i -= factor *
    row_pivot, never scaled, so the product of the diagonal times
    (-1)^swaps is the determinant.
    """
    U = M.astype(float).copy()
    rows, cols = U.shape
    swaps = 0
    pivots = 0
    scale = max(1.0, float(np.max(np.abs(U))))
    for col in range(min(rows, cols)):
        pivot = col + int(np.argmax(np.abs(U[col:, col])))
        if abs(U[pivot, col]) < tol * scale:
            continue
        if pivot != col:
            U[[col, pivot]] = U[[pivot, col]]
            swaps += 1
        pivots += 1
        for row in range(col + 1, rows):
            factor = U[row, col] / U[col, col]
            U[row, col:] -= factor * U[col, col:]
    return U, swaps, pivots


def determinant(A: np.ndarray) -> float:
    """Determinant from the diagonal of the eliminated matrix."""
    A = _as_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("determinant needs a square matrix")
    U, swaps, pivots = _eliminate(A)
    if pivots < A.shape[0]:
        return 0.0
    det = 1.0
    for i in range(A.shape[0]):
        det *= U[i, i]
    return float(-det if swaps % 2 else det)


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b by Gaussian elimination with partial pivoting."""
    A = _as_matrix(A)
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[0]
    if A.shape[1] != n or b.size != n:
        raise ValueError("solve needs a square A and matching b")

    aug = np.zeros((n, n + 1))
    aug[:, :n] = A
    aug[:, n] = b
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot, col]) < 1e-12 * max(1.0, float(np.max(np.abs(aug[:, :n])))):
            raise ValueError("matrix is singular")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(col + 1, n):
            factor = aug[row, col] / aug[col, col]
            aug[row, col:] -= factor * aug[col, col:]

    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        rest = sum(aug[row, j] * x[j] for j in range(row + 1, n))
        x[row] = (aug[row, n] - rest) / aug[row, row]
    return x


def rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """Number of pivots at least tol times the largest entry of A.

    Pivots below that relative size are treated as numerical noise, so
    larger tol values report a lower effective rank.
    """
    A = _as_matrix(A)
    _, _, pivots = _eliminate(A, tol)
    return pivots


def identity(n: int) -> np.ndarray:
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.0
    return I
