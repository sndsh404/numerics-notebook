"""Eigenvalues by hand: power iteration, inverse iteration, deflation.

All matrix algebra goes through linalg.py, and the random start vectors
come from the xorshift generator in montecarlo.py. No numpy.linalg
anywhere.
"""

from __future__ import annotations

import numpy as np

from calccode import linalg
from calccode.montecarlo import Xorshift32


def _norm(v: np.ndarray) -> float:
    return linalg.norm(v)


def _dot(u: np.ndarray, v: np.ndarray) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(u, v)))


def _start_vector(n: int, seed: int) -> np.ndarray:
    rng = Xorshift32(seed)
    v = np.array([rng.uniform() - 0.5 for _ in range(n)])
    return v / _norm(v)


def _matvec(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    return linalg.matmul(A, v.reshape(-1, 1)).ravel()


def _rayleigh(A: np.ndarray, v: np.ndarray) -> float:
    Av = _matvec(A, v)
    return _dot(v, Av) / _dot(v, v)


def power_iteration(
    A: np.ndarray, n_iter: int = 1000, tol: float = 1e-10, seed: int = 1
) -> tuple[float, np.ndarray]:
    """Dominant eigenvalue and eigenvector by repeated multiplication.

    Converges when the largest eigenvalue is strictly largest in
    magnitude. The ratio |lambda2 / lambda1| sets the speed.
    """
    A = np.asarray(A, dtype=float)
    v = _start_vector(A.shape[0], seed)
    lam_old = _rayleigh(A, v)
    for _ in range(n_iter):
        w = _matvec(A, v)
        norm = _norm(w)
        if norm == 0.0:
            raise ValueError("hit the zero vector; try another seed")
        v = w / norm
        lam = _rayleigh(A, v)
        if abs(lam - lam_old) < tol:
            return lam, v
        lam_old = lam
    return lam_old, v


def inverse_iteration(
    A: np.ndarray, shift: float, n_iter: int = 100, tol: float = 1e-12, seed: int = 1
) -> tuple[float, np.ndarray]:
    """Eigenvalue nearest the shift, by power iteration on (A - shift I)^{-1}.

    Each step solves (A - shift I) w = v with the Gaussian elimination
    from linalg.py. The Rayleigh quotient recovers the eigenvalue of A.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    shifted = A - shift * linalg.identity(n)
    v = _start_vector(n, seed)
    lam_old = _rayleigh(A, v)
    for _ in range(n_iter):
        w = linalg.solve(shifted, v)
        norm = _norm(w)
        if norm == 0.0:
            raise ValueError("hit the zero vector; try another shift")
        v = w / norm
        lam = _rayleigh(A, v)
        if abs(lam - lam_old) < tol:
            return lam, v
        lam_old = lam
    return lam_old, v


def deflate_symmetric(A: np.ndarray, lam: float, v: np.ndarray) -> np.ndarray:
    """Remove a known eigenpair from a symmetric matrix: A - lam v v^T.

    Power iteration on the result finds the next eigenvalue. Only valid
    for symmetric A, where the eigenvectors are orthogonal.
    """
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    v = v / _norm(v)
    return A - lam * np.outer(v, v)


def residual(A: np.ndarray, lam: float, v: np.ndarray) -> float:
    """||A v - lambda v||, the honest check for an eigenpair."""
    return _norm(_matvec(A, v) - lam * v)
