"""Orthogonal factorizations, written by hand on numpy arrays.

Gram-Schmidt two ways, Householder QR, and one-sided Jacobi SVD, plus
the tools the SVD buys for free: pseudoinverse, condition number, rank,
low-rank approximation, and least squares through the QR. Nothing here
touches numpy.linalg; every rotation and projection is an explicit loop.
"""

from __future__ import annotations

import math

import numpy as np

from calccode import linalg


def _as_matrix(A: np.ndarray) -> np.ndarray:
    M = np.asarray(A, dtype=float)
    if M.ndim != 2:
        raise ValueError("expected a 2D matrix")
    return M


def _dot(u: np.ndarray, v: np.ndarray) -> float:
    """Inner product as a plain loop."""
    return float(sum(float(a) * float(b) for a, b in zip(u, v)))


def _fro(A: np.ndarray) -> float:
    """Frobenius norm, sqrt of the sum of all squared entries."""
    return math.sqrt(sum(float(x) * float(x) for x in A.ravel()))


def classical_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR by classical Gram-Schmidt: project each column against the
    original column, then subtract every projection at once.

    Textbook clean and unstable in practice: when the columns of A are
    nearly dependent, the rounding error in one projection feeds the
    next, and Q drifts away from orthonormal.
    """
    A = _as_matrix(A)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = _dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = linalg.norm(v)
        if R[j, j] == 0.0:
            raise ValueError("columns of A are linearly dependent")
        Q[:, j] = v / R[j, j]
    return Q, R


def modified_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR by modified Gram-Schmidt: each projection is computed against
    the partially cleaned vector, not the original column.

    Same arithmetic count as the classical version, but the error from
    each subtraction gets projected away before the next one, so Q stays
    orthonormal to machine precision on bases where the classical loop
    loses it.
    """
    A = _as_matrix(A)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = _dot(Q[:, i], v)
            v -= R[i, j] * Q[:, i]
        R[j, j] = linalg.norm(v)
        if R[j, j] == 0.0:
            raise ValueError("columns of A are linearly dependent")
        Q[:, j] = v / R[j, j]
    return Q, R


def qr_householder(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR by Householder reflections. Returns the thin factorization:
    Q is m by n with orthonormal columns, R is n by n upper triangular.

    Each step reflects the subdiagonal part of column j onto the first
    axis with H = I - 2 v v^T, which zeroes it in one shot. The
    reflectors are stored as vectors and replayed against the identity
    columns at the end to build Q, so the factorization costs the same
    as one pass over the matrix.
    """
    A = _as_matrix(A)
    m, n = A.shape
    if m < n:
        raise ValueError("householder QR here needs at least as many rows as columns")
    R = A.copy()
    reflectors: list[np.ndarray | None] = []
    for j in range(n):
        x = R[j:, j]
        nx = linalg.norm(x)
        if nx == 0.0:
            reflectors.append(None)
            continue
        v = x.copy()
        v[0] += math.copysign(nx, x[0] if x[0] != 0.0 else 1.0)
        v = v / linalg.norm(v)
        for col in range(j, n):
            coeff = 2.0 * _dot(v, R[j:, col])
            R[j:, col] -= coeff * v
        reflectors.append(v)
    Q = np.zeros((m, n))
    for col in range(n):
        e = np.zeros(m)
        e[col] = 1.0
        for j in range(len(reflectors) - 1, -1, -1):
            v = reflectors[j]
            if v is None:
                continue
            e[j:] -= 2.0 * _dot(v, e[j:]) * v
        Q[:, col] = e
    return Q, R[:n, :]


def svd_jacobi(
    A: np.ndarray, tol: float = 1e-12, max_sweeps: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD by one-sided Jacobi rotations. Returns U (m by n), the
    singular values s sorted largest first, and V (n by n), so that
    A = U diag(s) V^T.

    Every rotation is a 2x2 Givens applied to a pair of columns of the
    working matrix (and accumulated into V). Rotating columns p and q by
    the angle that kills their inner product drives the columns toward
    mutual orthogonality; once every pairwise inner product is below tol
    times the column norms, the column norms are the singular values.
    Sweeps stop early when the off-diagonal norm of A^T A drops below
    tol times the Frobenius norm, and are capped at max_sweeps.
    """
    A = _as_matrix(A)
    m, n = A.shape
    U = A.copy()
    V = linalg.identity(n)
    scale = max(_fro(A), 1e-300)
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                app = _dot(U[:, p], U[:, p])
                aqq = _dot(U[:, q], U[:, q])
                apq = _dot(U[:, p], U[:, q])
                off += apq * apq
                if app == 0.0 or aqq == 0.0:
                    continue
                if abs(apq) <= tol * math.sqrt(app * aqq):
                    continue
                zeta = (aqq - app) / (2.0 * apq)
                t = math.copysign(1.0, zeta) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                up, uq = U[:, p].copy(), U[:, q].copy()
                U[:, p] = c * up - s * uq
                U[:, q] = s * up + c * uq
                vp, vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq
        if math.sqrt(off) <= tol * scale:
            break
    s = np.array([linalg.norm(U[:, j]) for j in range(n)])
    order = sorted(range(n), key=lambda j: -s[j])
    U_out = np.zeros((m, n))
    V_out = np.zeros((n, n))
    s_out = np.zeros(n)
    for new, old in enumerate(order):
        s_out[new] = s[old]
        if s[old] > 0.0:
            U_out[:, new] = U[:, old] / s[old]
        V_out[:, new] = V[:, old]
    return U_out, s_out, V_out


def pseudoinverse(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Moore-Penrose pseudoinverse V S^+ U^T from the Jacobi SVD.

    Singular values at or below tol times the largest one are treated
    as zero and inverted to zero, which is what keeps A^+ finite on
    rank-deficient input.
    """
    A = _as_matrix(A)
    m, n = A.shape
    U, s, V = svd_jacobi(A)
    cutoff = tol * (s[0] if s.size else 0.0)
    W = np.zeros((n, m))  # S^+ U^T
    for i in range(n):
        if s[i] > cutoff:
            for row in range(m):
                W[i, row] = U[row, i] / s[i]
    return linalg.matmul(V, W)


def condition_number(A: np.ndarray) -> float:
    """Ratio of the largest to the smallest singular value.

    Infinity when the smallest singular value is exactly zero.
    """
    _, s, _ = svd_jacobi(_as_matrix(A))
    if s.size == 0:
        raise ValueError("condition number needs a nonempty matrix")
    if s[-1] == 0.0:
        return math.inf
    return float(s[0] / s[-1])


def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """Number of singular values larger than tol times the largest one.

    A relative cutoff, so it answers "how many dimensions carry weight"
    rather than "how many pivots survived rounding", which is what makes
    it say the honest thing about the Hilbert matrix.
    """
    _, s, _ = svd_jacobi(_as_matrix(A))
    if s.size == 0 or s[0] == 0.0:
        return 0
    return int(sum(1 for x in s if x > tol * s[0]))


def low_rank_approx(A: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation: the first k terms of the SVD sum.

    Eckart-Young says the error in the 2-norm is exactly the first
    dropped singular value, which the test checks directly.
    """
    A = _as_matrix(A)
    m, n = A.shape
    if not 0 <= k <= n:
        raise ValueError("k must lie between 0 and the column count")
    U, s, V = svd_jacobi(A)
    out = np.zeros((m, n))
    for i in range(min(k, n)):
        for row in range(m):
            for col in range(n):
                out[row, col] += s[i] * U[row, i] * V[col, i]
    return out


def lstsq_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least squares solution of Ax = b through Householder QR.

    With A = QR the normal equations collapse to R x = Q^T b, a
    triangular solve. Unlike the normal equations route in
    regression.py, nothing gets squared here, so the condition number
    of the problem stays the condition number of A.
    """
    A = _as_matrix(A)
    b = np.asarray(b, dtype=float).ravel()
    if b.size != A.shape[0]:
        raise ValueError("b must have one entry per row of A")
    Q, R = qr_householder(A)
    n = R.shape[0]
    qtb = np.array([_dot(Q[:, i], b) for i in range(n)])
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        rest = sum(R[row, j] * x[j] for j in range(row + 1, n))
        if R[row, row] == 0.0:
            raise ValueError("A does not have full column rank")
        x[row] = (qtb[row] - rest) / R[row, row]
    return x
