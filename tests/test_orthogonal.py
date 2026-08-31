import math

import numpy as np
import pytest

from calccode import linalg, orthogonal, regression


def hilbert(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H


def orthonormality_error(Q):
    """Frobenius norm of Q^T Q - I, computed with the repo's own tools."""
    G = linalg.matmul(linalg.transpose(Q), Q)
    n = G.shape[0]
    return math.sqrt(sum((G[i, j] - (1.0 if i == j else 0.0)) ** 2 for i in range(n) for j in range(n)))


def test_modified_gram_schmidt_beats_classical_on_near_dependent_basis():
    # Columns (1,eps,0,0), (1,0,eps,0), (1,0,0,eps): each pair agrees to
    # seven digits, so the projections in Gram-Schmidt almost cancel.
    eps = 1e-8
    A = np.array([[1.0, 1.0, 1.0], [eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]])
    Q_classical, _ = orthogonal.classical_gram_schmidt(A)
    Q_modified, _ = orthogonal.modified_gram_schmidt(A)
    err_classical = orthonormality_error(Q_classical)
    err_modified = orthonormality_error(Q_modified)
    # Measured: classical 7.07e-01, modified 1.15e-08.
    assert err_classical > 1e-1
    assert err_modified < 1e-6
    assert err_modified < err_classical * 1e-6


def test_classical_gram_schmidt_is_fine_on_a_well_separated_basis():
    # The failure above is about the basis, not the code path.
    A = np.array([[1.0, 0.0, 1.0], [1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [1.0, 1.0, 1.0]])
    Q, R = orthogonal.classical_gram_schmidt(A)
    assert orthonormality_error(Q) < 1e-12
    assert np.max(np.abs(linalg.matmul(Q, R) - A)) < 1e-12


def test_qr_householder_on_fixed_6x4():
    A = np.array([
        [2.0, 1.0, 0.0, 3.0],
        [1.0, 4.0, 1.0, 0.0],
        [0.0, 1.0, 3.0, 1.0],
        [1.0, 0.0, 2.0, 2.0],
        [3.0, 1.0, 1.0, 1.0],
        [0.0, 2.0, 0.0, 1.0],
    ])
    Q, R = orthogonal.qr_householder(A)
    assert Q.shape == (6, 4)
    assert R.shape == (4, 4)
    # A = Q R to machine precision.
    assert np.max(np.abs(linalg.matmul(Q, R) - A)) < 1e-12
    # Q has orthonormal columns.
    assert orthonormality_error(Q) < 1e-12
    # R is upper triangular.
    for i in range(4):
        for j in range(i):
            assert R[i, j] == pytest.approx(0.0, abs=1e-14)


def test_svd_jacobi_recovers_known_singular_values():
    # A = U0 diag(5, 3, 1) V0^T with two fixed plane rotations.
    c1, s1 = math.cos(0.7), math.sin(0.7)
    c2, s2 = math.cos(1.1), math.sin(1.1)
    U0 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
    V0 = np.array([[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]])
    A = linalg.matmul(U0, linalg.matmul(np.diag([5.0, 3.0, 1.0]), linalg.transpose(V0)))
    U, s, V = orthogonal.svd_jacobi(A)
    assert np.allclose(s, [5.0, 3.0, 1.0], atol=1e-10)
    # Reconstruction A = U diag(s) V^T.
    S = np.diag(s)
    assert np.max(np.abs(linalg.matmul(U, linalg.matmul(S, linalg.transpose(V))) - A)) < 1e-10


def test_low_rank_approx_error_is_the_dropped_singular_value():
    c1, s1 = math.cos(0.7), math.sin(0.7)
    c2, s2 = math.cos(1.1), math.sin(1.1)
    U0 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
    V0 = np.array([[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]])
    A = linalg.matmul(U0, linalg.matmul(np.diag([5.0, 3.0, 1.0]), linalg.transpose(V0)))
    _, s, _ = orthogonal.svd_jacobi(A)
    A1 = orthogonal.low_rank_approx(A, 1)
    # Eckart-Young: ||A - A1||_2 equals sigma_2. The 2-norm of the error
    # is its own top singular value, read off with the same SVD.
    err_top = orthogonal.svd_jacobi(A - A1)[1][0]
    assert err_top == pytest.approx(s[1], abs=1e-8)
    assert err_top == pytest.approx(3.0, abs=1e-8)


def test_hilbert_rank_pivots_vs_singular_values():
    H = hilbert(8)
    # Gaussian elimination sees eight pivots and calls it full rank.
    assert linalg.rank(H) == 8
    # The singular values tell the truth: 1.7 down to 1.1e-10, so at a
    # relative cutoff of 1e-8 one direction is already noise (7), and at
    # 1e-4 over half of them are gone (4).
    assert orthogonal.matrix_rank(H, tol=1e-8) == 7
    assert orthogonal.matrix_rank(H, tol=1e-4) == 4


def test_hilbert10_condition_number_is_huge():
    assert orthogonal.condition_number(hilbert(10)) > 1e12


def test_lstsq_qr_beats_normal_equations_on_hilbert10():
    H = hilbert(10)
    x_true = np.arange(1.0, 11.0)
    b = linalg.matmul(H, x_true.reshape(-1, 1)).ravel()
    # Squaring the condition number kills the normal equations outright:
    # the 11x11 augmented system is singular to the elimination solver.
    with pytest.raises(ValueError):
        regression.ols_closed_form(H, b)
    # The QR route never squares anything and recovers the solution.
    x_qr = orthogonal.lstsq_qr(H, b)
    assert np.max(np.abs(x_qr - x_true)) < 1e-2


def test_lstsq_qr_on_a_tame_overdetermined_system():
    # 5 points on the line y = 2x - 1, exact fit through lstsq_qr.
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    b = np.array([-1.0, 1.0, 3.0, 5.0, 7.0])
    x = orthogonal.lstsq_qr(A, b)
    assert np.allclose(x, [-1.0, 2.0], atol=1e-10)


def test_pseudoinverse_satisfies_a_aplus_a_equals_a():
    A = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0], [2.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    pinv = orthogonal.pseudoinverse(A)
    assert np.max(np.abs(linalg.matmul(A, linalg.matmul(pinv, A)) - A)) < 1e-10


def test_pseudoinverse_on_rank_deficient_matrix():
    # Every column is a multiple of (1, 2, 3): rank 1.
    A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    pinv = orthogonal.pseudoinverse(A)
    assert np.max(np.abs(linalg.matmul(A, linalg.matmul(pinv, A)) - A)) < 1e-10
