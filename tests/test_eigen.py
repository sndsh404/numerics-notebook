import numpy as np

from calccode import eigen, linalg


def symmetric_demo_matrix():
    # Eigenvalues 3 and 1, eigenvectors (1, 1) / sqrt(2) and (1, -1) / sqrt(2).
    return np.array([[2.0, 1.0], [1.0, 2.0]])


def bigger_symmetric_matrix():
    # Symmetric with eigenvalues 4.879, 2.653, 1.468 (from this module itself).
    return np.array([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]])


def test_power_iteration_finds_dominant_eigenvalue():
    A = symmetric_demo_matrix()
    lam, v = eigen.power_iteration(A, tol=1e-14)
    assert abs(lam - 3.0) < 1e-10
    # Eigenvector is (1, 1) / sqrt(2) up to sign.
    assert abs(abs(v[0]) - 1.0 / np.sqrt(2.0)) < 1e-6
    assert abs(v[0] - v[1]) < 1e-6


def test_power_iteration_residual_is_small():
    A = bigger_symmetric_matrix()
    lam, v = eigen.power_iteration(A, n_iter=10000, tol=1e-14)
    assert abs(lam - 4.8793852415718) < 1e-8
    assert eigen.residual(A, lam, v) < 1e-6


def test_power_iteration_eigenvalue_matches_trace_argument():
    # For the 2x2 demo: eigenvalues sum to the trace, dominant one is 3.
    A = symmetric_demo_matrix()
    lam, _ = eigen.power_iteration(A)
    other = linalg.trace(A) - lam
    assert abs(other - 1.0) < 1e-8


def test_inverse_iteration_finds_non_dominant_eigenvalue():
    A = symmetric_demo_matrix()
    lam, v = eigen.inverse_iteration(A, shift=1.1)
    assert abs(lam - 1.0) < 1e-10
    # Eigenvector is (1, -1) / sqrt(2) up to sign.
    assert abs(v[0] + v[1]) < 1e-6
    assert eigen.residual(A, lam, v) < 1e-8


def test_inverse_iteration_on_bigger_matrix():
    A = bigger_symmetric_matrix()
    # The shift 2.1 sits closest to the middle eigenvalue 2.6527, so the
    # iteration skips both the dominant 4.879 and the smallest 1.468.
    lam, v = eigen.inverse_iteration(A, shift=2.1)
    assert abs(lam - 2.6527036446742) < 1e-8
    assert eigen.residual(A, lam, v) < 1e-5


def test_deflation_reveals_next_eigenvalue():
    A = symmetric_demo_matrix()
    lam1, v1 = eigen.power_iteration(A, tol=1e-14)
    A2 = eigen.deflate_symmetric(A, lam1, v1)
    lam2, v2 = eigen.power_iteration(A2, tol=1e-13)
    assert abs(lam2 - 1.0) < 1e-8
    assert eigen.residual(A, lam2, v2) < 1e-6


def test_eigenvectors_of_symmetric_matrix_are_orthogonal():
    A = symmetric_demo_matrix()
    _, v1 = eigen.power_iteration(A, tol=1e-14)
    _, v2 = eigen.inverse_iteration(A, shift=1.1)
    assert abs(float(v1 @ v2)) < 1e-6
