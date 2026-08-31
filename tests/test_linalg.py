
import numpy as np
import pytest

from calccode import linalg


def test_matmul_matches_manual_product():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(linalg.matmul(A, B), expected)


def test_matmul_rejects_bad_shapes():
    with pytest.raises(ValueError):
        linalg.matmul(np.ones((2, 3)), np.ones((2, 3)))


def test_transpose_swaps_entries():
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(linalg.transpose(A), A.T)


def test_trace_sums_diagonal():
    A = np.array([[1.0, 9.0], [9.0, 4.0]])
    assert linalg.trace(A) == 5.0


def test_determinant_matches_known_values():
    assert abs(linalg.determinant(np.array([[1.0, 2.0], [3.0, 4.0]])) - (-2.0)) < 1e-9
    assert abs(linalg.determinant(np.array([[6.0, 1.0], [5.0, 2.0]])) - 7.0) < 1e-9


def test_determinant_of_product_is_product():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    B = rng.normal(size=(4, 4))
    lhs = linalg.determinant(linalg.matmul(A, B))
    rhs = linalg.determinant(A) * linalg.determinant(B)
    assert abs(lhs - rhs) < 1e-8


def test_determinant_of_singular_matrix_is_zero():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert abs(linalg.determinant(A)) < 1e-12


def test_solve_recovers_known_solution():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    x = linalg.solve(A, b)
    assert np.allclose(x, [2.0, 3.0])


def test_solve_rejects_singular_matrix():
    with pytest.raises(ValueError):
        linalg.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))


def test_rank_counts_independent_rows():
    A = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 1.0]])
    assert linalg.rank(A) == 2


def test_identity_is_neutral():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(linalg.matmul(linalg.identity(2), A), A)
    assert np.allclose(linalg.matmul(A, linalg.identity(2)), A)
