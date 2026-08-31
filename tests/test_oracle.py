"""Oracle tests: hand-written calccode math against numpy and scipy.

The no-library-math rule binds calccode/ only. This file is the other
side of the contract: fixed inputs, answers from numpy and scipy, and a
tolerance between. The whole module skips when scipy is not installed;
CI installs requirements-dev.txt so these always run there.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.integrate  # noqa: E402
import scipy.linalg  # noqa: E402

from calccode import fourier, integrals, linalg, orthogonal  # noqa: E402

# A fixed overdetermined system: 6 points near y = 1 + 2x.
A_LS = np.array(
    [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
)
B_LS = np.array([1.1, 2.9, 5.2, 6.8, 9.1, 10.9])

# A fixed 5x4 for the SVD comparison.
A_SVD = np.array(
    [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [2.0, 1.0, 0.0, 3.0],
        [4.0, 3.0, 2.0, 1.0],
        [0.5, 1.5, 2.5, 3.5],
    ]
)

# A fixed well-conditioned 4x4 for matmul / det / solve.
A4 = np.array(
    [
        [4.0, 1.0, 2.0, 0.5],
        [1.0, 5.0, 1.0, 2.0],
        [2.0, 1.0, 6.0, 1.0],
        [0.5, 2.0, 1.0, 7.0],
    ]
)
B4 = np.array(
    [[1.0, 0.5, 2.0, 1.0], [3.0, 1.0, 0.0, 2.0], [1.0, 4.0, 1.0, 0.5], [2.0, 0.0, 3.0, 1.0]]
)
X4 = np.array([1.0, 2.0, 3.0, 4.0])


def test_lstsq_qr_matches_scipy():
    ours = orthogonal.lstsq_qr(A_LS, B_LS)
    theirs, _, _, _ = scipy.linalg.lstsq(A_LS, B_LS)
    assert ours == pytest.approx(theirs, abs=1e-10)


def test_svd_jacobi_singular_values_match_scipy():
    _, s_ours, _ = orthogonal.svd_jacobi(A_SVD)
    s_theirs = scipy.linalg.svdvals(A_SVD)
    assert s_ours == pytest.approx(s_theirs, rel=1e-9)


def test_fft_matches_numpy():
    rng = np.arange(64, dtype=float)
    signal = np.sin(rng / 5.0) + 0.5 * np.cos(rng / 11.0)
    ours = fourier.fft(signal)
    theirs = np.fft.fft(signal)
    assert ours == pytest.approx(theirs, abs=1e-10)


def test_matmul_matches_numpy():
    ours = linalg.matmul(A4, B4)
    assert ours == pytest.approx(A4 @ B4, abs=1e-12)


def test_determinant_matches_numpy():
    assert linalg.determinant(A4) == pytest.approx(np.linalg.det(A4), rel=1e-10)


def test_solve_matches_numpy():
    b = A4 @ X4
    ours = linalg.solve(A4, b)
    assert ours == pytest.approx(np.linalg.solve(A4, b), abs=1e-10)
    assert ours == pytest.approx(X4, abs=1e-10)


def test_simpson_matches_scipy_quad():
    ours = integrals.simpson(np.sin, 0.0, np.pi, 1000)
    theirs, _ = scipy.integrate.quad(np.sin, 0.0, np.pi)
    assert ours == pytest.approx(theirs, abs=1e-9)
    assert theirs == pytest.approx(2.0, abs=1e-12)
