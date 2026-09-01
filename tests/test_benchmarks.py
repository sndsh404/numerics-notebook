"""Wall-clock guards against accidental 10x slowdowns.

Thresholds are generous on purpose: they catch a refactor that turns a
fast path into a quadratic one, and they should never flake on CI.
"""

import time

import numpy as np

from calccode import fourier, linalg


def _timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_matmul_speed_guard():
    rng_rows = np.arange(80 * 80, dtype=float).reshape(80, 80)
    A = np.sin(rng_rows)
    B = np.cos(rng_rows)
    elapsed = _timed(lambda: linalg.matmul(A, B))
    assert elapsed < 5.0, f"matmul(80x80) took {elapsed:.2f}s"


def test_fft_speed_guard():
    n = 4096
    signal = np.sin(np.arange(n, dtype=float) * 0.1)
    elapsed = _timed(lambda: fourier.fft(signal))
    assert elapsed < 2.0, f"fft(4096) took {elapsed:.2f}s"
