"""Exercise: Fourier analysis.

Reference implementation: calccode/fourier.py.
"""

from __future__ import annotations

import numpy as np


def dft_magnitude(signal: np.ndarray, k: int) -> float:
    """Magnitude |X_k| of DFT bin k, from the direct O(n^2) sum.

    X_k = sum_n x_n exp(-2 pi i k n / N). For a pure tone of amplitude A
    at bin k, the magnitude is A * N / 2.
    """
    raise NotImplementedError


def dft_round_trip(signal: np.ndarray) -> np.ndarray:
    """Inverse DFT of the DFT of a signal.

    x_n = (1/N) sum_k X_k exp(+2 pi i k n / N). The result should match
    the input to roundoff.
    """
    raise NotImplementedError
