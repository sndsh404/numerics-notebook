"""Exercise: Monte Carlo.

Reference implementation: calccode/montecarlo.py.
"""

from __future__ import annotations

import numpy as np

from calccode.montecarlo import Xorshift32


def estimate_pi(n: int, seed: int = 42) -> float:
    """Fraction of n random points in the unit square that land inside the
    quarter circle of radius 1, times 4. Draw points from Xorshift32."""
    raise NotImplementedError


def pi_error_scaling(ns: np.ndarray, n_seeds: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviation of pi estimates at each sample count in ns.

    Use several seeds per n; the standard deviation across seeds is what
    should shrink like 1/sqrt(n). Returns (ns, stds).
    """
    raise NotImplementedError
