"""Convergence tests for infinite series, building on series.py.

The ratio test from series.py says nothing when the ratio tends to 1,
which is every p-series. This module adds the tools that decide those
cases: a geometric tail estimate for partial sums, the alternating
series error bound, the integral test run through Simpson's rule from
integrals.py, and the direct comparison test. All verdicts are numeric,
so they come with the usual caveat that a finite computation can
suggest divergence but never prove it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from calccode.integrals import simpson


def partial_sums(term: Callable[[int], float], n: int, start: int = 1) -> np.ndarray:
    """Cumulative sums S_start .. S_n as an array, for watching a series settle."""
    out = np.empty(n - start + 1)
    total = 0.0
    for k in range(start, n + 1):
        total += term(k)
        out[k - start] = total
    return out


def tail_estimate(term: Callable[[int], float], n: int) -> float:
    """Tail estimate |a_{n+1}| / (1 - r) with r = |a_{n+1} / a_n|.

    Exact when the terms from n on are geometric, a rough bound
    otherwise. Raises ValueError when r >= 1, where the estimate is
    meaningless because the terms are not shrinking fast enough.
    """
    a_n = term(n)
    a_next = term(n + 1)
    if a_n == 0.0:
        raise ValueError("term is zero, ratio undefined")
    r = abs(a_next / a_n)
    if r >= 1.0:
        raise ValueError("terms not shrinking, no geometric tail bound")
    return abs(a_next) / (1.0 - r)


def alternating_error_bound(term: Callable[[int], float], n: int) -> float:
    """|S - S_n| <= |a_{n+1}| for an alternating series whose unsigned
    terms decrease to 0. Returns abs(term(n + 1)), the first omitted term."""
    return abs(term(n + 1))


def integral_test(
    f: Callable[[float], float], a: float = 1.0, decades: int = 5, panels: int = 1000
) -> tuple[float, str]:
    """Numeric integral test for a positive decreasing f.

    Integrates f decade by decade from a out to a * 10^decades with
    Simpson's rule; splitting by decade keeps the panel width small where
    f changes fastest. If the last decade adds at most 1% of the running
    total, the integral has settled and the series converges with it. If
    the integral is still growing, the series diverges with it. Returns
    the integral estimate and the verdict.
    """
    total = 0.0
    last = 0.0
    lo = a
    for _ in range(decades):
        hi = lo * 10.0
        last = simpson(f, lo, hi, panels)
        total += last
        lo = hi
    verdict = "converges" if last <= 0.01 * total else "diverges"
    return total, verdict


def comparison_test(
    term: Callable[[int], float],
    benchmark: Callable[[int], float],
    n0: int = 1,
    n1: int = 1000,
) -> tuple[float, str]:
    """Direct comparison for positive series.

    Returns the largest ratio term(k)/benchmark(k) over k = n0..n1 and
    "bounded" if every ratio is at most 1, so the series converges
    whenever the benchmark does. "not bounded" means the comparison
    fails and says nothing about convergence.
    """
    worst = 0.0
    for k in range(n0, n1 + 1):
        ratio = term(k) / benchmark(k)
        if ratio > worst:
            worst = ratio
    verdict = "bounded" if worst <= 1.0 else "not bounded"
    return worst, verdict


def p_series_verdict(p: float, decades: int = 5) -> str:
    """Numeric verdict for the series of 1/k^p: converges iff p > 1.

    Runs the integral test on x^-p. The boundary p = 1 is the harmonic
    series and comes out divergent, as the theory demands.
    """
    _, verdict = integral_test(lambda x: x ** (-p), decades=decades)
    return verdict
