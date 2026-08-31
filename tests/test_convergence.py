import math

import numpy as np
import pytest

from calccode.convergence import (
    alternating_error_bound,
    comparison_test,
    integral_test,
    p_series_verdict,
    partial_sums,
    tail_estimate,
)


def test_partial_sums_of_geometric_series():
    sums = partial_sums(lambda k: 0.5**k, 30, start=0)
    assert np.all(np.diff(sums) > 0.0)
    assert abs(sums[-1] - 2.0) < 1e-8


def test_tail_estimate_exact_for_geometric():
    # The true tail of sum 0.5^k after n is 2 * 0.5^(n+1).
    assert abs(tail_estimate(lambda k: 0.5**k, 20) - 2.0 * 0.5**21) < 1e-12


def test_tail_estimate_bounds_true_error():
    n = 20
    sums = partial_sums(lambda k: 0.5**k, n + 1, start=0)
    true_error = abs(2.0 - sums[-2])
    assert true_error <= tail_estimate(lambda k: 0.5**k, n) + 1e-15


def test_tail_estimate_rejects_nonshrinking_terms():
    with pytest.raises(ValueError):
        tail_estimate(lambda k: 1.0, 10)


def test_alternating_harmonic_bound_holds_for_ln2():
    # sum (-1)^(k+1) / k = ln 2; the error stays under the first omitted term.
    for n in (10, 100, 1000):
        sums = partial_sums(lambda k: (-1.0) ** (k + 1) / k, n)
        err = abs(sums[-1] - math.log(2.0))
        assert err <= alternating_error_bound(lambda k: 1.0 / k, n)


def test_integral_test_p_series():
    assert integral_test(lambda x: x ** -0.5)[1] == "diverges"
    assert integral_test(lambda x: x ** -1.0)[1] == "diverges"
    assert integral_test(lambda x: x ** -2.0)[1] == "converges"


def test_integral_test_estimate_for_p2():
    total, verdict = integral_test(lambda x: x ** -2.0)
    assert verdict == "converges"
    assert abs(total - 1.0) < 1e-4


def test_comparison_confirms_one_over_n2_plus_one():
    # 1/(n^2+1) < 1/n^2, and sum 1/n^2 converges, so this one does too.
    worst, verdict = comparison_test(lambda n: 1.0 / (n * n + 1.0), lambda n: 1.0 / (n * n))
    assert verdict == "bounded"
    assert worst > 0.9  # the ratio approaches 1 from below


def test_comparison_fails_when_benchmark_is_smaller():
    _, verdict = comparison_test(lambda n: 1.0 / n, lambda n: 1.0 / (n * n))
    assert verdict == "not bounded"


def test_p_series_verdict_matches_theory():
    assert p_series_verdict(0.5) == "diverges"
    assert p_series_verdict(1.0) == "diverges"
    assert p_series_verdict(1.5) == "converges"
    assert p_series_verdict(2.0) == "converges"
