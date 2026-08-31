import math

import numpy as np

from calccode.series import (
    partial_sum,
    partial_sum_history,
    ratio_test,
    taylor_coefficients_from_expr,
    taylor_coefficients_numeric,
    taylor_error,
    taylor_polynomial,
)
from calccode.symbolic import Exp, Sin, Var

x = Var("x")


def test_sin_coefficients_from_symbolic():
    coeffs = taylor_coefficients_from_expr(Sin(x), 0.0, 11)
    expected = [0.0, 1.0, 0.0, -1.0 / 6, 0.0, 1.0 / 120, 0.0, -1.0 / 5040]
    for got, want in zip(coeffs[:8], expected):
        assert abs(got - want) < 1e-12


def test_taylor_polynomial_matches_sin_near_zero():
    coeffs = taylor_coefficients_from_expr(Sin(x), 0.0, 11)
    p = taylor_polynomial(coeffs, 0.0)
    assert abs(p(0.5) - math.sin(0.5)) < 1e-9


def test_exp_taylor_at_one():
    coeffs = taylor_coefficients_from_expr(Exp(x), 0.0, 10)
    p = taylor_polynomial(coeffs, 0.0)
    assert abs(p(1.0) - math.e) < 1e-7


def test_error_grows_far_from_expansion_point():
    coeffs = taylor_coefficients_from_expr(Sin(x), 0.0, 5)
    p = taylor_polynomial(coeffs, 0.0)
    err_near = abs(p(1.0) - math.sin(1.0))
    err_far = abs(p(3.0) - math.sin(3.0))
    assert err_far > 100.0 * err_near


def test_higher_degree_extends_accurate_region():
    xs = np.array([2.5])
    p5 = taylor_polynomial(taylor_coefficients_from_expr(Sin(x), 0.0, 5), 0.0)
    p15 = taylor_polynomial(taylor_coefficients_from_expr(Sin(x), 0.0, 15), 0.0)
    assert taylor_error(math.sin, p15, xs)[0] < taylor_error(math.sin, p5, xs)[0] / 100.0


def test_numeric_coefficients_match_symbolic():
    exact = taylor_coefficients_from_expr(Exp(x), 0.0, 5)
    numeric = taylor_coefficients_numeric(math.exp, 0.0, 5, h=2e-3)
    for got, want in zip(numeric, exact):
        assert abs(got - want) < 5e-3


def test_partial_sum_of_geometric_series():
    s = partial_sum(lambda k: 0.5**k, 30)
    assert abs(s - 2.0) < 1e-8


def test_partial_sum_history_is_cumulative():
    hist = partial_sum_history(lambda k: 1.0 / math.factorial(k), 12)
    assert np.all(np.diff(hist) > 0.0)
    assert abs(hist[-1] - math.e) < 1e-9
    assert hist[-1] == partial_sum(lambda k: 1.0 / math.factorial(k), 12)


def test_ratio_test_convergent_geometric():
    estimate, verdict = ratio_test(lambda k: 0.5**k)
    assert verdict == "converges"
    assert abs(estimate - 0.5) < 1e-9


def test_ratio_test_divergent_geometric():
    estimate, verdict = ratio_test(lambda k: 2.0**k)
    assert verdict == "diverges"
    assert abs(estimate - 2.0) < 1e-9


def test_ratio_test_inconclusive_on_harmonic():
    # a_{n+1}/a_n for 1/k tends to 1, where the ratio test says nothing.
    _, verdict = ratio_test(lambda k: 1.0 / k)
    assert verdict == "inconclusive"


def test_ratio_test_on_factorial_series():
    # 1/k! converges and the ratio 1/(k+1) tends to 0. Sampled at small
    # n because factorial(800) overflows a float.
    estimate, verdict = ratio_test(lambda k: 1.0 / math.factorial(k), n0=20, n1=150)
    assert verdict == "converges"
    assert estimate < 0.01
