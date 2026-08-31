import math

import numpy as np
import pytest

from calccode.integrals import (
    convergence_study,
    integrate_inv_sqrt,
    left_sum,
    midpoint_sum,
    orders,
    right_sum,
    simpson,
    trapezoid,
)


def test_riemann_sums_approach_known_integral():
    # Integral of sin over [0, pi] is 2.
    assert abs(left_sum(math.sin, 0.0, math.pi, 10000) - 2.0) < 1e-3
    assert abs(right_sum(math.sin, 0.0, math.pi, 10000) - 2.0) < 1e-3


def test_midpoint_beats_endpoint_rules_at_same_n():
    exact = math.e - 1.0  # integral of exp over [0, 1]
    err_left = abs(left_sum(math.exp, 0.0, 1.0, 100) - exact)
    err_mid = abs(midpoint_sum(math.exp, 0.0, 1.0, 100) - exact)
    assert err_mid < err_left / 100.0


def test_trapezoid_on_sin():
    assert abs(trapezoid(math.sin, 0.0, math.pi, 1000) - 2.0) < 5e-6


def test_simpson_is_exact_for_cubics():
    f = lambda x: x**3 - 2.0 * x**2 + x - 4.0
    exact = (81.0 / 4 - 18.0 + 4.5 - 12.0) - 0.0  # antiderivative at 3 minus at 0
    assert abs(simpson(f, 0.0, 3.0, 2) - exact) < 1e-12


def test_simpson_rejects_odd_panel_count():
    with pytest.raises(ValueError):
        simpson(math.sin, 0.0, 1.0, 5)


def test_trapezoid_is_second_order():
    study = convergence_study(math.exp, 0.0, 1.0, math.e - 1.0)
    assert 1.9 < orders(study, 0.0, 1.0)["trapezoid_err"] < 2.1


def test_simpson_is_fourth_order():
    study = convergence_study(math.exp, 0.0, 1.0, math.e - 1.0)
    assert 3.8 < orders(study, 0.0, 1.0)["simpson_err"] < 4.2


def test_midpoint_is_second_order():
    study = convergence_study(math.exp, 0.0, 1.0, math.e - 1.0)
    assert 1.9 < orders(study, 0.0, 1.0)["midpoint_err"] < 2.1


def test_simpson_far_more_accurate_than_trapezoid():
    exact = math.e - 1.0
    err_trap = abs(trapezoid(math.exp, 0.0, 1.0, 128) - exact)
    err_simp = abs(simpson(math.exp, 0.0, 1.0, 128) - exact)
    assert err_simp < err_trap / 1000.0


def test_improper_integral_substitution_beats_naive():
    result = integrate_inv_sqrt(1000)
    assert result["substituted_err"] < 1e-12
    assert result["naive_err"] > 1e3 * result["substituted_err"]


def test_naive_midpoint_on_inv_sqrt_converges_slowly():
    # Error decays like 1/sqrt(n): 100x more panels buys only 10x accuracy.
    err_coarse = integrate_inv_sqrt(100)["naive_err"]
    err_fine = integrate_inv_sqrt(10000)["naive_err"]
    assert 3.0 < err_coarse / err_fine < 30.0
