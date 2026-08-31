import math

import numpy as np

from calccode.derivatives import (
    backward_diff,
    central_diff,
    convergence_study,
    fit_order,
    forward_diff,
    second_derivative,
)


def test_forward_diff_on_exp():
    assert abs(forward_diff(math.exp, 1.0) - math.e) < 1e-4


def test_backward_diff_on_exp():
    assert abs(backward_diff(math.exp, 1.0) - math.e) < 1e-4


def test_central_diff_beats_forward_at_same_h():
    h = 1e-3
    exact = math.cos(0.7)
    fwd_err = abs(forward_diff(math.sin, 0.7, h) - exact)
    cen_err = abs(central_diff(math.sin, 0.7, h) - exact)
    assert cen_err < fwd_err / 100.0


def test_second_derivative_of_sin():
    assert abs(second_derivative(math.sin, 0.9) - (-math.sin(0.9))) < 1e-6


def test_central_difference_is_second_order():
    # Fit the log-log slope over h values safely above roundoff noise.
    hs = np.logspace(-1, -4, 10)
    study = convergence_study(math.sin, 1.0, math.cos(1.0), hs)
    slope = fit_order(study["hs"], study["central_err"])
    assert 1.9 < slope < 2.1


def test_forward_difference_is_first_order():
    hs = np.logspace(-1, -4, 10)
    study = convergence_study(math.sin, 1.0, math.cos(1.0), hs)
    slope = fit_order(study["hs"], study["forward_err"])
    assert 0.9 < slope < 1.1


def test_halving_h_quarters_central_error():
    err_h = abs(central_diff(math.exp, 0.5, 1e-2) - math.exp(0.5))
    err_h2 = abs(central_diff(math.exp, 0.5, 5e-3) - math.exp(0.5))
    assert 3.0 < err_h / err_h2 < 5.0


def test_tiny_h_suffers_cancellation():
    # Below 1e-10 the subtraction in the numerator loses digits and the
    # central difference error stops improving.
    err_small = abs(central_diff(math.exp, 1.0, 1e-6) - math.e)
    err_tiny = abs(central_diff(math.exp, 1.0, 1e-12) - math.e)
    assert err_tiny > err_small
