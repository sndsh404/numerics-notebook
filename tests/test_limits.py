import math

import numpy as np

from calccode.limits import limit, one_sided_limit


def test_sin_x_over_x_converges_to_one():
    res = limit(lambda x: math.sin(x) / x, 0.0)
    assert res.status == "converged"
    assert abs(res.value - 1.0) < 1e-3


def test_polynomial_limit_at_interior_point():
    res = limit(lambda x: x**2 + 3.0 * x, 2.0)
    assert res.status == "converged"
    assert abs(res.value - 10.0) < 1e-6


def test_one_over_x_diverges_from_each_side():
    right = one_sided_limit(lambda x: 1.0 / x, 0.0, "right")
    left = one_sided_limit(lambda x: 1.0 / x, 0.0, "left")
    assert right.status == "divergent" and right.value == math.inf
    assert left.status == "divergent" and left.value == -math.inf


def test_one_over_x_two_sided_does_not_exist():
    res = limit(lambda x: 1.0 / x, 0.0)
    assert res.status == "does not exist"


def test_one_over_x_squared_diverges_both_sides():
    res = limit(lambda x: 1.0 / x**2, 0.0)
    assert res.status == "divergent"
    assert res.value == math.inf


def test_sin_one_over_x_oscillates():
    res = limit(lambda x: math.sin(1.0 / x), 0.0)
    assert res.status == "oscillating"


def test_sqrt_left_of_zero_is_undefined():
    res = one_sided_limit(math.sqrt, 0.0, "left")
    assert res.status == "undefined"


def test_samples_shrink_toward_the_point():
    res = one_sided_limit(lambda x: math.exp(x), 1.0, "right")
    # Ordered far to near: a convergent sequence settles down.
    early_spread = abs(res.samples[1] - res.samples[0])
    late_spread = abs(res.samples[-1] - res.samples[-2])
    assert late_spread < early_spread
