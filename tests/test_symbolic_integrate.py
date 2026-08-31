import math

import numpy as np
import pytest

from calccode.integrals import simpson
from calccode.symbolic import Add, Const, Cos, Exp, Log, Mul, Pow, Sin, Var, diff, implicit_diff, partial
from calccode.symbolic import eval_multi
from calccode.symbolic_integrate import definite_integral, integrate

x = Var("x")
y = Var("y")


def check_antiderivative(integrand, lo, hi):
    """diff of the antiderivative must match the integrand numerically."""
    recovered = diff(integrate(integrand))
    rng = np.random.default_rng(3)
    for point in rng.uniform(lo, hi, size=10):
        point = float(point)
        assert abs(recovered.eval(point) - integrand.eval(point)) < 1e-9


def test_constant_rule():
    check_antiderivative(Const(3.0), -2.0, 2.0)


def test_power_rule():
    check_antiderivative(Pow(x, Const(2.0)), -3.0, 3.0)
    check_antiderivative(Pow(x, Const(5.0)), -2.0, 2.0)


def test_negative_power_gives_log():
    antiderivative = integrate(Pow(x, Const(-1.0)))
    assert isinstance(antiderivative, Log)
    assert abs(diff(antiderivative).eval(2.0) - 0.5) < 1e-12
    check_antiderivative(Pow(x, Const(-1.0)), 0.5, 4.0)


def test_sin_cos_exp_rules():
    check_antiderivative(Sin(x), -3.0, 3.0)
    check_antiderivative(Cos(x), -3.0, 3.0)
    check_antiderivative(Exp(x), -2.0, 2.0)


def test_sum_and_difference_rules():
    check_antiderivative(Add(Pow(x, Const(2.0)), Sin(x)), -2.0, 2.0)
    difference = Add(Exp(x), Mul(Const(-1.0), Pow(x, Const(3.0))))
    check_antiderivative(difference, -1.0, 1.0)


def test_constant_multiple_both_orders():
    check_antiderivative(Mul(Const(4.0), Pow(x, Const(3.0))), -2.0, 2.0)
    check_antiderivative(Mul(Cos(x), Const(2.5)), -3.0, 3.0)


def test_definite_integral_of_x_squared():
    value = definite_integral(Pow(x, Const(2.0)), 0.0, 1.0)
    assert abs(value - 1.0 / 3.0) < 1e-12


def test_definite_integral_of_sin():
    value = definite_integral(Sin(x), 0.0, math.pi)
    assert abs(value - 2.0) < 1e-12


def test_unsupported_product_raises_with_subtree():
    bad = Mul(x, x)
    with pytest.raises(NotImplementedError, match="x"):
        integrate(bad)


def test_unsupported_composition_raises():
    with pytest.raises(NotImplementedError):
        integrate(Sin(Pow(x, Const(2.0))))  # needs u-substitution
    with pytest.raises(NotImplementedError):
        integrate(Pow(Sin(x), Const(2.0)))


def test_fallback_matches_simpson():
    # sin(x^2) has no rule, so definite_integral routes to Simpson.
    integrand = Sin(Pow(x, Const(2.0)))
    value = definite_integral(integrand, 0.0, 2.0, panels=400)
    expected = simpson(integrand.eval, 0.0, 2.0, 400)
    assert abs(value - expected) < 1e-12


def test_implicit_diff_circle():
    # x^2 + y^2 = 25 gives dy/dx = -x/y.
    f = Add(Add(Pow(x, Const(2.0)), Pow(y, Const(2.0))), Const(-25.0))
    dydx = implicit_diff(partial(f, "x"), partial(f, "y"))
    assert abs(eval_multi(dydx, {"x": 3.0, "y": 4.0}) - (-0.75)) < 1e-12
    assert abs(eval_multi(dydx, {"x": 0.0, "y": 5.0})) < 1e-12


def test_implicit_diff_hyperbola():
    # x y = 1 gives dy/dx = -y/x.
    f = Add(Mul(x, y), Const(-1.0))
    dydx = implicit_diff(partial(f, "x"), partial(f, "y"))
    assert abs(eval_multi(dydx, {"x": 2.0, "y": 0.5}) - (-0.25)) < 1e-12
