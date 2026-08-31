import math

import numpy as np
import pytest

from calccode.derivatives import central_diff
from calccode.symbolic import Add, Const, Cos, Exp, Log, Mul, Pow, Sin, Var, diff

x = Var("x")


def test_const_and_var_rules():
    assert diff(Const(5.0)).eval(0.0) == 0.0
    assert diff(x).eval(0.0) == 1.0


def test_power_rule():
    d = diff(Pow(x, Const(3.0)))
    assert abs(d.eval(2.0) - 12.0) < 1e-12


def test_product_of_cube_and_sin_matches_finite_difference():
    f = Mul(Pow(x, Const(3.0)), Sin(x))
    df = diff(f)
    rng = np.random.default_rng(7)
    for point in rng.uniform(-3.0, 3.0, size=10):
        numeric = central_diff(f.eval, float(point), 1e-5)
        assert abs(df.eval(float(point)) - numeric) < 1e-6


def test_chain_rule_on_exp_of_square():
    f = Exp(Pow(x, Const(2.0)))
    df = diff(f)
    for point in (-1.2, 0.0, 0.8):
        numeric = central_diff(f.eval, point, 1e-5)
        assert abs(df.eval(point) - numeric) < 1e-6


def test_quotient_as_negative_power():
    # sin(x) / x, differentiated without a quotient rule.
    f = Mul(Sin(x), Pow(x, Const(-1.0)))
    df = diff(f)
    for point in (0.5, 1.7, 3.1):
        numeric = central_diff(f.eval, point, 1e-5)
        assert abs(df.eval(point) - numeric) < 1e-5


def test_log_rule():
    f = Log(Mul(x, x))
    df = diff(f)
    assert abs(df.eval(2.0) - 1.0) < 1e-9  # d/dx log(x^2) = 2/x


def test_simplify_folds_constants_and_identities():
    expr = Add(Mul(Const(1.0), x), Const(0.0)).simplify()
    assert isinstance(expr, Var)
    folded = Mul(Const(3.0), Const(4.0)).simplify()
    assert isinstance(folded, Const) and folded.value == 12.0
    zeroed = Mul(Const(0.0), Sin(x)).simplify()
    assert isinstance(zeroed, Const) and zeroed.value == 0.0


def test_pow_rejects_non_constant_exponent():
    with pytest.raises(TypeError):
        Pow(x, x)  # type: ignore[arg-type]


def test_to_string():
    assert str(diff(Pow(x, Const(2.0)))) == "(2 * x)"
    assert str(Sin(x)) == "sin(x)"


def test_cos_derivative_is_negative_sin():
    df = diff(Cos(x))
    assert abs(df.eval(0.9) - (-math.sin(0.9))) < 1e-12
