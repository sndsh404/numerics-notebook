import math

import pytest

from calccode.related_rates import solve_rate, time_derivative
from calccode.symbolic import Add, Const, Cos, Mul, Pow, Var, eval_multi

x = Var("x")
y = Var("y")
theta = Var("theta")


def test_time_derivative_chain_rule():
    # F = x^2 + y^2 - 169 differentiates to 2x x_dt + 2y y_dt.
    f = Add(Add(Pow(x, Const(2.0)), Pow(y, Const(2.0))), Const(-169.0))
    dfdt = time_derivative(f)
    env = {"x": 5.0, "y": 12.0, "x_dt": 1.0, "y_dt": -5.0 / 12.0}
    assert abs(eval_multi(dfdt, env)) < 1e-12


def test_ladder_sliding_down_wall():
    # 13 m ladder: x^2 + y^2 = 169. The bottom slides out at 1 m/s.
    # At x = 5, y = 12 the textbook answer is dy/dt = -x/y dx/dt = -5/12.
    f = Add(Add(Pow(x, Const(2.0)), Pow(y, Const(2.0))), Const(-169.0))
    dydt = solve_rate(f, {"x": 5.0, "y": 12.0}, {"x": 1.0}, "y")
    assert abs(dydt - (-5.0 / 12.0)) < 1e-12


def test_ladder_angle_rate():
    # With x = 13 cos(theta), at x = 5, y = 12: sin(theta) = 12/13,
    # so dtheta/dt = -(dx/dt) / (13 sin(theta)) = -1/12 rad/s.
    f = Add(x, Mul(Const(-13.0), Cos(theta)))
    angle = math.acos(5.0 / 13.0)
    dthetadt = solve_rate(f, {"x": 5.0, "theta": angle}, {"x": 1.0}, "theta")
    assert abs(dthetadt - (-1.0 / 12.0)) < 1e-12


def test_expanding_circle():
    # A = pi r^2, dr/dt = 2 at r = 3: dA/dt = 2 pi r dr/dt = 12 pi.
    area = Var("A")
    r = Var("r")
    f = Add(area, Mul(Const(-math.pi), Pow(r, Const(2.0))))
    dadt = solve_rate(
        f,
        {"A": math.pi * 9.0, "r": 3.0},
        {"r": 2.0},
        "A",
    )
    assert abs(dadt - 12.0 * math.pi) < 1e-12


def test_cone_water_tank():
    # Tank with radius 2 and height 5, so r = (2/5) h and
    # V = (1/3) pi r^2 h = (4 pi / 75) h^3. Water enters at 3 m^3/min.
    # dV/dh = (4 pi / 25) h^2, so at h = 2:
    # dh/dt = 3 / (16 pi / 25) = 75 / (16 pi).
    v = Var("V")
    h = Var("h")
    k = 4.0 * math.pi / 75.0
    f = Add(v, Mul(Const(-k), Pow(h, Const(3.0))))
    dhdt = solve_rate(
        f,
        {"V": k * 8.0, "h": 2.0},
        {"V": 3.0},
        "h",
    )
    assert abs(dhdt - 75.0 / (16.0 * math.pi)) < 1e-12


def test_unknown_not_in_relation_raises():
    f = Add(Pow(x, Const(2.0)), Const(-4.0))
    with pytest.raises(ValueError, match="does not appear"):
        solve_rate(f, {"x": 2.0}, {"x": 1.0}, "y")


def test_zero_partial_raises():
    # At y = 0 the partial of x^2 + y^2 - 169 with respect to y is 0,
    # so dy/dt drops out of the equation and there is nothing to solve.
    f = Add(Add(Pow(x, Const(2.0)), Pow(y, Const(2.0))), Const(-169.0))
    with pytest.raises(ValueError, match="drops out"):
        solve_rate(f, {"x": 13.0, "y": 0.0}, {"x": 1.0}, "y")
