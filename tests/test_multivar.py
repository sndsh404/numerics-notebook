import math

import numpy as np
import pytest

from calccode import multivar
from calccode.symbolic import Const, Exp, Mul, Pow, Sin, Var

x, y = Var("x"), Var("y")


def test_partial_diff_matches_analytic():
    f = lambda v: v[0] ** 2 * v[1] + math.sin(v[1])
    # df/dx = 2xy at (1, 0.5)
    assert abs(multivar.partial_diff(f, np.array([1.0, 0.5]), 0) - 1.0) < 1e-7
    # df/dy = x^2 + cos(y) at (1, 0.5)
    want = 1.0 + math.cos(0.5)
    assert abs(multivar.partial_diff(f, np.array([1.0, 0.5]), 1) - want) < 1e-7


def test_gradient_on_quadratic():
    f = lambda v: (v[0] - 1.0) ** 2 + 3.0 * (v[1] + 2.0) ** 2
    g = multivar.gradient(f, np.array([3.0, 4.0]))
    assert np.allclose(g, [4.0, 36.0], atol=1e-4)


def test_jacobian_of_vector_function():
    F = lambda v: np.array([v[0] ** 2 + v[1], v[0] * v[1]])
    J = multivar.jacobian(F, np.array([2.0, 3.0]))
    assert np.allclose(J, [[4.0, 1.0], [3.0, 2.0]], atol=1e-5)


def test_hessian_of_quadratic():
    f = lambda v: v[0] ** 2 + v[0] * v[1] + v[1] ** 2
    H = multivar.hessian(f, np.array([0.7, -1.2]))
    assert np.allclose(H, [[2.0, 1.0], [1.0, 2.0]], atol=1e-4)


def test_hessian_is_symmetric_on_smooth_function():
    f = lambda v: math.sin(v[0]) * math.exp(v[1]) + v[0] ** 3 * v[1]
    H = multivar.hessian(f, np.array([0.4, 0.9]))
    assert abs(H[0, 1] - H[1, 0]) < 1e-6


def test_directional_derivative_matches_gradient_dot_unit():
    f = lambda v: v[0] ** 2 + 2.0 * v[1] ** 2
    point = np.array([1.0, 2.0])
    direction = np.array([3.0, 4.0])
    got = multivar.directional_derivative(f, point, direction)
    want = np.array([2.0, 8.0]) @ (direction / 5.0)
    assert abs(got - want) < 1e-4


def test_directional_derivative_rejects_zero_direction():
    with pytest.raises(ValueError):
        multivar.directional_derivative(lambda v: v[0], np.array([1.0]), np.array([0.0]))


def test_gradient_check_product_with_sin():
    expr = Mul(Pow(x, Const(2.0)), Sin(y))
    err = multivar.gradient_check(expr, ["x", "y"], np.array([1.3, 0.8]))
    assert err < 1e-5


def test_gradient_check_exp_of_product():
    expr = Exp(Mul(x, y))
    err = multivar.gradient_check(expr, ["x", "y"], np.array([0.7, -1.1]))
    assert err < 1e-5


def test_gradient_check_three_variables():
    z = Var("z")
    expr = Mul(x, y) + Mul(y, z) + Mul(z, x)
    err = multivar.gradient_check(expr, ["x", "y", "z"], np.array([1.0, 2.0, 3.0]))
    assert err < 1e-5


def test_gradient_check_detects_wrong_point():
    # Checking at a different point than the one evaluated should fail.
    expr = Mul(Pow(x, Const(2.0)), Sin(y))
    err = multivar.gradient_check(expr, ["x", "y"], np.array([1.3, 0.8]))
    err_other = multivar.gradient_check(expr, ["x", "y"], np.array([1.3001, 0.8]))
    assert err_other > err
