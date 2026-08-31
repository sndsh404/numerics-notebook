"""Tests for the exercises/ folder.

Skipped by default so the main suite stays green while exercises are
unsolved. Run with: python -m pytest tests/test_exercises.py --run-exercises
"""

import math

import numpy as np
import pytest

from exercises import (
    ex_autograd,
    ex_derivatives,
    ex_integrals,
    ex_linalg,
    ex_limits,
    ex_optimize,
    ex_series,
    ex_transforms,
)

pytestmark = pytest.mark.usefixtures("_require_exercises_flag")


@pytest.fixture(autouse=True)
def _require_exercises_flag(request):
    if not request.config.getoption("--run-exercises"):
        pytest.skip("pass --run-exercises to run these")


class TestLimits:
    def test_sample_near_approaches_point(self):
        ys = ex_limits.sample_near(math.exp, 1.0, "right")
        assert len(ys) == 24
        assert abs(ys[-1] - math.e) < abs(ys[0] - math.e)

    def test_one_sided_limit_of_sin_x_over_x(self):
        assert abs(ex_limits.one_sided_limit(lambda x: math.sin(x) / x, 0.0, "right") - 1.0) < 1e-3

    def test_oscillation_detection(self):
        assert ex_limits.is_oscillating(lambda x: math.sin(1.0 / x), 0.0)
        assert not ex_limits.is_oscillating(lambda x: math.sin(x) / x, 0.0)

    def test_two_sided_limit(self):
        assert abs(ex_limits.two_sided_limit(lambda x: x * x + 3 * x, 2.0) - 10.0) < 1e-6
        assert ex_limits.two_sided_limit(lambda x: 1.0 / x, 0.0) is None


class TestDerivatives:
    def test_forward_and_central(self):
        assert abs(ex_derivatives.forward_diff(math.exp, 1.0) - math.e) < 1e-4
        assert abs(ex_derivatives.central_diff(math.sin, 0.7) - math.cos(0.7)) < 1e-8

    def test_second_derivative(self):
        assert abs(ex_derivatives.second_derivative(math.sin, 0.9) + math.sin(0.9)) < 1e-6

    def test_order_estimation(self):
        hs = [10 ** (-k) for k in range(1, 5)]
        errs = ex_derivatives.errors_vs_h(math.sin, 1.0, math.cos(1.0), hs)
        slope = ex_derivatives.estimate_order(hs, errs)
        assert 1.9 < slope < 2.1


class TestIntegrals:
    def test_rules_on_sin(self):
        assert abs(ex_integrals.left_sum(math.sin, 0.0, math.pi, 10000) - 2.0) < 1e-3
        assert abs(ex_integrals.midpoint_sum(math.sin, 0.0, math.pi, 1000) - 2.0) < 1e-5
        assert abs(ex_integrals.trapezoid(math.sin, 0.0, math.pi, 1000) - 2.0) < 5e-6

    def test_simpson_exact_for_cubic(self):
        f = lambda x: x**3 - 2.0 * x**2 + x - 4.0
        exact = 81.0 / 4 - 18.0 + 4.5 - 12.0
        assert abs(ex_integrals.simpson(f, 0.0, 3.0, 2) - exact) < 1e-12


class TestSeries:
    def test_partial_sum(self):
        assert ex_series.partial_sum(lambda k: 1.0, 9) == 10.0

    def test_geometric_series(self):
        assert abs(ex_series.geometric_series(1.0, 0.5, 30) - 2.0) < 1e-8

    def test_taylor_exp(self):
        assert abs(ex_series.taylor_exp(1.0, 12) - math.e) < 1e-9

    def test_taylor_sin(self):
        assert abs(ex_series.taylor_sin(0.5, 11) - math.sin(0.5)) < 1e-9

    def test_ratio_estimate(self):
        assert abs(ex_series.ratio_estimate(lambda k: 0.5**k) - 0.5) < 1e-9
        assert abs(ex_series.ratio_estimate(lambda k: 1.0 / math.factorial(k))) < 0.01


class TestLinalg:
    def test_matmul(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        assert np.allclose(ex_linalg.matmul(A, B), [[19.0, 22.0], [43.0, 50.0]])

    def test_transpose(self):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.allclose(ex_linalg.transpose(A), A.T)

    def test_determinant(self):
        assert abs(ex_linalg.determinant(np.array([[2.0, 1.0], [5.0, 3.0]])) - 1.0) < 1e-12

    def test_solve(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        x = ex_linalg.solve(A, np.array([9.0, 8.0]))
        assert np.allclose(x, [2.0, 3.0])

    def test_rank(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 1.0]])
        assert ex_linalg.rank(A) == 2


class TestTransforms:
    def test_rot2(self):
        assert np.allclose(ex_transforms.rot2(math.pi / 2) @ np.array([1.0, 0.0]), [0.0, 1.0], atol=1e-12)

    def test_rotz(self):
        assert np.allclose(ex_transforms.rotz(math.pi / 2) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)

    def test_axis_angle_matches_rotz(self):
        R = ex_transforms.rot_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.7)
        assert np.allclose(R, ex_transforms.rotz(0.7), atol=1e-12)

    def test_invert_transform(self):
        T = np.eye(4)
        T[:3, :3] = ex_transforms.rotz(0.5)
        T[:3, 3] = [1.0, 2.0, 3.0]
        assert np.allclose(T @ ex_transforms.invert_transform(T), np.eye(4), atol=1e-12)

    def test_planar_arm_fk(self):
        joints = ex_transforms.planar_arm_fk(0.0, 0.0, 2.0, 1.0)
        assert np.allclose(joints[-1], [3.0, 0.0])


class TestAutograd:
    def test_relu(self):
        assert ex_autograd.relu(-2.0) == 0.0
        assert ex_autograd.relu(3.0) == 3.0

    def test_tanh_grad(self):
        assert abs(ex_autograd.tanh_grad(0.5) - (1.0 - math.tanh(0.5) ** 2)) < 1e-12

    def test_neuron_output(self):
        got = ex_autograd.neuron_output([1.0, -1.0], [2.0, 0.5], 0.25)
        assert abs(got - math.tanh(1.75)) < 1e-12

    def test_mse(self):
        assert abs(ex_autograd.mse_loss([1.0, 3.0], [0.0, 0.0]) - 5.0) < 1e-12
        grads = ex_autograd.mse_grad([1.0, 3.0], [0.0, 0.0])
        assert np.allclose(grads, [1.0, 3.0])

    def test_gradient_step(self):
        assert ex_autograd.gradient_step([1.0, 2.0], [0.5, 0.5], 2.0) == [0.0, 1.0]


class TestOptimize:
    def test_bisection(self):
        root = ex_optimize.bisection(lambda x: x * x - 2.0, 1.0, 2.0)
        assert abs(root - math.sqrt(2.0)) < 1e-9

    def test_newton(self):
        root = ex_optimize.newton(lambda x: x * x - 2.0, 1.5)
        assert abs(root - math.sqrt(2.0)) < 1e-9

    def test_secant(self):
        root = ex_optimize.secant(lambda x: x * x - 2.0, 1.0, 2.0)
        assert abs(root - math.sqrt(2.0)) < 1e-9

    def test_newton_step_moves_toward_root(self):
        f = lambda x: x * x - 2.0
        x1 = ex_optimize.newton_step(f, 2.0)
        assert abs(f(x1)) < abs(f(2.0))
