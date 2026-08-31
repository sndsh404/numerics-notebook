import math

import numpy as np
import pytest

from calccode.integrators import (
    backward_euler,
    euler,
    oscillator_energy,
    rk4,
    rk45_adaptive,
    semi_implicit_euler,
)


def decay(k: float):
    return lambda t, y: -k * y


def oscillator(t, y):
    return np.array([y[1], -y[0]])


def test_euler_on_exponential_decay():
    ts, ys = euler(decay(2.0), 0.0, np.array([1.0]), h=0.05, n_steps=20)
    assert abs(ys[-1, 0] - math.exp(-2.0)) < 2e-2


def test_rk4_far_more_accurate_than_euler_at_same_steps():
    h, n = 0.05, 20
    exact = math.exp(-2.0)
    euler_err = abs(euler(decay(2.0), 0.0, np.array([1.0]), h, n)[1][-1, 0] - exact)
    rk4_err = abs(rk4(decay(2.0), 0.0, np.array([1.0]), h, n)[1][-1, 0] - exact)
    assert rk4_err < euler_err / 1000.0


def test_rk4_matches_exact_solution_closely():
    ts, ys = rk4(decay(1.0), 0.0, np.array([1.0]), h=0.1, n_steps=10)
    assert abs(ys[-1, 0] - math.exp(-1.0)) < 1e-6


def test_explicit_euler_blows_up_harmonic_oscillator():
    ts, ys = euler(oscillator, 0.0, np.array([1.0, 0.0]), h=0.05, n_steps=10000)
    energy = oscillator_energy(ys)
    # (1 + h^2) growth per step compounds to a factor of e^12 over 10000 steps.
    assert energy[-1] > 10.0 * energy[0]


def test_semi_implicit_euler_keeps_energy_bounded():
    ts, ys = semi_implicit_euler(oscillator, 0.0, np.array([1.0, 0.0]), h=0.05, n_steps=10000)
    energy = oscillator_energy(ys)
    assert energy.max() < 1.5 * energy[0]
    assert energy.min() > 0.5 * energy[0]


def test_rk4_harmonic_oscillator_energy_nearly_constant():
    ts, ys = rk4(oscillator, 0.0, np.array([1.0, 0.0]), h=0.05, n_steps=10000)
    energy = oscillator_energy(ys)
    assert abs(energy[-1] - energy[0]) < 1e-5 * energy[0]


def test_integrators_return_full_history():
    ts, ys = euler(decay(1.0), 0.0, np.array([1.0]), h=0.1, n_steps=10)
    assert ts.shape == (11,)
    assert ys.shape == (11, 1)
    assert ts[-1] == pytest.approx(1.0)


def test_rk45_hits_exponential_decay_within_tol():
    ts, ys = rk45_adaptive(decay(1.0), 0.0, np.array([1.0]), 5.0, tol=1e-9)
    assert abs(ys[-1, 0] - math.exp(-5.0)) < 1e-7
    assert ts[-1] == pytest.approx(5.0)


def test_rk45_hits_oscillator_within_tol():
    ts, ys = rk45_adaptive(oscillator, 0.0, np.array([1.0, 0.0]), 20.0, tol=1e-9)
    assert abs(ys[-1, 0] - math.cos(20.0)) < 1e-6
    assert abs(ys[-1, 1] + math.sin(20.0)) < 1e-6


def test_rk45_takes_far_fewer_steps_than_rk4_at_same_accuracy():
    f = decay(1.0)
    ts, ys = rk45_adaptive(f, 0.0, np.array([1.0]), 5.0, tol=1e-9)
    adaptive_steps = ts.size - 1
    adaptive_err = abs(ys[-1, 0] - math.exp(-5.0))
    # Fixed RK4 at 100 steps over the same span is less accurate than the
    # adaptive run, which used roughly half as many steps.
    n_rk4 = 100
    _, ys4 = rk4(f, 0.0, np.array([1.0]), h=5.0 / n_rk4, n_steps=n_rk4)
    assert abs(ys4[-1, 0] - math.exp(-5.0)) > adaptive_err
    assert adaptive_steps < n_rk4 * 0.75


def van_der_pol(mu: float):
    def f(t, y):
        return np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])

    def jac(t, y):
        return np.array(
            [[0.0, 1.0], [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] ** 2)]]
        )

    return f, jac


def test_stiff_van_der_pol_explicit_vs_implicit():
    # mu = 1000 is stiff: the fast mode is long dead, but explicit RK45
    # stays pinned to a stability-limited step (measured: 86605 accepted
    # steps over t in [0, 100] at tol = 1e-4). Backward Euler at h = 0.05
    # covers the same span in 2000 steps and stays bounded.
    f, jac = van_der_pol(1000.0)
    ts, ys = rk45_adaptive(f, 0.0, np.array([2.0, 0.0]), 100.0, tol=1e-4)
    assert ts.size - 1 > 20000
    assert np.max(np.abs(ys)) <= 2.0 + 1e-6

    ts_be, ys_be = backward_euler(f, jac, (0.0, 100.0), np.array([2.0, 0.0]), h=0.05)
    assert ts_be.size - 1 == 2000
    assert ts_be.size - 1 < (ts.size - 1) / 20
    assert np.all(np.isfinite(ys_be))
    assert np.max(np.abs(ys_be)) < 10.0


def test_backward_euler_matches_closed_form_on_linear_decay():
    # On y' = -k y with analytic Jacobian, Newton is exact in one
    # iteration, so the result is (1 + k h)^-n to roundoff.
    k, h, n = 2.0, 0.01, 100
    f = lambda t, y: -k * y
    jac = lambda t, y: np.array([[-k]])
    ts, ys = backward_euler(f, jac, (0.0, n * h), np.array([1.0]), h=h)
    assert abs(ys[-1, 0] - (1.0 + k * h) ** -n) < 1e-10


def test_backward_euler_without_jacobian_uses_finite_differences():
    k, h, n = 2.0, 0.01, 100
    f = lambda t, y: -k * y
    ts, ys = backward_euler(f, None, (0.0, n * h), np.array([1.0]), h=h)
    assert abs(ys[-1, 0] - (1.0 + k * h) ** -n) < 1e-8


def test_backward_euler_first_order_on_decay():
    # Halving h roughly halves the global error against e^{-2}.
    k = 2.0
    f = lambda t, y: -k * y
    jac = lambda t, y: np.array([[-k]])
    _, a = backward_euler(f, jac, (0.0, 1.0), np.array([1.0]), h=0.01)
    _, b = backward_euler(f, jac, (0.0, 1.0), np.array([1.0]), h=0.005)
    exact = math.exp(-2.0)
    err_a = abs(a[-1, 0] - exact)
    err_b = abs(b[-1, 0] - exact)
    assert 1.5 < err_a / err_b < 2.5
