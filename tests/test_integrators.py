import math

import numpy as np
import pytest

from calccode.integrators import euler, oscillator_energy, rk4, semi_implicit_euler


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
