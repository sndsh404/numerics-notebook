
import numpy as np

from calccode import ode_systems


def test_pendulum_period_matches_small_angle_theory():
    measured = ode_systems.pendulum_period_numerical(theta0=0.05)
    theory = ode_systems.pendulum_period_small_angle()
    assert abs(measured - theory) / theory < 1e-3


def test_pendulum_large_amplitude_has_longer_period():
    # The exact period grows with amplitude; at 1.0 rad it is about 6% over.
    small = ode_systems.pendulum_period_numerical(theta0=0.05)
    large = ode_systems.pendulum_period_numerical(theta0=1.0)
    assert 1.03 < large / small < 1.10


def test_lotka_volterra_conserves_invariant():
    f = ode_systems.lotka_volterra_rhs()
    ts, ys = ode_systems.rk4_system(f, 0.0, np.array([10.0, 5.0]), h=0.005, n_steps=10000)
    H = ode_systems.lotka_volterra_invariant(ys)
    spread = (H.max() - H.min()) / abs(H.mean())
    assert spread < 1e-4


def test_lotka_volterra_orbit_closes():
    # After one full oscillation the populations return near the start.
    f = ode_systems.lotka_volterra_rhs()
    ts, ys = ode_systems.rk4_system(f, 0.0, np.array([10.0, 5.0]), h=0.002, n_steps=4000)
    prey = ys[:, 0]
    # Find the first return of prey to its start value after leaving it.
    for i in range(10, len(ts)):
        if prey[i - 1] < 10.0 <= prey[i]:
            dist = np.linalg.norm(ys[i] - ys[0])
            assert dist < 0.5
            return
    raise AssertionError("orbit did not return within the integration window")


def test_lorenz_stays_bounded():
    f = ode_systems.lorenz_rhs()
    ts, ys = ode_systems.rk4_system(f, 0.0, np.array([1.0, 1.0, 1.0]), h=0.005, n_steps=4000)
    assert np.all(np.isfinite(ys))
    assert np.abs(ys).max() < 100.0


def test_lorenz_nearby_starts_diverge():
    f = ode_systems.lorenz_rhs()
    y0 = np.array([1.0, 1.0, 1.0])
    ts, a = ode_systems.rk4_system(f, 0.0, y0, h=0.005, n_steps=6000)
    _, b = ode_systems.rk4_system(f, 0.0, y0 + np.array([1e-6, 0.0, 0.0]), h=0.005, n_steps=6000)
    # t = 30 of chaotic amplification turns 1e-6 into order-one separation.
    assert np.linalg.norm(a[-1] - b[-1]) > 1.0


def test_lorenz_divergence_grows_from_tiny_seed():
    f = ode_systems.lorenz_rhs()
    y0 = np.array([1.0, 1.0, 1.0])
    for n_steps in (5000, 6000):
        _, a = ode_systems.rk4_system(f, 0.0, y0, h=0.005, n_steps=n_steps)
        _, b = ode_systems.rk4_system(f, 0.0, y0 + np.array([1e-6, 0.0, 0.0]), h=0.005, n_steps=n_steps)
        if n_steps == 5000:
            assert np.linalg.norm(a[-1] - b[-1]) < 1.0
        else:
            assert np.linalg.norm(a[-1] - b[-1]) > 1.0
