import math

import numpy as np

from calccode import dynamics, integrators, linalg


def test_mass_matrix_symmetric_positive_definite():
    q = np.array([0.4, -0.9])
    M = dynamics.mass_matrix(q, m1=2.0, m2=0.5, l1=1.2, l2=0.8)
    assert np.allclose(M, linalg.transpose(M))
    assert linalg.determinant(M) > 0.0
    # Point-mass arm at q2 = 0: M11 reduces to (m1 + m2) l1^2 + m2 l2^2 + 2 m2 l1 l2.
    M0 = dynamics.mass_matrix(np.zeros(2), m1=2.0, m2=0.5, l1=1.2, l2=0.8)
    expected = 2.0 * 1.2**2 + 0.5 * (1.2**2 + 0.8**2 + 2 * 1.2 * 0.8)
    assert math.isclose(M0[0, 0], expected)


def test_zero_velocity_reduces_to_gravity():
    q = np.array([0.7, 0.2])
    qd = np.zeros(2)
    # No motion means no Coriolis terms.
    assert np.allclose(dynamics.coriolis_vector(q, qd), np.zeros(2))
    # Holding torque equal to gravity exactly cancels acceleration.
    tau = dynamics.gravity_vector(q)
    qdd = dynamics.forward_dynamics(q, qd, tau)
    assert np.allclose(qdd, np.zeros(2), atol=1e-12)
    # Without that torque the arm accelerates; straight down is equilibrium.
    qdd_free = dynamics.forward_dynamics(q, qd, np.zeros(2))
    assert np.max(np.abs(qdd_free)) > 1.0
    qdd_hanging = dynamics.forward_dynamics(np.array([math.pi / 2, 0.0]), qd, np.zeros(2))
    assert np.allclose(qdd_hanging, np.zeros(2), atol=1e-12)


def test_energy_conservation_in_gravity_free_space():
    # No gravity, no torque: total energy is pure kinetic and RK4 should keep it.
    m1, m2, l1, l2 = 2.0, 1.0, 1.0, 0.7
    q0 = np.array([0.3, -0.5])
    qd0 = np.array([0.8, -1.2])
    y0 = np.concatenate([q0, qd0])

    def rhs(t, y):
        q, qd = y[:2], y[2:]
        qdd = dynamics.forward_dynamics(q, qd, np.zeros(2), m1, m2, l1, l2, g=0.0)
        return np.concatenate([qd, qdd])

    h, n_steps = 1e-3, 2000
    ts, ys = integrators.rk4(rhs, 0.0, y0, h, n_steps)
    e0 = dynamics.kinetic_energy(y0[:2], y0[2:], m1, m2, l1, l2)
    e_final = dynamics.kinetic_energy(ys[-1, :2], ys[-1, 2:], m1, m2, l1, l2)
    assert math.isclose(e_final, e0, rel_tol=1e-6)
    # Spot check along the way, not just the endpoints.
    for i in (500, 1000, 1500):
        ei = dynamics.kinetic_energy(ys[i, :2], ys[i, 2:], m1, m2, l1, l2)
        assert math.isclose(ei, e0, rel_tol=1e-6)


def test_total_energy_conservation_with_gravity():
    # Double pendulum swinging in gravity, no torque: T + V stays constant.
    y0 = np.array([math.pi / 2 + 0.4, 0.1, 0.0, 0.0])

    def rhs(t, y):
        q, qd = y[:2], y[2:]
        qdd = dynamics.forward_dynamics(q, qd, np.zeros(2))
        return np.concatenate([qd, qdd])

    ts, ys = integrators.rk4(rhs, 0.0, y0, 1e-3, 2000)
    total0 = dynamics.kinetic_energy(y0[:2], y0[2:]) + dynamics.potential_energy(y0[:2])
    total_final = dynamics.kinetic_energy(ys[-1, :2], ys[-1, 2:]) + dynamics.potential_energy(
        ys[-1, :2]
    )
    assert math.isclose(total_final, total0, rel_tol=1e-6)
