"""Lagrangian dynamics of a 2-link planar arm.

Point masses m1 at the elbow and m2 at the tip, massless links of lengths
l1 and l2. Joint angles q = (q1, q2) measure link 1 from horizontal and
link 2 relative to link 1. Deriving the equations of motion means building
the kinetic and potential energy,

    T = (1/2) qd^T M(q) qd,   V = (m1 + m2) g l1 sin(q1) + m2 g l2 sin(q1 + q2),

and applying the Euler-Lagrange equation. The result has the standard form
M(q) qdd + C(q, qd) + g(q) = tau, with

    M11 = m1 l1^2 + m2 (l1^2 + l2^2 + 2 l1 l2 cos q2)
    M12 = M21 = m2 (l2^2 + l1 l2 cos q2)
    M22 = m2 l2^2

    C1 = -m2 l1 l2 sin(q2) (2 qd1 qd2 + qd2^2)
    C2 =  m2 l1 l2 sin(q2) qd1^2

    g1 = (m1 + m2) g l1 cos(q1) + m2 g l2 cos(q1 + q2)
    g2 = m2 g l2 cos(q1 + q2)

forward_dynamics solves for qdd given tau, using linalg.solve on M. No
numpy.linalg anywhere.
"""

from __future__ import annotations

import math

import numpy as np

from calccode import linalg

G_DEFAULT = 9.81


def mass_matrix(
    q: np.ndarray, m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0
) -> np.ndarray:
    """Configuration-dependent 2x2 mass matrix M(q)."""
    q2 = float(q[1])
    c2 = math.cos(q2)
    M = np.array(
        [
            [m1 * l1**2 + m2 * (l1**2 + l2**2 + 2 * l1 * l2 * c2), m2 * (l2**2 + l1 * l2 * c2)],
            [m2 * (l2**2 + l1 * l2 * c2), m2 * l2**2],
        ]
    )
    return M


def coriolis_vector(
    q: np.ndarray,
    qd: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
) -> np.ndarray:
    """Coriolis and centrifugal terms C(q, qd), the velocity-product terms."""
    q2 = float(q[1])
    qd1, qd2 = float(qd[0]), float(qd[1])
    h = -m2 * l1 * l2 * math.sin(q2)
    return np.array([h * (2 * qd1 * qd2 + qd2**2), -h * qd1**2])


def gravity_vector(
    q: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = G_DEFAULT,
) -> np.ndarray:
    """Gravity torque vector g(q) = dV/dq."""
    q1, q2 = float(q[0]), float(q[1])
    return np.array(
        [
            (m1 + m2) * g * l1 * math.cos(q1) + m2 * g * l2 * math.cos(q1 + q2),
            m2 * g * l2 * math.cos(q1 + q2),
        ]
    )


def kinetic_energy(
    q: np.ndarray,
    qd: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
) -> float:
    """T = (1/2) qd^T M(q) qd."""
    qd = np.asarray(qd, dtype=float)
    M = mass_matrix(q, m1, m2, l1, l2)
    return 0.5 * float(qd @ linalg.matmul(M, qd.reshape(-1, 1)).ravel())


def potential_energy(
    q: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = G_DEFAULT,
) -> float:
    """V = (m1 + m2) g l1 sin(q1) + m2 g l2 sin(q1 + q2). Zero at q = 0."""
    q1, q2 = float(q[0]), float(q[1])
    return (m1 + m2) * g * l1 * math.sin(q1) + m2 * g * l2 * math.sin(q1 + q2)


def forward_dynamics(
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = G_DEFAULT,
) -> np.ndarray:
    """Solve M(q) qdd = tau - C(q, qd) - g(q) for the joint accelerations."""
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    tau = np.asarray(tau, dtype=float)
    M = mass_matrix(q, m1, m2, l1, l2)
    rhs = tau - coriolis_vector(q, qd, m1, m2, l1, l2) - gravity_vector(q, m1, m2, l1, l2, g)
    return linalg.solve(M, rhs)
