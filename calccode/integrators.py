"""Hand-written ODE integrators, fixed step and adaptive.

Fixed step: explicit Euler, semi-implicit Euler, RK4. The stability demo
that matters for simulation work: on the harmonic oscillator, explicit
Euler adds energy every step and blows up, while semi-implicit Euler
keeps the energy bounded forever.

Adaptive and implicit: rk45_adaptive is Dormand-Prince with an embedded
error estimate and step-size control; backward_euler is the implicit
first order method with Newton iterations and a hand-written solve per
step. On stiff problems the contrast is the lesson: the explicit
controller is pinned to a tiny step by stability, not accuracy, while
backward Euler takes whatever step the physics allows.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from calccode import linalg

RHS = Callable[[float, np.ndarray], np.ndarray]


def _integrate(f: RHS, t0: float, y0: np.ndarray, h: float, n_steps: int, step) -> tuple[np.ndarray, np.ndarray]:
    y0 = np.asarray(y0, dtype=float)
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, y0.size))
    t, y = t0, y0.copy()
    ts[0], ys[0] = t, y
    for i in range(1, n_steps + 1):
        t, y = step(f, t, y, h)
        ts[i], ys[i] = t, y
    return ts, ys


def euler(f: RHS, t0: float, y0: np.ndarray, h: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler: y_{n+1} = y_n + h f(t_n, y_n). First order."""

    def step(f: RHS, t: float, y: np.ndarray, h: float):
        return t + h, y + h * np.asarray(f(t, y), dtype=float)

    return _integrate(f, t0, y0, h, n_steps, step)


def semi_implicit_euler(f: RHS, t0: float, y0: np.ndarray, h: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Symplectic (kick-drift) Euler for y = [q..., v...].

    f must return [dq/dt..., dv/dt...]. The velocity half is updated
    first, then the position half uses the new velocity. That one change
    is what keeps the harmonic oscillator's energy bounded.
    """

    def step(f: RHS, t: float, y: np.ndarray, h: float):
        half = y.size // 2
        k = np.asarray(f(t, y), dtype=float)
        v_new = y[half:] + h * k[half:]
        q_new = y[:half] + h * v_new
        return t + h, np.concatenate([q_new, v_new])

    return _integrate(f, t0, y0, h, n_steps, step)


def rk4(f: RHS, t0: float, y0: np.ndarray, h: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Classic fourth order Runge-Kutta. Four f evaluations per step."""

    def step(f: RHS, t: float, y: np.ndarray, h: float):
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + h / 2.0, y + h * k1 / 2.0), dtype=float)
        k3 = np.asarray(f(t + h / 2.0, y + h * k2 / 2.0), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
        return t + h, y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return _integrate(f, t0, y0, h, n_steps, step)


def oscillator_energy(ys: np.ndarray) -> np.ndarray:
    """Energy 0.5 * (q^2 + v^2) of a unit harmonic oscillator trajectory."""
    q, v = ys[:, 0], ys[:, 1]
    return 0.5 * (q**2 + v**2)


# Dormand-Prince 5(4) tableau. b is the 5th order update, b_star the
# embedded 4th order one; their difference is the error estimate.
_DP_A = (
    (),
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
_DP_C = (0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)
_DP_B = _DP_A[6] + (0.0,)
_DP_B_STAR = (5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0)
_DP_ERR = tuple(b - bs for b, bs in zip(_DP_B, _DP_B_STAR))


def rk45_adaptive(
    f: RHS,
    t0: float,
    y0: np.ndarray,
    t1: float,
    tol: float = 1e-8,
    h0: float | None = None,
    max_steps: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Dormand-Prince RK with embedded error control, tol per unit step.

    Each step computes a 5th order and a 4th order update from the same
    seven f evaluations; their difference, measured against
    tol * (1 + |y|) per component, decides accept or reject and the next
    step size. Returns the accepted (t, y) history, so the step count is
    len(ts) - 1. On smooth problems a handful of steps beats thousands
    of fixed RK4 steps; on stiff problems the count explodes because
    stability, not accuracy, caps the step.
    """
    t, y = t0, np.asarray(y0, dtype=float).copy()
    h = h0 if h0 is not None else (t1 - t0) / 100.0
    h = min(h, t1 - t0)
    ts, ys = [t], [y.copy()]
    n_steps = 0
    while t < t1:
        n_steps += 1
        if n_steps > max_steps:
            raise RuntimeError(f"rk45_adaptive exceeded max_steps={max_steps} before t={t1}")
        h = min(h, t1 - t)

        ks = [np.asarray(f(t, y), dtype=float)]
        for stage in range(1, 7):
            dy = sum(a * k for a, k in zip(_DP_A[stage], ks))
            ks.append(np.asarray(f(t + _DP_C[stage] * h, y + h * dy), dtype=float))
        y5 = y + h * sum(b * k for b, k in zip(_DP_B, ks))
        err_vec = h * sum(e * k for e, k in zip(_DP_ERR, ks))
        scale = tol * (1.0 + np.abs(y5))
        err = float(np.max(np.abs(err_vec) / scale))

        if err <= 1.0:
            t, y = t + h, y5
            ts.append(t)
            ys.append(y.copy())
        if err == 0.0:
            factor = 5.0
        else:
            factor = 0.9 * err ** (-0.2)
        h *= min(5.0, max(0.2, factor))
    return np.array(ts), np.array(ys)


def _fd_jacobian(f: RHS, t: float, y: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """Central-difference Jacobian of f(t, y) in y, used when no analytic one is given."""
    n = y.size
    J = np.zeros((n, n))
    for j in range(n):
        dy = np.zeros(n)
        dy[j] = h
        J[:, j] = (np.asarray(f(t, y + dy)) - np.asarray(f(t, y - dy))) / (2.0 * h)
    return J


def backward_euler(
    f: RHS,
    jac_f: Callable[[float, np.ndarray], np.ndarray] | None,
    t_span: tuple[float, float],
    y0: np.ndarray,
    h: float,
    newton_tol: float = 1e-10,
    newton_max: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Implicit Euler: y_{n+1} = y_n + h f(t_{n+1}, y_{n+1}).

    Each step runs Newton on F(z) = z - y_n - h f(t_{n+1}, z), with
    Jacobian I - h J_f. Pass jac_f=None to fall back on central
    differences. First order, but unconditionally stable for decaying
    linear modes, which is what makes stiff problems affordable.
    """
    t0, t1 = t_span
    n_steps = int(round((t1 - t0) / h))
    t, y = t0, np.asarray(y0, dtype=float).copy()
    ts, ys = [t], [y.copy()]
    for _ in range(n_steps):
        t_next = t + h

        def F(z: np.ndarray, t_next: float = t_next, y: np.ndarray = y) -> np.ndarray:
            return z - y - h * np.asarray(f(t_next, z), dtype=float)

        z = y + h * np.asarray(f(t, y), dtype=float)  # explicit Euler guess
        for _ in range(newton_max):
            J_f = jac_f(t_next, z) if jac_f is not None else _fd_jacobian(f, t_next, z)
            A = linalg.identity(y.size) - h * np.asarray(J_f, dtype=float)
            delta = linalg.solve(A, -F(z))
            z = z + delta
            if linalg.norm(delta) < newton_tol * (1.0 + linalg.norm(z)):
                break
        else:
            raise RuntimeError(f"backward_euler Newton did not converge at t={t_next}")
        t, y = t_next, z
        ts.append(t)
        ys.append(y.copy())
    return np.array(ts), np.array(ys)
