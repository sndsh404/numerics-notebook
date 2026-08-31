"""Hand-written ODE integrators: explicit Euler, semi-implicit Euler, RK4.

All three advance y' = f(t, y) by a fixed step h. The stability demo that
matters for simulation work: on the harmonic oscillator, explicit Euler
adds energy every step and blows up, while semi-implicit Euler keeps the
energy bounded forever.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

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
