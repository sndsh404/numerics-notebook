"""Root finding: bisection, Newton, and secant methods.

Newton uses the hand-written central difference from derivatives.py for
f', so nothing here needs an analytic derivative. Every method returns a
RootResult with the full x history, which makes convergence order
measurable instead of assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from calccode.derivatives import central_diff


@dataclass
class RootResult:
    root: float
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> RootResult:
    """Bracketed root finding. Halves the interval every step.

    f must change sign on [a, b]; that sign change is the entire
    guarantee. The error after n steps is (b - a) / 2^n.
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return RootResult(a, 0, True, [a])
    if fb == 0.0:
        return RootResult(b, 0, True, [b])
    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    history = []
    for i in range(1, max_iter + 1):
        mid = 0.5 * (a + b)
        fm = f(mid)
        history.append(mid)
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return RootResult(mid, i, True, history)
        if fa * fm < 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return RootResult(0.5 * (a + b), max_iter, False, history)


def newton(
    f: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 50,
    h: float = 1e-6,
) -> RootResult:
    """Newton's method with a central difference derivative.

    Quadratic convergence near a simple root, but no guarantee anywhere
    else: a flat or cycling start point can send the iterates off to
    infinity. converged is False in that case.
    """
    x = x0
    history = [x0]
    for i in range(1, max_iter + 1):
        fx = f(x)
        if abs(fx) < tol:
            return RootResult(x, i - 1, True, history)
        dfx = central_diff(f, x, h)
        if dfx == 0.0 or not math.isfinite(dfx):
            return RootResult(x, i - 1, False, history)
        x = x - fx / dfx
        history.append(x)
        if not math.isfinite(x) or abs(x) > 1e12:
            return RootResult(x, i, False, history)
    return RootResult(x, max_iter, False, history)


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> RootResult:
    """Secant method: Newton with the slope from the last two iterates.

    Superlinear with order about 1.618, no derivative needed at all.
    """
    history = [x0, x1]
    f0, f1 = f(x0), f(x1)
    for i in range(1, max_iter + 1):
        if abs(f1) < tol:
            return RootResult(x1, i, True, history)
        denom = f1 - f0
        if denom == 0.0:
            return RootResult(x1, i, False, history)
        x2 = x1 - f1 * (x1 - x0) / denom
        history.append(x2)
        if not math.isfinite(x2) or abs(x2) > 1e12:
            return RootResult(x2, i, False, history)
        x0, x1, f0, f1 = x1, x2, f1, f(x2)
    return RootResult(x1, max_iter, False, history)


def compare_convergence(
    f: Callable[[float], float],
    bracket: tuple[float, float],
    x0: float,
    tol: float = 1e-10,
) -> dict[str, RootResult]:
    """Run all three methods on the same root and compare iteration counts."""
    return {
        "bisection": bisection(f, bracket[0], bracket[1], tol),
        "newton": newton(f, x0, tol),
        "secant": secant(f, bracket[0], bracket[1], tol),
    }
