"""Exercise: root finding.

Reference implementation: calccode/optimize.py.
"""

from typing import Callable


def bisection(f: Callable[[float], float], a: float, b: float, tol: float = 1e-10) -> float:
    """Root by repeated bracket halving. f(a) and f(b) must differ in sign."""
    raise NotImplementedError


def newton_step(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """One Newton step using a central difference for f'."""
    raise NotImplementedError


def newton(f: Callable[[float], float], x0: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """Newton's method built from newton_step."""
    raise NotImplementedError


def secant(
    f: Callable[[float], float], x0: float, x1: float, tol: float = 1e-10, max_iter: int = 50
) -> float:
    """Secant method: Newton with the slope from the last two iterates."""
    raise NotImplementedError
