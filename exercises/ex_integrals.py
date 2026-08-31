"""Exercise: numerical integration.

Reference implementation: calccode/integrals.py.
"""

from typing import Callable


def left_sum(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Left endpoint Riemann sum."""
    raise NotImplementedError


def midpoint_sum(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Midpoint rule. Never evaluates the endpoints."""
    raise NotImplementedError


def trapezoid(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Trapezoid rule, second order."""
    raise NotImplementedError


def simpson(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Simpson's rule, fourth order. n must be even."""
    raise NotImplementedError
