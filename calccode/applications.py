"""Calc II geometry applications: arc length, volumes, surface area.

Every quantity is an integral of a known formula, evaluated with
Simpson's rule from integrals.py. The derivatives inside the arc
length and surface area integrands come from central_diff in
derivatives.py. Nothing here is symbolic; f is a plain Python
callable.

The endpoint caveat: for a curve like the semicircle sqrt(r^2 - x^2)
the derivative blows up at the endpoints, and central_diff evaluates
f slightly outside the domain, which is a math domain error for
sqrt. The honest fix is to shrink the interval by a small epsilon and
compare against the closed form for the shrunk arc; the tests show
the pattern.
"""

from __future__ import annotations

import math
from typing import Callable

from calccode.derivatives import central_diff
from calccode.integrals import simpson


def arc_length(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Length of y = f(x) from a to b: integral of sqrt(1 + (f')^2)."""
    def integrand(x: float) -> float:
        slope = central_diff(f, x)
        return math.sqrt(1.0 + slope * slope)

    return simpson(integrand, a, b, n)


def volume_disk(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Volume of the solid from rotating f about the x-axis: pi * int(f^2)."""
    return math.pi * simpson(lambda x: f(x) ** 2, a, b, n)


def volume_shell(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Volume of the solid from rotating f about the y-axis: 2 pi * int(x f)."""
    return 2.0 * math.pi * simpson(lambda x: x * f(x), a, b, n)


def surface_area(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Area of the surface from rotating f about the x-axis.

    Integrand is 2 pi f sqrt(1 + (f')^2); same endpoint derivative
    caveat as arc_length.
    """
    def integrand(x: float) -> float:
        slope = central_diff(f, x)
        return f(x) * math.sqrt(1.0 + slope * slope)

    return 2.0 * math.pi * simpson(integrand, a, b, n)
