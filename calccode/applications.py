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
the pattern. The cleaner fix is a change of variable:
arc_length_trig_circle substitutes x = r sin(t) and the singularity
disappears from the integral altogether.
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


def arc_length_trig_circle(r: float, n: int = 200) -> float:
    """Length of the semicircle y = sqrt(r^2 - x^2) by trig substitution.

    With x = r sin(t), dx = r cos(t) dt and sqrt(1 + (f')^2) dx folds
    to r dt, with t on [-pi/2, pi/2]. The endpoint singularity that
    forces arc_length onto a shrunk interval is gone: the new integrand
    is the constant r, which Simpson integrates exactly, so the answer
    is pi r to machine precision.
    """
    return simpson(lambda t: r, -math.pi / 2.0, math.pi / 2.0, n)


def arc_length_parametric(
    x: Callable[[float], float],
    y: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1000,
) -> float:
    """Length of the parametric curve (x(t), y(t)) from a to b.

    Integral of sqrt((dx/dt)^2 + (dy/dt)^2). No endpoint singularities
    as long as x and y are smooth, which is why a circle parametrized
    by angle is easier than the same circle as y = f(x).
    """
    def integrand(t: float) -> float:
        dx = central_diff(x, t)
        dy = central_diff(y, t)
        return math.sqrt(dx * dx + dy * dy)

    return simpson(integrand, a, b, n)


def arc_length_polar(r: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Length of the polar curve r = r(theta) from a to b.

    Writing x = r cos(theta), y = r sin(theta) and differentiating gives
    the integrand sqrt(r^2 + (dr/dtheta)^2).
    """
    def integrand(theta: float) -> float:
        r_val = r(theta)
        dr = central_diff(r, theta)
        return math.sqrt(r_val * r_val + dr * dr)

    return simpson(integrand, a, b, n)
