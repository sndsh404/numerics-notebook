"""Exercise: applications of integration.

Reference implementation: calccode/applications.py.
"""

from __future__ import annotations

from typing import Callable


def arc_length(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Length of y = f(x) from a to b: integral of sqrt(1 + (f')^2).

    The test uses the straight line f(x) = 2x + 1, where the answer is
    the distance formula: (b - a) * sqrt(5).
    """
    raise NotImplementedError


def volume_disk(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Volume of the solid from rotating f about the x-axis: pi * int(f^2).

    The test uses the constant f(x) = 2, a cylinder of radius 2.
    """
    raise NotImplementedError
