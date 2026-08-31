"""Exercise: finite difference derivatives.

Reference implementation: calccode/derivatives.py.
"""

from typing import Callable


def forward_diff(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """First order forward difference."""
    raise NotImplementedError


def central_diff(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Second order central difference."""
    raise NotImplementedError


def second_derivative(f: Callable[[float], float], x: float, h: float = 1e-4) -> float:
    """Central second difference, error O(h^2)."""
    raise NotImplementedError


def errors_vs_h(
    f: Callable[[float], float], x: float, exact: float, hs: list[float]
) -> list[float]:
    """Absolute central difference error at each step size in hs."""
    raise NotImplementedError


def estimate_order(hs: list[float], errors: list[float]) -> float:
    """Slope of log(error) vs log(h): the empirical order of accuracy."""
    raise NotImplementedError
