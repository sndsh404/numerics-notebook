"""Exercise: series.

Reference implementation: calccode/series.py.
"""

from typing import Callable


def partial_sum(term: Callable[[int], float], n: int) -> float:
    """Sum of term(k) for k = 0..n."""
    raise NotImplementedError


def geometric_series(a: float, r: float, n: int) -> float:
    """Sum of a r^k for k = 0..n."""
    raise NotImplementedError


def taylor_exp(x: float, n: int) -> float:
    """Taylor polynomial of exp around 0, terms 0..n, evaluated at x."""
    raise NotImplementedError


def taylor_sin(x: float, n: int) -> float:
    """Taylor polynomial of sin around 0 with odd terms up to degree n."""
    raise NotImplementedError


def ratio_estimate(term: Callable[[int], float], n: int = 500) -> float:
    """Estimate lim |a_{n+1} / a_n| by evaluating the ratio at large n."""
    raise NotImplementedError
