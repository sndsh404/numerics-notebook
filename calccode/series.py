"""Taylor polynomials, partial sums, and the ratio test.

Coefficients come from the machinery already in this repo: exact ones
by repeatedly differentiating an expression tree from symbolic.py, or
numeric ones by recursive central differences from derivatives.py.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from calccode.derivatives import central_diff
from calccode.symbolic import Expr, diff


def taylor_coefficients_from_expr(expr: Expr, a: float, n: int) -> list[float]:
    """Exact coefficients c_k = f^(k)(a) / k! from a symbolic tree."""
    coeffs = []
    current = expr
    for k in range(n + 1):
        coeffs.append(current.eval(a) / math.factorial(k))
        current = diff(current)
    return coeffs


def _nth_derivative(f: Callable[[float], float], x: float, n: int, h: float) -> float:
    """Central difference applied n times. Cost is 2^n evaluations."""
    if n == 0:
        return f(x)
    return central_diff(lambda t: _nth_derivative(f, t, n - 1, h), x, h)


def taylor_coefficients_numeric(
    f: Callable[[float], float], a: float, n: int, h: float = 1e-3
) -> list[float]:
    """Coefficients from recursive finite differences.

    Roundoff compounds with each differentiation level, so keep n
    small (5 or 6) and prefer the symbolic path when a tree is available.
    """
    return [
        _nth_derivative(f, a, k, h) / math.factorial(k) for k in range(n + 1)
    ]


def taylor_polynomial(coeffs: list[float], a: float) -> Callable[[float], float]:
    """Build the polynomial sum c_k (x - a)^k, evaluated by Horner's method."""
    def p(x: float) -> float:
        total = 0.0
        for c in reversed(coeffs):
            total = total * (x - a) + c
        return total

    return p


def taylor_error(
    f: Callable[[float], float],
    poly: Callable[[float], float],
    xs: np.ndarray,
) -> np.ndarray:
    """Absolute error |f(x) - P(x)| at each point in xs."""
    return np.array([abs(f(float(x)) - poly(float(x))) for x in xs])


def partial_sum(term: Callable[[int], float], n: int) -> float:
    """Sum of term(k) for k = 0..n."""
    return sum(term(k) for k in range(n + 1))


def partial_sum_history(term: Callable[[int], float], n: int) -> np.ndarray:
    """Cumulative sums S_0 .. S_n, for watching a series settle."""
    out = np.empty(n + 1)
    total = 0.0
    for k in range(n + 1):
        total += term(k)
        out[k] = total
    return out


def ratio_test(term: Callable[[int], float], n0: int = 100, n1: int = 800) -> tuple[float, str]:
    """Numeric ratio test: estimate L = lim |a_{n+1} / a_n|.

    Ratios are sampled at n0 and n1. A ratio that is decreasing (or
    steady) and sits well under 1 means the series converges. A ratio
    well above 1 and not decreasing means it diverges. A ratio drifting
    up toward 1 is the classic inconclusive case: the test says nothing
    about series like 1/k, whose ratios also approach 1 from below.
    """
    pairs = []
    for n in (n0, n1):
        a_n, a_next = term(n), term(n + 1)
        if a_n == 0.0:
            raise ValueError("term is zero, ratio undefined")
        pairs.append(abs(a_next / a_n))
    r0, r1 = pairs
    if r1 > 1.005 and r1 >= r0 - 1e-12:
        return r1, "diverges"
    if r1 < 0.995 and r1 <= r0 + 1e-12:
        return r1, "converges"
    return r1, "inconclusive"
