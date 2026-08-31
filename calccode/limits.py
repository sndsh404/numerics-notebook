"""Estimate limits by sampling f at shrinking distances from a point.

No library does the math here. We evaluate f at a geometric sequence of
offsets and classify what the values do: settle, blow up, oscillate, or
hit a domain error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

Side = Literal["left", "right"]

CONVERGED_TOL = 1e-3


@dataclass
class LimitResult:
    """Outcome of a numerical limit estimate.

    value is the estimated limit when status is "converged", a signed
    infinity when status is "divergent", and NaN otherwise. samples holds
    the f values ordered from farthest to closest to the point.
    """

    value: float
    status: str  # "converged", "divergent", "oscillating", "undefined", "does not exist"
    samples: np.ndarray


def sample_sequence(
    f: Callable[[float], float],
    a: float,
    side: Side,
    n: int = 24,
    start: float = 1e-1,
    ratio: float = 0.25,
) -> np.ndarray:
    """Evaluate f at n points approaching ``a`` from one side.

    Offsets shrink geometrically from ``start`` by ``ratio`` each step,
    so the last samples sit near 1e-15 away from the point.
    """
    hs = start * ratio ** np.arange(n)
    xs = a - hs if side == "left" else a + hs
    try:
        return np.array([f(float(x)) for x in xs], dtype=float)
    except (ValueError, ZeroDivisionError, OverflowError):
        # Evaluate point by point so one bad x does not hide the rest.
        ys = []
        for x in xs:
            try:
                ys.append(f(float(x)))
            except (ValueError, ZeroDivisionError, OverflowError):
                ys.append(math.nan)
        return np.array(ys, dtype=float)


def _classify(ys: np.ndarray) -> tuple[str, float]:
    """Turn a far-to-near sample sequence into a status and value."""
    if np.any(np.isnan(ys)):
        return "undefined", math.nan
    if np.any(np.isinf(ys)):
        return "divergent", math.copysign(math.inf, ys[np.argmax(np.isinf(ys))])

    tail = ys[-5:]
    mags = np.abs(tail)
    if np.all(np.diff(mags) > 0) and mags[-1] > 10.0 * mags[0]:
        # Steady growth in magnitude as we approach: a blow-up, not noise.
        return "divergent", math.copysign(math.inf, tail[-1])

    scale = max(1.0, float(np.max(np.abs(ys))))
    spread = float(np.max(tail) - np.min(tail))
    if spread / scale < CONVERGED_TOL:
        return "converged", float(np.median(tail))
    return "oscillating", math.nan


def one_sided_limit(f: Callable[[float], float], a: float, side: Side) -> LimitResult:
    """Estimate the limit of f as x approaches ``a`` from one side."""
    ys = sample_sequence(f, a, side)
    status, value = _classify(ys)
    return LimitResult(value=value, status=status, samples=ys)


def limit(f: Callable[[float], float], a: float) -> LimitResult:
    """Estimate the two-sided limit of f at ``a``.

    Both sides must converge to the same value, or diverge to the same
    signed infinity. Anything else reports "does not exist".
    """
    left = one_sided_limit(f, a, "left")
    right = one_sided_limit(f, a, "right")
    samples = np.concatenate([left.samples, right.samples])

    if left.status == right.status == "converged":
        scale = max(1.0, abs(left.value), abs(right.value))
        if abs(left.value - right.value) < CONVERGED_TOL * scale:
            return LimitResult(0.5 * (left.value + right.value), "converged", samples)
        return LimitResult(math.nan, "does not exist", samples)
    if (
        left.status == right.status == "divergent"
        and math.copysign(1.0, left.value) == math.copysign(1.0, right.value)
    ):
        return LimitResult(left.value, "divergent", samples)
    if left.status == right.status and left.status in ("oscillating", "undefined"):
        return LimitResult(math.nan, left.status, samples)
    return LimitResult(math.nan, "does not exist", samples)
