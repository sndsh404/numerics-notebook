"""Monte Carlo integration with a hand-written random number generator.

The RNG is xorshift32: three shifts and xors per draw, no dependencies,
and a fixed seed makes every result reproducible. Integration is the
mean of f over uniform samples times the domain volume. Error shrinks
like 1/sqrt(n) regardless of dimension, which is the whole appeal.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


class Xorshift32:
    """Minimal xorshift generator. Deterministic for a given seed."""

    def __init__(self, seed: int = 42):
        if seed <= 0:
            raise ValueError("seed must be a positive integer")
        self.state = seed & 0xFFFFFFFF

    def next_uint(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x

    def uniform(self) -> float:
        """Uniform on [0, 1)."""
        return self.next_uint() / 2**32

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.uniform() for _ in range(n)])


class WeakLCG:
    """RANDU: x_{n+1} = 65539 x_n mod 2^31. Deliberately bad.

    The classic cautionary generator. Consecutive triples of its output
    lie on 15 planes in the unit cube, so a 3D scatter shows sheets
    instead of a cloud. Kept for the lattice figure; never used for
    integration.
    """

    def __init__(self, seed: int = 42):
        if seed <= 0:
            raise ValueError("seed must be a positive integer")
        self.state = seed % 2**31 or 1

    def next_uint(self) -> int:
        self.state = (65539 * self.state) % 2**31
        return self.state

    def uniform(self) -> float:
        """Uniform on [0, 1)."""
        return self.next_uint() / 2**31

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.uniform() for _ in range(n)])


def mc_integrate_1d(
    f: Callable[[float], float], a: float, b: float, n: int, seed: int = 42
) -> float:
    """(b - a) times the mean of f at n uniform points in [a, b]."""
    rng = Xorshift32(seed)
    us = rng.uniforms(n)
    total = 0.0
    for u in us:
        total += f(a + (b - a) * u)
    return (b - a) * total / n


def mc_integrate_nd(
    f: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    n: int,
    seed: int = 42,
) -> float:
    """Volume times the mean of f at n uniform points in the box."""
    rng = Xorshift32(seed)
    volume = 1.0
    for lo, hi in bounds:
        volume *= hi - lo
    total = 0.0
    for _ in range(n):
        point = np.array([lo + (hi - lo) * rng.uniform() for lo, hi in bounds])
        total += f(point)
    return volume * total / n


def estimate_pi(n: int, seed: int = 42) -> float:
    """Fraction of random points in the unit square that land in the quarter circle, times 4."""
    rng = Xorshift32(seed)
    inside = 0
    for _ in range(n):
        x, y = rng.uniform(), rng.uniform()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n


def pi_error_scaling(
    ns: np.ndarray, n_seeds: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviation of pi estimates vs sample count.

    Uses several seeds per n because a single-seed error is one draw of
    a random variable; the standard deviation across seeds is what scales
    like 1/sqrt(n).
    """
    ns = np.asarray(ns, dtype=int)
    stds = np.empty(ns.size)
    for i, n in enumerate(ns):
        estimates = np.array([estimate_pi(int(n), seed=1000 + s) for s in range(n_seeds)])
        stds[i] = float(np.std(estimates))
    return ns, stds


def integrate_inv_sqrt_uniform(n: int, seed: int = 42) -> float:
    """Naive Monte Carlo for the integral of 1/sqrt(x) on (0, 1], exact value 2."""
    rng = Xorshift32(seed)
    total = 0.0
    for _ in range(n):
        u = max(rng.uniform(), 1e-300)  # stay off the singularity
        total += 1.0 / math.sqrt(u)
    return total / n


def integrate_inv_sqrt_importance(n: int, seed: int = 42) -> float:
    """Importance sampling for the same integral with p(x) = 1 / (2 sqrt(x)).

    The inverse CDF of p is x = u^2, and the integrand becomes f/p = 2,
    a constant. A constant has zero variance, so the estimate is exact
    at any n. This is the clean case that shows what a good proposal
    density buys.
    """
    rng = Xorshift32(seed)
    total = 0.0
    for _ in range(n):
        u = rng.uniform()
        x = u * u
        total += (1.0 / math.sqrt(x)) / (1.0 / (2.0 * math.sqrt(x)))
    return total / n
