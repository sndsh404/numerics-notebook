"""Probability and statistics with everything hand-written.

Sampling reuses the xorshift32 generator from montecarlo.py: uniforms come
straight from the generator, exponentials come from the inverse CDF
x = -ln(1 - u) / lam, and normals come from Box-Muller. The normal CDF is
Simpson's rule on the PDF, no math.erf anywhere. Means, variances, and
percentiles are plain loops plus one sorted() call; sorting is bookkeeping,
so the built-in timsort is allowed while the math stays hand-written.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from calccode.integrals import simpson
from calccode.montecarlo import Xorshift32


# Sampling


def uniform_samples(n: int, seed: int = 42, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """n uniforms on [a, b) from the hand-written xorshift32."""
    rng = Xorshift32(seed)
    return np.array([a + (b - a) * rng.uniform() for _ in range(n)])


def exponential_samples(n: int, lam: float, seed: int = 42) -> np.ndarray:
    """n exponential draws by inverse transform: x = -ln(1 - u) / lam."""
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    rng = Xorshift32(seed)
    out = np.empty(n)
    for i in range(n):
        u = max(rng.uniform(), 1e-300)  # stay off log(0)
        out[i] = -math.log(1.0 - u) / lam
    return out


def normal_samples(n: int, seed: int = 42, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """n normal draws from Box-Muller, two uniforms in, two normals out."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    rng = Xorshift32(seed)
    out: list[float] = []
    while len(out) < n:
        u1 = max(rng.uniform(), 1e-300)
        u2 = rng.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        out.append(mu + sigma * r * math.cos(2.0 * math.pi * u2))
        if len(out) < n:
            out.append(mu + sigma * r * math.sin(2.0 * math.pi * u2))
    return np.array(out)


# Densities and cumulative probabilities


def uniform_pdf(x: float, a: float = 0.0, b: float = 1.0) -> float:
    """Constant 1 / (b - a) on [a, b], zero outside."""
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def uniform_cdf(x: float, a: float = 0.0, b: float = 1.0) -> float:
    if x < a:
        return 0.0
    if x > b:
        return 1.0
    return (x - a) / (b - a)


def exponential_pdf(x: float, lam: float) -> float:
    """lam * e^(-lam x) for x >= 0, zero for negative x."""
    if x < 0.0:
        return 0.0
    return lam * math.exp(-lam * x)


def exponential_cdf(x: float, lam: float) -> float:
    """1 - e^(-lam x), exact in closed form."""
    if x < 0.0:
        return 0.0
    return 1.0 - math.exp(-lam * x)


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0, panels: int = 400) -> float:
    """Phi by Simpson's rule on the PDF.

    Standardize to z = (x - mu) / sigma, then 0.5 plus the integral of the
    standard normal density from 0 to z. The symmetry trick keeps the
    integration interval short and away from the far tail.
    """
    z = (x - mu) / sigma
    standard = lambda t: math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    if z >= 0.0:
        return 0.5 + simpson(standard, 0.0, z, panels)
    return 0.5 - simpson(standard, z, 0.0, panels)


# Sample statistics


def sample_mean(xs: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=float)
    total = 0.0
    for x in xs:
        total += x
    return total / xs.size


def sample_variance(xs: np.ndarray) -> float:
    """Unbiased sample variance: sum of squared deviations over n - 1."""
    xs = np.asarray(xs, dtype=float)
    if xs.size < 2:
        raise ValueError("variance needs at least two samples")
    mu = sample_mean(xs)
    total = 0.0
    for x in xs:
        total += (x - mu) ** 2
    return total / (xs.size - 1)


def sample_std(xs: np.ndarray) -> float:
    return math.sqrt(sample_variance(xs))


def median(xs: np.ndarray) -> float:
    """Middle value of the sorted data, or the mean of the middle two."""
    s = sorted(float(x) for x in xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def percentile(xs: np.ndarray, p: float) -> float:
    """p-th percentile by linear interpolation on the sorted data.

    The rank is (p / 100) * (n - 1); when it lands between two sorted
    values we interpolate. So the 50th percentile of an even-length list
    is the mean of the middle two, matching median above.
    """
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must lie in [0, 100]")
    s = sorted(float(x) for x in xs)
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


# Expectations by quadrature


def expected_value(pdf: Callable[[float], float], a: float, b: float, panels: int = 2000) -> float:
    """Integral of x * f(x) over [a, b] by Simpson's rule."""
    return simpson(lambda x: x * pdf(x), a, b, panels)


def distribution_variance(
    pdf: Callable[[float], float], a: float, b: float, panels: int = 2000
) -> float:
    """Integral of (x - mu)^2 * f(x), with mu computed by the same rule."""
    mu = expected_value(pdf, a, b, panels)
    return simpson(lambda x: (x - mu) ** 2 * pdf(x), a, b, panels)


# Central limit theorem demo


def clt_demo(
    lam: float = 1.0, sample_size: int = 30, n_means: int = 4000, seed: int = 42
) -> dict[str, np.ndarray | float]:
    """Sample means of an exponential population look normal.

    Draws n_means averages, each over sample_size exponential draws, and
    returns the population, the means, and the normal curve the CLT
    predicts: mean 1 / lam, spread 1 / (lam * sqrt(sample_size)).
    """
    population = exponential_samples(n_means * sample_size, lam, seed)
    means = np.array(
        [
            sample_mean(population[i * sample_size : (i + 1) * sample_size])
            for i in range(n_means)
        ]
    )
    mu = 1.0 / lam
    sigma = 1.0 / (lam * math.sqrt(sample_size))
    grid = np.linspace(mu - 4.0 * sigma, mu + 4.0 * sigma, 200)
    curve = np.array([normal_pdf(float(x), mu, sigma) for x in grid])
    return {
        "population": population,
        "means": means,
        "grid": grid,
        "normal_curve": curve,
        "mu": mu,
        "sigma": sigma,
    }
