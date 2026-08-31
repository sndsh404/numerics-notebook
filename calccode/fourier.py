"""Discrete Fourier transform written as the plain O(n^2) sum.

No numpy.fft anywhere. dft and idft are direct evaluations of the
defining sums; dominant_frequencies reads peaks off the one-sided
amplitude spectrum.
"""

from __future__ import annotations

import cmath
import math

import numpy as np


def dft(signal: np.ndarray) -> np.ndarray:
    """X_k = sum_n x_n exp(-2 pi i k n / N), evaluated directly."""
    x = np.asarray(signal, dtype=complex)
    n = x.size
    out = np.empty(n, dtype=complex)
    for k in range(n):
        total = 0j
        for j in range(n):
            total += x[j] * cmath.exp(-2j * math.pi * k * j / n)
        out[k] = total
    return out


def idft(coeffs: np.ndarray) -> np.ndarray:
    """x_n = (1/N) sum_k X_k exp(+2 pi i k n / N)."""
    X = np.asarray(coeffs, dtype=complex)
    n = X.size
    out = np.empty(n, dtype=complex)
    for j in range(n):
        total = 0j
        for k in range(n):
            total += X[k] * cmath.exp(2j * math.pi * k * j / n)
        out[j] = total / n
    return out


def amplitude_spectrum(signal: np.ndarray) -> np.ndarray:
    """One-sided amplitude spectrum of a real signal: 2 |X_k| / N."""
    x = np.asarray(signal, dtype=float)
    n = x.size
    coeffs = dft(x)
    amps = 2.0 * np.abs(coeffs[: n // 2]) / n
    amps[0] /= 2.0  # the DC term is not doubled
    return amps


def dominant_frequencies(
    signal: np.ndarray, sample_rate: float, k: int = 3
) -> list[tuple[float, float]]:
    """The k strongest (frequency, amplitude) pairs in the signal.

    Frequencies come out at bin resolution sample_rate / N; refine them
    with a longer signal, not with interpolation tricks.
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    amps = amplitude_spectrum(x)
    order = np.argsort(amps)[::-1]
    picked: list[tuple[float, float]] = []
    for idx in order:
        if idx == 0:
            continue  # skip DC
        if len(picked) >= k:
            break
        freq = float(idx) * sample_rate / n
        picked.append((freq, float(amps[idx])))
    return picked
