"""Discrete and fast Fourier transforms, written by hand.

No numpy.fft anywhere. dft and idft are direct evaluations of the
defining sums; dominant_frequencies reads peaks off the one-sided
amplitude spectrum. fft and ifft are the same sums with the redundancy
factored out: a radix-2 Cooley-Tukey butterfly over bit-reversed input.
"""

from __future__ import annotations

import cmath
import math
import time

import numpy as np

from calccode.montecarlo import Xorshift32


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


def next_power_of_2(n: int) -> int:
    """Smallest power of 2 that is >= n."""
    if n < 1:
        raise ValueError("n must be positive")
    return 1 << (n - 1).bit_length()


def pad_to_power_of_2(signal: np.ndarray) -> np.ndarray:
    """Zero-pad a signal up to the next power-of-2 length."""
    x = np.asarray(signal)
    target = next_power_of_2(x.size)
    if target == x.size:
        return x.copy()
    return np.concatenate([x, np.zeros(target - x.size, dtype=x.dtype)])


def fft(signal: np.ndarray) -> np.ndarray:
    """Radix-2 Cooley-Tukey FFT. Same sum as dft, O(n log n).

    Iterative, so depth never becomes a problem: the input is permuted
    into bit-reversed order once, then butterflies of size 2, 4, ..., n
    combine it in place. Length must be a power of 2; pad first with
    pad_to_power_of_2 if it is not.
    """
    x = np.asarray(signal, dtype=complex)
    n = x.size
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("fft needs a power-of-2 length; pad with pad_to_power_of_2")

    out = np.empty(n, dtype=complex)
    bits = n.bit_length() - 1
    for i in range(n):
        rev, v = 0, i
        for _ in range(bits):
            rev = (rev << 1) | (v & 1)
            v >>= 1
        out[rev] = x[i]

    size = 2
    while size <= n:
        half = size // 2
        w_step = cmath.exp(-2j * math.pi / size)
        for start in range(0, n, size):
            w = 1 + 0j
            for k in range(half):
                a = out[start + k]
                b = out[start + k + half] * w
                out[start + k] = a + b
                out[start + k + half] = a - b
                w *= w_step
        size *= 2
    return out


def ifft(coeffs: np.ndarray) -> np.ndarray:
    """Inverse FFT by conjugation: conj(fft(conj(X))) / N."""
    X = np.asarray(coeffs, dtype=complex)
    return np.conj(fft(np.conj(X))) / X.size


def fft_vs_dft_sizes(
    sizes: list[int] | np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wall times of dft and fft at each size, for the O(n^2) vs O(n log n) plot.

    Returns (sizes, dft_times, fft_times). Signals are drawn from
    Xorshift32 so the comparison is reproducible. Non-power-of-2 sizes
    are padded up before timing, so fft always gets a legal length.
    """
    sizes = np.asarray(sizes, dtype=int)
    dft_times = np.empty(sizes.size)
    fft_times = np.empty(sizes.size)
    for i, n in enumerate(sizes):
        rng = Xorshift32(seed + i)
        signal = pad_to_power_of_2(rng.uniforms(int(n)))

        start = time.perf_counter()
        dft(signal)
        dft_times[i] = time.perf_counter() - start

        start = time.perf_counter()
        fft(signal)
        fft_times[i] = time.perf_counter() - start
    return sizes, dft_times, fft_times
