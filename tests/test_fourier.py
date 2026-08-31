import math

import numpy as np
import pytest

from calccode import fourier
from calccode.montecarlo import Xorshift32


def make_signal(freqs_amps, n=512, sample_rate=512.0):
    t = np.arange(n) / sample_rate
    signal = np.zeros(n)
    for freq, amp in freqs_amps:
        signal += amp * np.sin(2.0 * math.pi * freq * t)
    return signal


def test_dft_of_pure_tone_has_single_peak():
    n = 64
    signal = make_signal([(4.0, 2.0)], n=n, sample_rate=64.0)
    coeffs = fourier.dft(signal)
    # A 2 sin(2 pi 4 t) tone puts all energy in bin 4 with magnitude N.
    assert abs(abs(coeffs[4]) - n) < 1e-8
    for k in (1, 2, 3, 5, 6):
        assert abs(coeffs[k]) < 1e-8


def test_dft_idft_round_trip():
    rng = np.random.default_rng(5)
    signal = rng.normal(size=64)
    assert np.allclose(fourier.idft(fourier.dft(signal)), signal, atol=1e-10)


def test_amplitude_spectrum_recovers_amplitudes():
    signal = make_signal([(3.0, 1.5), (7.0, 0.7)])
    amps = fourier.amplitude_spectrum(signal)
    assert abs(amps[3] - 1.5) < 1e-10
    assert abs(amps[7] - 0.7) < 1e-10


def test_dominant_frequencies_on_clean_signal():
    signal = make_signal([(3.0, 1.5), (7.0, 0.7), (11.0, 0.2)])
    top = fourier.dominant_frequencies(signal, sample_rate=512.0, k=3)
    freqs = sorted(f for f, _ in top)
    amps = sorted((a for _, a in top), reverse=True)
    assert freqs == [3.0, 7.0, 11.0]
    assert np.allclose(amps, [1.5, 0.7, 0.2], atol=1e-10)


def test_dominant_frequencies_survive_noise():
    rng = Xorshift32(9)
    signal = make_signal([(5.0, 1.0), (13.0, 0.6)])
    noise = np.array([(rng.uniform() - 0.5) * 0.1 for _ in range(signal.size)])
    top = fourier.dominant_frequencies(signal + noise, sample_rate=512.0, k=2)
    freqs = sorted(f for f, _ in top)
    assert freqs == [5.0, 13.0]


def test_dft_matches_naive_matrix_product():
    # Cross-check the loop against the Vandermonde-style matrix form.
    n = 16
    rng = np.random.default_rng(2)
    signal = rng.normal(size=n)
    k = np.arange(n)
    W = np.exp(-2j * math.pi * np.outer(k, k) / n)
    assert np.allclose(fourier.dft(signal), W @ signal, atol=1e-10)


def test_fft_matches_dft_on_fixed_inputs():
    rng = np.random.default_rng(11)
    for n in (1, 2, 4, 8, 64, 256, 1024):
        signal = rng.normal(size=n)
        assert np.allclose(fourier.fft(signal), fourier.dft(signal), atol=1e-10)


def test_fft_handles_complex_input():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=128) + 1j * rng.normal(size=128)
    assert np.allclose(fourier.fft(signal), fourier.dft(signal), atol=1e-10)


def test_fft_rejects_non_power_of_2():
    with pytest.raises(ValueError):
        fourier.fft(np.ones(100))


def test_pad_to_power_of_2():
    signal = np.arange(100, dtype=float)
    padded = fourier.pad_to_power_of_2(signal)
    assert padded.size == 128
    assert np.array_equal(padded[:100], signal)
    assert np.all(padded[100:] == 0.0)
    # Padded FFT equals the plain DFT of the padded signal.
    assert np.allclose(fourier.fft(padded), fourier.dft(padded), atol=1e-10)


def test_fft_ifft_round_trip():
    rng = np.random.default_rng(6)
    signal = rng.normal(size=512)
    assert np.allclose(fourier.ifft(fourier.fft(signal)), signal, atol=1e-10)


def test_fft_pure_tone_peak():
    n = 64
    signal = make_signal([(4.0, 2.0)], n=n, sample_rate=64.0)
    coeffs = fourier.fft(signal)
    assert abs(abs(coeffs[4]) - n) < 1e-8
    for k in (1, 2, 3, 5, 6):
        assert abs(coeffs[k]) < 1e-8


def test_fft_vs_dft_sizes_returns_matching_coeffs():
    # Correctness only: timing values are for the plot, not asserted.
    sizes, dft_times, fft_times = fourier.fft_vs_dft_sizes([8, 16])
    assert list(sizes) == [8, 16]
    assert dft_times.shape == fft_times.shape == (2,)
    assert np.all(dft_times > 0.0) and np.all(fft_times > 0.0)
