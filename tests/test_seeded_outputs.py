"""Seeded generators must be exactly reproducible, run to run.

The xorshift32 stream in montecarlo.py feeds the samplers in
probability.py and the blob generator in ml.py. Same seed in, same bits
out: no tolerance, no approx.
"""

import numpy as np

from calccode import ml, montecarlo, probability


def test_estimate_pi_is_reproducible():
    assert montecarlo.estimate_pi(5000, seed=7) == montecarlo.estimate_pi(5000, seed=7)


def test_normal_samples_are_bit_identical():
    a = probability.normal_samples(500, seed=3, mu=1.5, sigma=0.5)
    b = probability.normal_samples(500, seed=3, mu=1.5, sigma=0.5)
    assert np.array_equal(a, b)


def test_uniform_samples_are_bit_identical():
    a = probability.uniform_samples(500, seed=9, a=-2.0, b=3.0)
    b = probability.uniform_samples(500, seed=9, a=-2.0, b=3.0)
    assert np.array_equal(a, b)


def test_make_blobs_is_reproducible():
    centers = [(0.0, 0.0), (3.0, 3.0), (-2.0, 1.0)]
    X1, y1 = ml.make_blobs(centers, spread=0.4, n_per=40, seed=11)
    X2, y2 = ml.make_blobs(centers, spread=0.4, n_per=40, seed=11)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_different_seeds_give_different_draws():
    a = probability.uniform_samples(100, seed=1)
    b = probability.uniform_samples(100, seed=2)
    assert not np.array_equal(a, b)
