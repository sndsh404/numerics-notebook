import math

import numpy as np
import pytest

from calccode import montecarlo
from calccode.derivatives import fit_order
from calccode.montecarlo import Xorshift32


def test_rng_is_deterministic():
    a = Xorshift32(7).uniforms(5)
    b = Xorshift32(7).uniforms(5)
    assert np.array_equal(a, b)


def test_rng_different_seeds_differ():
    a = Xorshift32(1).uniforms(5)
    b = Xorshift32(2).uniforms(5)
    assert not np.array_equal(a, b)


def test_rng_rejects_bad_seed():
    with pytest.raises(ValueError):
        Xorshift32(0)


def test_rng_looks_uniform():
    samples = Xorshift32(3).uniforms(20000)
    assert 0.49 < float(np.mean(samples)) < 0.51
    assert float(np.min(samples)) >= 0.0
    assert float(np.max(samples)) < 1.0


def test_mc_integrate_1d_polynomial():
    # Integral of x^3 over [0, 1] is 0.25.
    got = montecarlo.mc_integrate_1d(lambda x: x**3, 0.0, 1.0, 20000, seed=1)
    assert abs(got - 0.25) < 0.01


def test_mc_integrate_nd_unit_square():
    # Integral of x^2 + y^2 over the unit square is 2/3.
    got = montecarlo.mc_integrate_nd(
        lambda p: p[0] ** 2 + p[1] ** 2, [(0.0, 1.0), (0.0, 1.0)], 20000, seed=2
    )
    assert abs(got - 2.0 / 3.0) < 0.02


def test_estimate_pi_at_fixed_seed():
    got = montecarlo.estimate_pi(100000, seed=42)
    assert abs(got - math.pi) < 0.02


def test_error_scales_like_inverse_sqrt_n():
    ns = np.array([100, 1000, 10000, 100000])
    _, stds = montecarlo.pi_error_scaling(ns, n_seeds=25)
    slope = fit_order(ns, stds)  # log(std) vs log(n) slope
    assert -0.75 < slope < -0.3
    # Ten times more samples, about sqrt(10) times less spread.
    assert stds[-1] < stds[0] / 10.0


def test_importance_sampling_is_exact_for_matched_density():
    got = montecarlo.integrate_inv_sqrt_importance(100, seed=5)
    assert abs(got - 2.0) < 1e-12


def test_importance_sampling_beats_uniform():
    uniform_err = abs(montecarlo.integrate_inv_sqrt_uniform(5000, seed=5) - 2.0)
    importance_err = abs(montecarlo.integrate_inv_sqrt_importance(5000, seed=5) - 2.0)
    assert importance_err < uniform_err / 100.0
