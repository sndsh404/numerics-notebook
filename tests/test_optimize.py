import math

import numpy as np
import pytest

from calccode.optimize import bisection, compare_convergence, newton, secant


def test_bisection_finds_sqrt2():
    res = bisection(lambda x: x * x - 2.0, 1.0, 2.0)
    assert res.converged
    assert abs(res.root - math.sqrt(2.0)) < 1e-9


def test_bisection_bracket_halves_each_step():
    res = bisection(lambda x: x * x - 2.0, 1.0, 2.0, tol=1e-10)
    # Interval starts at width 1 and halves per iteration.
    assert res.iterations <= math.ceil(math.log2(1.0 / 1e-10)) + 1
    hist = np.array(res.history)
    # Each iterate stays inside the shrinking bracket.
    assert np.all(hist >= 1.0) and np.all(hist <= 2.0)


def test_bisection_requires_sign_change():
    with pytest.raises(ValueError):
        bisection(lambda x: x * x + 1.0, -1.0, 1.0)


def test_newton_finds_sqrt2_fast():
    res = newton(lambda x: x * x - 2.0, 1.0)
    assert res.converged
    assert abs(res.root - math.sqrt(2.0)) < 1e-9
    assert res.iterations < 10


def test_newton_converges_quadratically():
    # For a simple root, e_{n+1} ~ C e_n^2, so log(e_{n+1})/log(e_n) ~ 2.
    res = newton(lambda x: x * x - 2.0, 1.5)
    errs = [abs(v - math.sqrt(2.0)) for v in res.history]
    errs = [e for e in errs if 1e-14 < e < 1e-1]
    orders = [
        math.log(errs[i + 1]) / math.log(errs[i]) for i in range(len(errs) - 1)
    ]
    # Skip the first step, which is not yet in the asymptotic regime.
    assert len(orders) >= 2
    assert all(1.7 < o < 2.4 for o in orders[1:])


def test_secant_superlinear_between_1_and_2():
    res = secant(lambda x: x * x - 2.0, 1.0, 2.0)
    assert res.converged
    assert abs(res.root - math.sqrt(2.0)) < 1e-9
    errs = [abs(v - math.sqrt(2.0)) for v in res.history]
    errs = [e for e in errs if 1e-14 < e < 1e-1]
    orders = [
        math.log(errs[i + 1]) / math.log(errs[i]) for i in range(len(errs) - 1)
    ]
    assert all(1.2 < o < 2.2 for o in orders)


def test_compare_convergence_orders_methods():
    results = compare_convergence(lambda x: x * x - 2.0, (1.0, 2.0), 1.5)
    assert all(r.converged for r in results.values())
    assert results["newton"].iterations < results["bisection"].iterations
    assert results["secant"].iterations < results["bisection"].iterations


def test_newton_diverges_on_arctan_from_large_start():
    # Classic failure: for |x0| above about 1.39, Newton on arctan
    # amplifies each iterate instead of approaching the root at 0.
    res = newton(math.atan, 3.0)
    assert not res.converged


def test_newton_divergence_is_visible_in_history():
    res = newton(math.atan, 3.0)
    hist = np.abs(np.array(res.history))
    # Iterates grow past the start instead of settling.
    assert hist[-1] > 10.0 * hist[0]


def test_newton_on_arctan_converges_from_small_start():
    res = newton(math.atan, 0.5)
    assert res.converged
    assert abs(res.root) < 1e-9


def test_cube_root_via_newton():
    res = newton(lambda x: x**3 - 8.0, 2.5)
    assert res.converged
    assert abs(res.root - 2.0) < 1e-8
