import math

import numpy as np
import pytest

from calccode.gradient import gradient_descent
from calccode.linalg import norm
from calccode.multivar import gradient
from calccode.optimize import (
    adam,
    backtracking_line_search,
    bfgs,
    bisection,
    compare_convergence,
    golden_section,
    momentum_gd,
    nesterov_gd,
    newton,
    newton_minimize,
    secant,
)


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


# ---------------------------------------------------------------- minimization

rosenbrock = lambda v: (1.0 - v[0]) ** 2 + 100.0 * (v[1] - v[0] ** 2) ** 2  # noqa: E731
ROSEN_START = np.array([-1.2, 1.0])


def _first_below(path: np.ndarray, tol: float) -> int:
    """First index where the Rosenbrock value drops under tol."""
    for i, p in enumerate(path):
        if rosenbrock(p) < tol:
            return i
    return len(path)


def test_golden_section_finds_quartic_minimizer():
    # (x - 1)^4 + 4 has its only minimum at x = 1 with value 4.
    res = golden_section(lambda x: (x - 1.0) ** 4 + 4.0, -2.0, 5.0)
    assert abs(res.minimizer - 1.0) < 1e-3
    assert abs(res.fmin - 4.0) < 1e-9


def test_golden_section_bracket_shrinks_by_golden_ratio():
    # Width ~ 7 * 0.618^n, so about 52 steps to reach 1e-10 from width 7.
    res = golden_section(lambda x: (x - 1.0) ** 4 + 4.0, -2.0, 5.0, tol=1e-10)
    assert res.iterations <= 55


def test_backtracking_satisfies_armijo():
    bowl = lambda v: (v[0] - 1.0) ** 2 + 3.0 * (v[1] + 2.0) ** 2  # noqa: E731
    x = np.array([0.0, 0.0])
    g = gradient(bowl, x)
    direction = -g
    c = 1e-4
    alpha, x_new = backtracking_line_search(bowl, lambda z: gradient(bowl, z), x, direction, c=c)
    slope = float(g @ direction)
    assert slope < 0.0
    assert bowl(x_new) <= bowl(x) + c * alpha * slope + 1e-15


def test_backtracking_rejects_ascent_direction():
    bowl = lambda v: (v[0] - 1.0) ** 2 + 3.0 * (v[1] + 2.0) ** 2  # noqa: E731
    x = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        backtracking_line_search(bowl, lambda z: gradient(bowl, z), x, np.array([1.0, 1.0]))


def test_newton_minimize_quadratic_bowl_in_few_steps():
    bowl = lambda v: (v[0] - 1.0) ** 2 + 3.0 * (v[1] + 2.0) ** 2  # noqa: E731
    res = newton_minimize(bowl, np.array([0.0, 0.0]))
    assert res.converged
    assert res.iterations <= 5
    assert norm(res.x - np.array([1.0, -2.0])) < 1e-6


def test_bfgs_minimizes_rosenbrock():
    res = bfgs(rosenbrock, ROSEN_START)
    assert res.converged
    assert res.iterations < 200
    assert norm(res.x - np.array([1.0, 1.0])) < 1e-5


def test_gd_variants_beat_plain_gd_on_rosenbrock():
    tol = 1e-4
    plain = gradient_descent(rosenbrock, ROSEN_START, lr=0.001, n_iter=20000)
    plain_iters = _first_below(plain, tol)
    assert plain_iters < 20000

    runs = {
        "momentum": momentum_gd(rosenbrock, ROSEN_START, lr=0.001, beta=0.9, tol=0.0, max_iter=3000),
        "nesterov": nesterov_gd(rosenbrock, ROSEN_START, lr=0.001, beta=0.9, tol=0.0, max_iter=3000),
        "adam": adam(rosenbrock, ROSEN_START, lr=0.1, tol=0.0, max_iter=3000),
    }
    for name, res in runs.items():
        iters = _first_below(res.history, tol)
        assert iters < res.history.shape[0], f"{name} never reached the tolerance"
        assert iters < plain_iters, f"{name} took {iters}, plain GD took {plain_iters}"
