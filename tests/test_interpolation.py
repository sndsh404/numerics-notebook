import math

import numpy as np

from calccode import interpolation as interp


def runge(x):
    return 1.0 / (1.0 + 25.0 * x * x)


def test_lagrange_hits_every_node_exactly():
    xs = np.linspace(-1.0, 1.0, 9)
    ys = np.array([runge(float(x)) for x in xs])
    for x, y in zip(xs, ys):
        assert abs(interp.lagrange_eval(xs, ys, float(x)) - y) < 1e-12


def test_newton_hits_every_node_exactly():
    xs = np.linspace(0.0, 2.0, 7)
    ys = np.sin(xs)
    for x, y in zip(xs, ys):
        assert abs(interp.newton_eval(xs, ys, float(x)) - y) < 1e-10


def test_piecewise_linear_hits_nodes_and_midpoints():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 2.0, 0.0])
    for x, y in zip(xs, ys):
        assert interp.piecewise_linear(xs, ys, float(x)) == y
    assert abs(interp.piecewise_linear(xs, ys, 0.5) - 1.0) < 1e-12
    assert abs(interp.piecewise_linear(xs, ys, 1.5) - 1.0) < 1e-12


def test_barycentric_agrees_with_divided_differences():
    xs = np.linspace(-1.0, 1.0, 8)
    ys = np.array([runge(float(x)) for x in xs])
    for x in np.linspace(-0.95, 0.95, 41):
        a = interp.lagrange_eval(xs, ys, float(x))
        b = interp.newton_eval(xs, ys, float(x))
        assert abs(a - b) < 1e-8


def test_interpolant_recovers_quadratic_exactly():
    # A degree-4 interpolant of a quadratic is the quadratic itself.
    f = lambda x: 2.0 * x * x - 3.0 * x + 1.0
    xs = np.linspace(-1.0, 1.0, 5)
    ys = np.array([f(float(x)) for x in xs])
    for x in np.linspace(-1.0, 1.0, 21):
        assert abs(interp.lagrange_eval(xs, ys, float(x)) - f(float(x))) < 1e-10


def test_spline_hits_nodes_and_has_natural_ends():
    xs = np.linspace(0.0, 2.0 * math.pi, 9)
    ys = np.sin(xs)
    s = interp.natural_cubic_spline(xs, ys)
    for x, y in zip(xs, ys):
        assert abs(s(float(x)) - y) < 1e-10
    # Natural boundary: second derivative vanishes at both ends.
    h = 1e-4
    for x0 in (xs[0], xs[-1]):
        d2 = (s(float(x0) + h) - 2.0 * s(float(x0)) + s(float(x0) - h)) / (h * h)
        assert abs(d2) < 1e-2


def test_spline_second_derivative_is_continuous_at_knots():
    xs = np.array([0.0, 0.7, 1.4, 2.2, 3.0])
    ys = np.array([0.0, 1.0, 0.3, 1.5, 0.2])
    s = interp.natural_cubic_spline(xs, ys)
    h = 1e-5

    def d2(x):
        return (s(x + h) - 2.0 * s(x) + s(x - h)) / (h * h)

    for knot in xs[1:-1]:
        eps = 10.0 * h
        assert abs(d2(float(knot) - eps) - d2(float(knot) + eps)) < 1e-2


def test_chebyshev_beats_equispaced_on_runge():
    errors = interp.runge_comparison(n=15)
    assert errors["equispaced_err"] > 1.0  # oscillation near the ends
    assert errors["chebyshev_err"] < 0.1
    assert errors["chebyshev_err"] < errors["equispaced_err"]


def test_chebyshev_nodes_cluster_near_ends():
    nodes = interp.chebyshev_nodes(11)
    assert abs(nodes[0] - math.cos(math.pi / 22.0)) < 1e-12
    assert abs(nodes[-1] + nodes[0]) < 1e-12  # symmetric about 0
    # End gaps are tighter than the middle gap.
    assert nodes[0] - nodes[1] < nodes[5] - nodes[6]
