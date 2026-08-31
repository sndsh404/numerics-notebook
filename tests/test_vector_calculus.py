import math

import numpy as np

from calccode import vector_calculus as vc

# The disk-region tests (Green, Stokes) use n = 400 and tol = 5e-4 on
# the area side. The y bounds are sqrt(1 - x^2), whose derivative blows
# up at the endpoints, and that singularity drags Simpson's error down
# to roughly n^-1.5 instead of the usual n^-4. n = 400 keeps the test
# near a second; tightening the tol would cost a much larger n.

circle2d = lambda t: np.array([math.cos(t), math.sin(t)])
circle3d = lambda t: np.array([math.cos(t), math.sin(t), 0.0])
disk_region = (
    (-1.0, 1.0),
    (lambda x: -math.sqrt(max(0.0, 1.0 - x * x)), lambda x: math.sqrt(max(0.0, 1.0 - x * x))),
)


def test_double_integral_over_triangle():
    # x*y over the triangle x in [0, 1], y in [0, 1 - x] is 1/24.
    val = vc.double_integral(lambda x, y: x * y, (0.0, 1.0), (lambda x: 0.0, lambda x: 1.0 - x), 20)
    assert abs(val - 1.0 / 24.0) < 1e-12


def test_double_integral_rectangular_matches_iteration():
    # Simpson is exact on this integrand, so both regions agree to roundoff.
    f = lambda x, y: x * x + y
    rect = vc.double_integral(f, (0.0, 2.0), (1.0, 3.0), 10)
    exact = 2.0 * (8.0 / 3.0) + 2.0 * 4.0  # (int x^2 dx) * height + (int y dy) * width
    assert abs(rect - exact) < 1e-12


def test_line_integral_conservative_field_two_paths():
    # F = grad(x^2 y), so the work depends only on the endpoints.
    F = lambda p: np.array([2.0 * p[0] * p[1], p[0] ** 2])
    straight = lambda t: np.array([t, t])
    parabola = lambda t: np.array([t, t * t])
    a = vc.line_integral_vector(F, straight, (0.0, 1.0), 50)
    b = vc.line_integral_vector(F, parabola, (0.0, 1.0), 50)
    assert abs(a - 1.0) < 1e-9
    assert abs(b - 1.0) < 1e-9
    assert abs(a - b) < 1e-9


def test_line_integral_conservative_field_closed_loop_is_zero():
    F = lambda p: np.array([2.0 * p[0] * p[1], p[0] ** 2])
    work = vc.line_integral_vector(F, circle2d, (0.0, 2.0 * math.pi), 100)
    assert abs(work) < 1e-8


def test_surface_area_of_sphere():
    r = 2.0
    sphere = lambda u, v: np.array(
        [r * math.sin(u) * math.cos(v), r * math.sin(u) * math.sin(v), r * math.cos(u)]
    )
    area = vc.surface_integral_scalar(lambda p: 1.0, sphere, (0.0, math.pi), (0.0, 2.0 * math.pi), 60)
    assert abs(area - 4.0 * math.pi * r * r) < 1e-5


def test_green_theorem_unit_disk():
    # F = (-y, x) has 2D curl 2, so both sides are 2 * area = 2 pi.
    F = lambda p: np.array([-p[1], p[0]])
    line, area = vc.check_green(F, circle2d, (0.0, 2.0 * math.pi), disk_region, 400)
    assert abs(line - 2.0 * math.pi) < 1e-8
    assert abs(area - 2.0 * math.pi) < 5e-4
    assert abs(line - area) < 5e-4


def test_divergence_theorem_unit_cube():
    # F = (x, y, z) has divergence 3 over a unit volume, and each unit
    # face at coordinate 1 contributes flux 1.
    F = lambda p: np.array([p[0], p[1], p[2]])
    box = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    flux, volume = vc.check_divergence(F, box, 10)
    assert abs(flux - 3.0) < 1e-9
    assert abs(volume - 3.0) < 1e-9


def test_stokes_theorem_disk_in_plane():
    # F = (-y, x, 0) has curl (0, 0, 2); over the unit disk in z = 0
    # both sides are 2 pi.
    F = lambda p: np.array([-p[1], p[0], 0.0])
    g = lambda x, y: 0.0
    circulation, flux = vc.check_stokes(F, (g, disk_region, circle3d, (0.0, 2.0 * math.pi)), 400)
    assert abs(circulation - 2.0 * math.pi) < 1e-8
    assert abs(flux - 2.0 * math.pi) < 5e-4
    assert abs(circulation - flux) < 5e-4


def test_lagrange_maximizes_xy_on_unit_circle():
    f = lambda p: p[0] * p[1]
    g = lambda p: p[0] ** 2 + p[1] ** 2 - 1.0
    point = vc.lagrange_critical(f, g, np.array([1.0, 0.5]))
    assert np.allclose(point, [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], atol=1e-9)
    assert abs(f(point) - 0.5) < 1e-9


def test_lagrange_finds_second_branch_from_other_start():
    f = lambda p: p[0] * p[1]
    g = lambda p: p[0] ** 2 + p[1] ** 2 - 1.0
    point = vc.lagrange_critical(f, g, np.array([-1.0, -0.3]))
    assert np.allclose(point, [-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)], atol=1e-9)
