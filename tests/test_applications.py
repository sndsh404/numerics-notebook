import math

from calccode.applications import arc_length, surface_area, volume_disk, volume_shell

R = 2.0


def semicircle(x: float) -> float:
    return math.sqrt(R * R - x * x)


def test_arc_length_of_line_segment():
    # f(x) = 2x + 1 on [0, 3]: the curve is a straight segment with
    # run 3 and rise 6, so its length is sqrt(45) = 3 sqrt(5).
    f = lambda t: 2.0 * t + 1.0
    assert abs(arc_length(f, 0.0, 3.0, 100) - 3.0 * math.sqrt(5.0)) < 1e-9


def test_semicircle_arc_length_shrunk_interval():
    # The derivative of sqrt(r^2 - x^2) blows up at x = r, and
    # central_diff steps outside the domain there, so the integral
    # runs on [-a, a] with a = r - eps. The closed form for the arc
    # over [-a, a] is 2 r arcsin(a / r).
    eps = 1e-3
    a = R - eps
    got = arc_length(semicircle, -a, a, 20000)
    assert abs(got - 2.0 * R * math.asin(a / R)) < 1e-5
    # The two missing end caps cost about 2 sqrt(2 r eps) ~ 0.126,
    # so the full half circumference pi r is only this close.
    assert abs(got - math.pi * R) < 0.15


def test_semicircle_volume_sphere():
    # f^2 = r^2 - x^2 is a polynomial, so no endpoint trouble here.
    got = volume_disk(semicircle, -R, R, 2000)
    assert abs(got - 4.0 / 3.0 * math.pi * R**3) < 1e-9


def test_semicircle_surface_area_sphere():
    # f sqrt(1 + (f')^2) cancels to the constant r analytically, and
    # the numerical derivative is close enough that the same endpoint
    # shrink gives a clean answer on [-a, a], where the closed form
    # is 4 pi r a.
    eps = 1e-3
    a = R - eps
    got = surface_area(semicircle, -a, a, 2000)
    assert abs(got - 4.0 * math.pi * R * a) < 1e-5
    assert abs(got - 4.0 * math.pi * R * R) < 0.03


def test_shell_method_matches_disk_method():
    # Region under f(x) = x on [0, r], rotated about the y-axis.
    # Shells: 2 pi int(x * x) = 2/3 pi r^3. The same solid is the
    # cylinder of radius and height r minus the cone of 1/3 pi r^3,
    # and the cylinder volume is what volume_disk computes for the
    # constant function r (same integral either way).
    shell = volume_shell(lambda t: t, 0.0, R, 2000)
    cylinder = volume_disk(lambda t: R, 0.0, R, 100)
    cone = math.pi * R**3 / 3.0
    assert abs(shell - 2.0 / 3.0 * math.pi * R**3) < 1e-9
    assert abs(shell - (cylinder - cone)) < 1e-9
