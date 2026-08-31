"""Vector calculus: iterated integrals, line and surface integrals, and the three big theorems.

Everything is iterated Simpson from integrals.py, with tangent vectors,
partial derivatives of parametrizations, curls, and divergences from
central differences and the multivar.py jacobian. The theorem checkers
(Green, divergence, Stokes) compute both sides of the identity
independently and return both numbers, so the tests can compare them
against each other and against the exact value.

Lagrange multipliers reduce to a root finding problem: solve
grad f = lambda * grad g together with g = 0 using Newton's method on
the stacked system, with linalg.solve for each step.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from calccode.derivatives import central_diff
from calccode.integrals import simpson
from calccode.linalg import norm, solve
from calccode.multivar import gradient, jacobian

ScalarFn2 = Callable[[float, float], float]
ScalarFn3 = Callable[[float, float, float], float]
PointFn = Callable[[np.ndarray], float]
FieldFn = Callable[[np.ndarray], np.ndarray]
CurveFn = Callable[[float], np.ndarray]
SurfaceFn = Callable[[float, float], np.ndarray]

DIFF_H = 1e-5


def _y_bounds(y_range_or_fn) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """Accept constant bounds (c, d) or a pair of functions of x."""
    if callable(y_range_or_fn[0]):
        return y_range_or_fn
    c, d = y_range_or_fn
    return (lambda x: c), (lambda x: d)


def double_integral(f: ScalarFn2, x_range: tuple[float, float], y_range_or_fn, n: int) -> float:
    """Double integral of f over a region, iterated Simpson both ways.

    The y bounds may be constants or functions of x, so triangles,
    disks, and other non-rectangular regions work directly. Simpson is
    exact for low degree polynomials, so rectangles with polynomial
    integrands come back at machine precision; curved boundaries cost
    real panels because the boundary shape enters through the outer
    integrand.
    """
    ylo, yhi = _y_bounds(y_range_or_fn)
    a, b = x_range

    def inner(x: float) -> float:
        return simpson(lambda y: f(x, y), ylo(x), yhi(x), n)

    return simpson(inner, a, b, n)


def triple_integral(
    f: ScalarFn3,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    n: int,
) -> float:
    """Triple integral over a box, one more level of iterated Simpson."""
    return simpson(
        lambda x: double_integral(lambda y, z: f(x, y, z), y_range, z_range, n),
        x_range[0],
        x_range[1],
        n,
    )


def line_integral_scalar(f: PointFn, curve: CurveFn, t_range: tuple[float, float], n: int) -> float:
    """Integral of f along a curve with respect to arc length.

    The tangent r'(t) is a central difference, so the curve only needs
    to be a callable from t to a point.
    """

    def integrand(t: float) -> float:
        point = np.asarray(curve(t), dtype=float)
        tangent = central_diff(lambda s: np.asarray(curve(s), dtype=float), t, DIFF_H)
        return f(point) * norm(tangent)

    return simpson(integrand, t_range[0], t_range[1], n)


def line_integral_vector(F: FieldFn, curve: CurveFn, t_range: tuple[float, float], n: int) -> float:
    """Work integral of a vector field along a curve, F . dr."""

    def integrand(t: float) -> float:
        point = np.asarray(curve(t), dtype=float)
        tangent = central_diff(lambda s: np.asarray(curve(s), dtype=float), t, DIFF_H)
        return float(np.asarray(F(point), dtype=float) @ tangent)

    return simpson(integrand, t_range[0], t_range[1], n)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors, written out by hand."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def surface_integral_scalar(
    f: PointFn,
    surface: SurfaceFn,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    n: int,
) -> float:
    """Integral of f over a parametrized surface, f times |r_u x r_v|.

    The partials of the parametrization are central differences in u
    and v. With f = 1 this is surface area.
    """

    def integrand(u: float, v: float) -> float:
        r_u = central_diff(lambda s: np.asarray(surface(s, v), dtype=float), u, DIFF_H)
        r_v = central_diff(lambda s: np.asarray(surface(u, s), dtype=float), v, DIFF_H)
        return f(np.asarray(surface(u, v), dtype=float)) * norm(cross(r_u, r_v))

    return double_integral(integrand, u_range, v_range, n)


def curl(F: FieldFn, p: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Curl of a 3D field at p, from the hand-written jacobian."""
    J = jacobian(F, np.asarray(p, dtype=float), h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def divergence(F: FieldFn, p: np.ndarray, h: float = 1e-5) -> float:
    """Divergence of a field at p: the trace of its jacobian."""
    J = jacobian(F, np.asarray(p, dtype=float), h)
    return float(sum(J[i, i] for i in range(J.shape[0])))


def check_green(
    F: FieldFn,
    curve: CurveFn,
    t_range: tuple[float, float],
    region,
    n: int,
) -> tuple[float, float]:
    """Green's theorem, both sides.

    Left side: line integral of F around the closed curve. Right side:
    double integral of the 2D curl (dF2/dx - dF1/dy) over the region,
    given as (x_range, y_range_or_fn). Returns (line, area).
    """
    x_range, y_range_or_fn = region
    line = line_integral_vector(F, curve, t_range, n)

    def curl2d(x: float, y: float) -> float:
        J = jacobian(F, np.array([x, y]))
        return J[1, 0] - J[0, 1]

    area = double_integral(curl2d, x_range, y_range_or_fn, n)
    return line, area


def _face_flux(F: FieldFn, box_ranges, axis: int, upper: bool, n: int) -> float:
    """Outward flux of F through one face of an axis-aligned box."""
    others = [a for a in range(3) if a != axis]
    sign = 1.0 if upper else -1.0
    fixed = box_ranges[axis][1 if upper else 0]

    def integrand(u: float, v: float) -> float:
        p = np.zeros(3)
        p[axis] = fixed
        p[others[0]] = u
        p[others[1]] = v
        return sign * float(np.asarray(F(p), dtype=float)[axis])

    return double_integral(integrand, box_ranges[others[0]], box_ranges[others[1]], n)


def check_divergence(F: FieldFn, box_ranges, n: int) -> tuple[float, float]:
    """Divergence theorem over an axis-aligned box, both sides.

    Left side: outward flux through the six faces, each face a double
    integral of F . n where n is the axis unit vector. Right side:
    triple integral of the divergence over the box. Returns
    (flux, volume).
    """
    flux = sum(
        _face_flux(F, box_ranges, axis, upper, n)
        for axis in range(3)
        for upper in (False, True)
    )
    volume = triple_integral(
        lambda x, y, z: divergence(F, np.array([x, y, z])),
        box_ranges[0],
        box_ranges[1],
        box_ranges[2],
        n,
    )
    return flux, volume


def check_stokes(F: FieldFn, surface_and_boundary, n: int) -> tuple[float, float]:
    """Stokes' theorem for a graph z = g(x, y), both sides.

    surface_and_boundary is (g, region, curve, t_range): g maps (x, y)
    to z, region is (x_range, y_range_or_fn) for the base domain, and
    curve parametrizes the boundary. The unnormalized normal of the
    graph is (-dg/dx, -dg/dy, 1), which keeps dS exact. Returns
    (circulation, flux_of_curl).
    """
    g, region, curve, t_range = surface_and_boundary
    circulation = line_integral_vector(F, curve, t_range, n)

    def integrand(x: float, y: float) -> float:
        gx = central_diff(lambda s: g(s, y), x, DIFF_H)
        gy = central_diff(lambda s: g(x, s), y, DIFF_H)
        c = curl(F, np.array([x, y, g(x, y)]))
        return -c[0] * gx - c[1] * gy + c[2]

    flux = double_integral(integrand, region[0], region[1], n)
    return circulation, flux


def lagrange_critical(
    f: PointFn,
    g: PointFn,
    x0: np.ndarray,
    lam0: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> np.ndarray:
    """Critical point of f subject to g = 0, by Newton's method.

    The unknowns are x and lambda together. The system is
    grad f(x) - lambda * grad g(x) = 0 stacked on g(x) = 0, the
    jacobian comes from multivar.jacobian, and each step is solved with
    linalg.solve. Newton needs a decent start: from far away it can
    jump to the wrong root or diverge, and it returns one point per
    start, so symmetric problems with several extrema need one call
    each. Raises if it does not converge.
    """
    x0 = np.asarray(x0, dtype=float)
    d = x0.size
    z = np.concatenate([x0, [lam0]])

    def system(zv: np.ndarray) -> np.ndarray:
        point, lam = zv[:d], zv[d]
        return np.concatenate(
            [gradient(f, point) - lam * gradient(g, point), [g(point)]]
        )

    for _ in range(max_iter):
        Fz = system(z)
        if norm(Fz) < tol:
            return z[:d]
        z = z - solve(jacobian(system, z), Fz)
    raise ValueError("lagrange_critical did not converge; try a closer x0")
