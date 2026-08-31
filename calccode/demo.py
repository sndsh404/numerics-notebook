"""python -m calccode.demo <topic>: headline numbers from one module.

Each topic prints a few lines of real output and finishes in well under
a second. Run with no arguments to list the topics.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np


def _limits() -> None:
    from calccode import limits

    r1 = limits.limit(lambda x: math.sin(x) / x, 0.0)
    r2 = limits.limit(lambda x: 1.0 / x, 0.0)
    r3 = limits.limit(lambda x: math.sin(1.0 / x), 0.0)
    print("numerical limits, sampled from both sides:")
    print(f"  sin(x)/x at x = 0: {r1.value:.6f} ({r1.status})")
    print(f"  1/x at x = 0:      {r2.status}")
    print(f"  sin(1/x) at x = 0: {r3.status}")


def _derivatives() -> None:
    from calccode import derivatives

    x, h = 1.0, 1e-3
    exact = math.cos(1.0)
    fwd = derivatives.forward_diff(math.sin, x, h)
    ctr = derivatives.central_diff(math.sin, x, h)
    print("d/dx sin(x) at x = 1, exact cos(1) = 0.540302, h = 1e-3:")
    print(f"  forward difference error: {abs(fwd - exact):.3e}")
    print(f"  central difference error: {abs(ctr - exact):.3e}")
    print(f"  central is O(h^2), here {abs(fwd - exact) / abs(ctr - exact):.0f}x more accurate")


def _integrals() -> None:
    from calccode import integrals

    n = 100
    trap = integrals.trapezoid(math.sin, 0.0, math.pi, n)
    simp = integrals.simpson(math.sin, 0.0, math.pi, n)
    print("integral of sin over [0, pi], exact value 2, n = 100 panels:")
    print(f"  trapezoid: {trap:.8f}  (error {abs(trap - 2.0):.2e})")
    print(f"  simpson:   {simp:.8f}  (error {abs(simp - 2.0):.2e})")


def _series() -> None:
    from calccode import series
    from calccode.symbolic import Sin, Var

    coeffs = series.taylor_coefficients_from_expr(Sin(Var("x")), 0.0, 9)
    poly = series.taylor_polynomial(coeffs, 0.0)
    approx = poly(1.0)
    s = series.partial_sum(lambda k: 1.0 / (k + 1) ** 2, 999)
    ratio, verdict = series.ratio_test(lambda k: 0.5**k)
    print("taylor series of sin at 0, degree 9, evaluated at x = 1:")
    print(f"  P9(1) = {approx:.8f}  (sin(1) = {math.sin(1.0):.8f})")
    print(f"sum of 1/k^2 to k = 1000: {s:.6f}  (pi^2/6 = {math.pi**2 / 6:.6f})")
    print(f"ratio test on (1/2)^k: ratio -> {ratio:.4f}, series {verdict}")


def _ode() -> None:
    from calccode import integrators

    f = lambda t, y: -y  # noqa: E731
    y0 = np.array([1.0])
    t_e, y_e = integrators.euler(f, 0.0, y0, 0.01, 100)
    t_r, y_r = integrators.rk4(f, 0.0, y0, 0.01, 100)
    exact = math.exp(-1.0)
    print("y' = -y, y(0) = 1, integrated to t = 1 with h = 0.01:")
    print(f"  exact y(1) = e^-1 = {exact:.8f}")
    print(f"  euler error: {abs(y_e[-1, 0] - exact):.2e}")
    print(f"  rk4 error:   {abs(y_r[-1, 0] - exact):.2e}")


def _fft() -> None:
    from calccode import fourier, montecarlo

    n = 1024
    rng = montecarlo.Xorshift32(42)
    signal = rng.uniforms(n)
    start = time.perf_counter()
    fourier.dft(signal)
    t_dft = time.perf_counter() - start
    start = time.perf_counter()
    fourier.fft(signal)
    t_fft = time.perf_counter() - start
    print(f"transform of a random signal, n = {n}:")
    print(f"  direct dft, O(n^2):   {t_dft:.3f} s")
    print(f"  radix-2 fft, O(n log n): {t_fft:.4f} s")
    print(f"  fft is {t_dft / t_fft:.0f}x faster")


def _linalg() -> None:
    from calccode import linalg

    A = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 1.0], [2.0, 1.0, 5.0]])
    b = np.array([10.0, 12.0, 21.0])
    x = linalg.solve(A, b)
    resid = linalg.norm(linalg.matmul(A, x.reshape(-1, 1)).ravel() - b)
    print("gaussian elimination on a fixed 3x3 system:")
    print(f"  x = [{x[0]:.1f}, {x[1]:.1f}, {x[2]:.1f}]")
    print(f"  det(A) = {linalg.determinant(A):.1f}")
    print(f"  residual ||Ax - b|| = {resid:.2e}")


def _orthogonal() -> None:
    from calccode import linalg, orthogonal

    A = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 4.0], [1.0, 3.0, 9.0], [1.0, 4.0, 16.0]])
    Q, R = orthogonal.qr_householder(A)
    gram = linalg.matmul(linalg.transpose(Q), Q)
    off = max(abs(gram[i, j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3))
    xs = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    y = np.array([1.1, 2.9, 5.2, 6.8, 9.1])
    c = orthogonal.lstsq_qr(xs, y)
    print("householder qr of a 4x3 vandermonde block:")
    print(f"  max |Q^T Q - I| = {off:.2e}")
    print("least squares fit of y = a + b x to 5 noisy points on y = 1 + 2x:")
    print(f"  a = {c[0]:.4f}, b = {c[1]:.4f}")


def _eigen() -> None:
    from calccode import eigen

    A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
    lam, v = eigen.power_iteration(A, seed=7)
    lam2, _ = eigen.inverse_iteration(A, shift=1.5)
    print("power iteration on a fixed symmetric 3x3:")
    print(f"  dominant eigenvalue: {lam:.6f}")
    print(f"  residual ||Av - lam v||: {eigen.residual(A, lam, v):.2e}")
    print(f"  eigenvalue nearest 1.5 (inverse iteration): {lam2:.6f}")


def _optimize() -> None:
    from calccode import optimize

    r = optimize.newton(lambda x: math.cos(x) - x, 1.0)
    rosen = lambda z: (1.0 - z[0]) ** 2 + 100.0 * (z[1] - z[0] ** 2) ** 2  # noqa: E731
    m = optimize.bfgs(rosen, np.array([-1.2, 1.0]))
    print("newton's method on cos(x) - x from x0 = 1:")
    print(f"  root {r.root:.6f} in {r.iterations} iterations")
    print("bfgs on the rosenbrock valley from (-1.2, 1):")
    print(f"  minimum at ({m.x[0]:.4f}, {m.x[1]:.4f}), f = {m.fun:.2e}, {m.iterations} iterations")


def _ml() -> None:
    from calccode import ml

    X, y = ml.make_blobs([(0.0, 0.0), (3.0, 3.0)], spread=1.0, n_per=60)
    X_train, X_test, y_train, y_test = ml.train_test_split(X, y)
    clf = ml.KNNClassifier(k=5).fit(X_train, y_train)
    acc = ml.accuracy(y_test, clf.predict(X_test))
    print("two gaussian blobs, 60 points each, 5-nn classifier:")
    print(f"  train {X_train.shape[0]} points, test {X_test.shape[0]} points")
    print(f"  test accuracy: {acc:.3f}")


def _kinematics() -> None:
    from calccode import transforms

    theta1, theta2 = 0.5, 0.8
    joints = transforms.planar_arm_fk(theta1, theta2)
    print(f"planar 2-link arm, l1 = l2 = 1, theta = ({theta1}, {theta2}):")
    print(f"  base:        ({joints[0, 0]:.4f}, {joints[0, 1]:.4f})")
    print(f"  elbow:       ({joints[1, 0]:.4f}, {joints[1, 1]:.4f})")
    print(f"  end effector: ({joints[2, 0]:.4f}, {joints[2, 1]:.4f})")


def _probability() -> None:
    from calccode import probability

    p = probability.normal_cdf(1.96)
    demo = probability.clt_demo(n_means=2000)
    means = demo["means"]
    print("normal cdf by simpson quadrature:")
    print(f"  Phi(1.96) = {p:.6f}  (table value 0.975002)")
    print("clt demo: means of 30 exponential draws, 2000 repetitions:")
    print(f"  mean of means {probability.sample_mean(means):.4f} (predicted {demo['mu']:.4f})")
    print(f"  std of means  {probability.sample_std(means):.4f} (predicted {demo['sigma']:.4f})")


def _montecarlo() -> None:
    from calccode import montecarlo

    pi_hat = montecarlo.estimate_pi(100_000)
    est = montecarlo.mc_integrate_1d(math.sin, 0.0, math.pi, 100_000)
    print("monte carlo with the hand-written xorshift32 rng:")
    print(f"  pi from 100000 darts: {pi_hat:.5f}  (error {abs(pi_hat - math.pi):.5f})")
    print(f"  integral of sin over [0, pi] from 100000 samples: {est:.5f}  (exact 2)")


TOPICS = {
    "limits": _limits,
    "derivatives": _derivatives,
    "integrals": _integrals,
    "series": _series,
    "ode": _ode,
    "fft": _fft,
    "linalg": _linalg,
    "orthogonal": _orthogonal,
    "eigen": _eigen,
    "optimize": _optimize,
    "ml": _ml,
    "kinematics": _kinematics,
    "probability": _probability,
    "montecarlo": _montecarlo,
}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python -m calccode.demo <topic>")
        print("topics: " + ", ".join(TOPICS))
        return 0
    topic = args[0]
    if topic not in TOPICS:
        print(f"unknown topic {topic!r}")
        print("topics: " + ", ".join(TOPICS))
        return 1
    TOPICS[topic]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
