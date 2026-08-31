"""Root finding and minimization, hand-written.

Root finding: bisection, Newton, and secant methods. Newton uses the
hand-written central difference from derivatives.py for f', so nothing
here needs an analytic derivative. Every method returns a RootResult
with the full x history, which makes convergence order measurable
instead of assumed.

Minimization: golden section in 1D, a backtracking Armijo line search,
Newton on the gradient system, BFGS with a hand-written inverse Hessian
update, and the gradient descent variants (momentum, Nesterov, Adam)
that note 05 promised. Gradients and Hessians come from multivar.py;
the Newton solve uses linalg.solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from calccode.derivatives import central_diff
from calccode.linalg import identity, matmul, norm, solve
from calccode.multivar import gradient, hessian


@dataclass
class RootResult:
    root: float
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> RootResult:
    """Bracketed root finding. Halves the interval every step.

    f must change sign on [a, b]; that sign change is the entire
    guarantee. The error after n steps is (b - a) / 2^n.
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return RootResult(a, 0, True, [a])
    if fb == 0.0:
        return RootResult(b, 0, True, [b])
    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    history = []
    for i in range(1, max_iter + 1):
        mid = 0.5 * (a + b)
        fm = f(mid)
        history.append(mid)
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return RootResult(mid, i, True, history)
        if fa * fm < 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return RootResult(0.5 * (a + b), max_iter, False, history)


def newton(
    f: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 50,
    h: float = 1e-6,
) -> RootResult:
    """Newton's method with a central difference derivative.

    Quadratic convergence near a simple root, but no guarantee anywhere
    else: a flat or cycling start point can send the iterates off to
    infinity. converged is False in that case.
    """
    x = x0
    history = [x0]
    for i in range(1, max_iter + 1):
        fx = f(x)
        if abs(fx) < tol:
            return RootResult(x, i - 1, True, history)
        dfx = central_diff(f, x, h)
        if dfx == 0.0 or not math.isfinite(dfx):
            return RootResult(x, i - 1, False, history)
        x = x - fx / dfx
        history.append(x)
        if not math.isfinite(x) or abs(x) > 1e12:
            return RootResult(x, i, False, history)
    return RootResult(x, max_iter, False, history)


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> RootResult:
    """Secant method: Newton with the slope from the last two iterates.

    Superlinear with order about 1.618, no derivative needed at all.
    """
    history = [x0, x1]
    f0, f1 = f(x0), f(x1)
    for i in range(1, max_iter + 1):
        if abs(f1) < tol:
            return RootResult(x1, i, True, history)
        denom = f1 - f0
        if denom == 0.0:
            return RootResult(x1, i, False, history)
        x2 = x1 - f1 * (x1 - x0) / denom
        history.append(x2)
        if not math.isfinite(x2) or abs(x2) > 1e12:
            return RootResult(x2, i, False, history)
        x0, x1, f0, f1 = x1, x2, f1, f(x2)
    return RootResult(x1, max_iter, False, history)


def compare_convergence(
    f: Callable[[float], float],
    bracket: tuple[float, float],
    x0: float,
    tol: float = 1e-10,
) -> dict[str, RootResult]:
    """Run all three methods on the same root and compare iteration counts."""
    return {
        "bisection": bisection(f, bracket[0], bracket[1], tol),
        "newton": newton(f, x0, tol),
        "secant": secant(f, bracket[0], bracket[1], tol),
    }


ScalarFn = Callable[[np.ndarray], float]


@dataclass
class GoldenResult:
    minimizer: float
    fmin: float
    iterations: int
    history: list[float] = field(default_factory=list)


@dataclass
class MinResult:
    x: np.ndarray
    fun: float
    iterations: int
    converged: bool
    history: np.ndarray


def golden_section(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> GoldenResult:
    """1D minimization on a bracket, shrinking by the golden ratio.

    Two interior probes at the golden section points decide which third
    of the bracket to drop, and one probe gets reused each step, so the
    cost is one new function evaluation per iteration. The bracket width
    falls by a factor of about 0.618 per step.
    """
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    history = [c, d]
    for i in range(1, max_iter + 1):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
            history.append(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
            history.append(d)
        if b - a < tol:
            mid = 0.5 * (a + b)
            return GoldenResult(mid, f(mid), i, history)
    mid = 0.5 * (a + b)
    return GoldenResult(mid, f(mid), max_iter, history)


def backtracking_line_search(
    f: ScalarFn,
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    direction: np.ndarray,
    alpha0: float = 1.0,
    c: float = 1e-4,
    beta: float = 0.5,
    max_iter: int = 50,
) -> tuple[float, np.ndarray]:
    """Armijo backtracking: shrink alpha until the decrease is sufficient.

    Accepts the first alpha in alpha0, alpha0*beta, alpha0*beta^2, ...
    with f(x + alpha*d) <= f(x) + c*alpha*(grad . d). The right side is
    a fraction of the decrease the linear model predicts, so an accepted
    step is guaranteed to go downhill by a measurable amount. Returns
    (alpha, x_new).
    """
    x = np.asarray(x, dtype=float)
    direction = np.asarray(direction, dtype=float)
    fx = f(x)
    slope = float(np.asarray(grad_f(x)) @ direction)
    if slope >= 0.0:
        raise ValueError("direction must be a descent direction")
    alpha = alpha0
    x_new = x + alpha * direction
    for _ in range(max_iter):
        x_new = x + alpha * direction
        if f(x_new) <= fx + c * alpha * slope:
            return alpha, x_new
        alpha *= beta
    return alpha, x_new


def newton_minimize(
    f: ScalarFn,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> MinResult:
    """Newton's method for minimization: solve Hessian * d = -gradient.

    One step lands on the minimizer of a quadratic bowl. Away from a
    quadratic the raw step can point uphill or overshoot, so every step
    goes through the Armijo line search, and a non-descent or singular
    Hessian falls back to steepest descent for that step.
    """
    x = np.asarray(x0, dtype=float).copy()
    grad_f = lambda z: gradient(f, z)  # noqa: E731
    history = [x.copy()]
    for i in range(max_iter):
        g = grad_f(x)
        if norm(g) < tol:
            return MinResult(x, f(x), i, True, np.array(history))
        try:
            d = solve(hessian(f, x), -g)
        except ValueError:
            d = -g
        if float(g @ d) >= 0.0:
            d = -g
        _, x = backtracking_line_search(f, grad_f, x, d)
        history.append(x.copy())
    return MinResult(x, f(x), max_iter, norm(grad_f(x)) < tol, np.array(history))


def _matvec(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    return matmul(A, v.reshape(-1, 1)).ravel()


def _outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return matmul(a.reshape(-1, 1), b.reshape(1, -1))


def bfgs(
    f: ScalarFn,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> MinResult:
    """BFGS with a hand-written inverse Hessian update.

    H starts as the identity and picks up curvature from the iterate
    pairs: H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T, with
    rho = 1 / (y . s). The update keeps H symmetric positive definite
    as long as y . s > 0, which the Armijo line search makes likely but
    not certain; a bad pair is skipped instead of poisoning H.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    grad_f = lambda z: gradient(f, z)  # noqa: E731
    H = identity(n)
    g = grad_f(x)
    history = [x.copy()]
    for i in range(max_iter):
        if norm(g) < tol:
            return MinResult(x, f(x), i, True, np.array(history))
        d = -_matvec(H, g)
        if float(g @ d) >= 0.0:
            d = -g
            H = identity(n)
        _, x_new = backtracking_line_search(f, grad_f, x, d)
        g_new = grad_f(x_new)
        s = x_new - x
        y = g_new - g
        ys = float(y @ s)
        if ys > 1e-12:
            rho = 1.0 / ys
            eye = identity(n)
            left = eye - rho * _outer(s, y)
            right = eye - rho * _outer(y, s)
            H = matmul(matmul(left, H), right) + rho * _outer(s, s)
        x, g = x_new, g_new
        history.append(x.copy())
    return MinResult(x, f(x), max_iter, norm(grad_f(x)) < tol, np.array(history))


def momentum_gd(
    f: ScalarFn,
    x0: np.ndarray,
    lr: float = 0.01,
    beta: float = 0.9,
    tol: float = 1e-8,
    max_iter: int = 10000,
    h: float = 1e-6,
) -> MinResult:
    """Gradient descent with a velocity term: v = beta*v - lr*g, x += v.

    The velocity averages recent gradients, which damps the zigzag along
    the stiff direction of a narrow valley and builds speed along the
    shallow one. It is the cheap fix for the conditioning problem from
    note 05.
    """
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    for i in range(max_iter):
        g = gradient(f, x, h)
        if norm(g) < tol:
            return MinResult(x, f(x), i, True, np.array(history))
        v = beta * v - lr * g
        x = x + v
        history.append(x.copy())
    return MinResult(x, f(x), max_iter, norm(gradient(f, x, h)) < tol, np.array(history))


def nesterov_gd(
    f: ScalarFn,
    x0: np.ndarray,
    lr: float = 0.01,
    beta: float = 0.9,
    tol: float = 1e-8,
    max_iter: int = 10000,
    h: float = 1e-6,
) -> MinResult:
    """Nesterov momentum: evaluate the gradient at the lookahead point.

    Same interface as momentum_gd. The gradient is taken at
    x + beta*v, where the velocity is about to send the iterate, which
    lets the method brake before an overshoot instead of after it.
    """
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    for i in range(max_iter):
        g = gradient(f, x, h)
        if norm(g) < tol:
            return MinResult(x, f(x), i, True, np.array(history))
        g_look = gradient(f, x + beta * v, h)
        v = beta * v - lr * g_look
        x = x + v
        history.append(x.copy())
    return MinResult(x, f(x), max_iter, norm(gradient(f, x, h)) < tol, np.array(history))


def adam(
    f: ScalarFn,
    x0: np.ndarray,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tol: float = 1e-8,
    max_iter: int = 10000,
    h: float = 1e-6,
) -> MinResult:
    """Adam: per-coordinate step sizes from first and second moments.

    m tracks the mean gradient, v the mean squared gradient, both
    bias-corrected for their zero start. Dividing by sqrt(v) gives each
    coordinate its own effective learning rate, so flat directions move
    faster and steep ones slower without any hand tuning per axis.
    """
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [x.copy()]
    for i in range(1, max_iter + 1):
        g = gradient(f, x, h)
        if norm(g) < tol:
            return MinResult(x, f(x), i - 1, True, np.array(history))
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1**i)
        v_hat = v / (1.0 - beta2**i)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(x.copy())
    return MinResult(x, f(x), max_iter, norm(gradient(f, x, h)) < tol, np.array(history))
