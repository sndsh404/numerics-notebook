"""Screw theory kinematics, following Lynch and Park, Modern Robotics.

Twists are 6-vectors ordered (omega, v): angular part first, linear part
second. Matrices are numpy arrays, but every matrix product goes through
linalg.matmul and every linear solve through linalg.solve. No numpy.linalg.

Conventions: fk_space uses space-frame screw axes S and builds
T = exp([S1] t1) ... exp([Sn] tn) M. fk_body uses body-frame screw axes B
and builds T = M exp([B1] t1) ... exp([Bn] tn). The matrix exponential
functions take the se(3) or so(3) matrix with the angle already folded in,
so matrix_exp_so3 receives [w] theta and matrix_log_so3 returns [w] theta.
"""

from __future__ import annotations

import math

import numpy as np

from calccode import linalg, transforms

_NEAR_ZERO = 1e-9


def _mv(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Matrix times vector through linalg.matmul."""
    return linalg.matmul(A, np.asarray(v, dtype=float).reshape(-1, 1)).ravel()


def vec_to_so3(omega: np.ndarray) -> np.ndarray:
    """3-vector to its skew-symmetric so(3) matrix."""
    return transforms.skew(np.asarray(omega, dtype=float))


def so3_to_vec(so3mat: np.ndarray) -> np.ndarray:
    """Skew-symmetric so(3) matrix back to its 3-vector."""
    return np.array([so3mat[2, 1], so3mat[0, 2], so3mat[1, 0]])


def vec_to_se3(V: np.ndarray) -> np.ndarray:
    """Twist (omega, v) to its 4x4 se(3) matrix."""
    V = np.asarray(V, dtype=float)
    se3mat = np.zeros((4, 4))
    se3mat[:3, :3] = vec_to_so3(V[:3])
    se3mat[:3, 3] = V[3:]
    return se3mat


def se3_to_vec(se3mat: np.ndarray) -> np.ndarray:
    """4x4 se(3) matrix back to the twist (omega, v)."""
    return np.concatenate([so3_to_vec(se3mat[:3, :3]), se3mat[:3, 3]])


def matrix_exp_so3(so3mat: np.ndarray) -> np.ndarray:
    """Exponential of [w] theta in so(3): Rodrigues in matrix form.

    R = I + sin(theta) [w_hat] + (1 - cos theta) [w_hat]^2, where
    theta = |w theta| and [w_hat] = [w theta] / theta.
    """
    omega_theta = so3_to_vec(so3mat)
    theta = math.sqrt(float(np.sum(omega_theta * omega_theta)))
    if theta < _NEAR_ZERO:
        return linalg.identity(3)
    omgmat = so3mat / theta
    omgmat2 = linalg.matmul(omgmat, omgmat)
    c, s = math.cos(theta), math.sin(theta)
    return linalg.identity(3) + omgmat * s + omgmat2 * (1.0 - c)


def matrix_log_so3(R: np.ndarray) -> np.ndarray:
    """Log of a rotation matrix: the so(3) matrix [w] theta with |theta| <= pi."""
    R = np.asarray(R, dtype=float)
    acosinput = (linalg.trace(R) - 1.0) / 2.0
    acosinput = min(1.0, max(-1.0, acosinput))
    if acosinput >= 1.0:
        return np.zeros((3, 3))
    if acosinput <= -1.0:
        # theta = pi: pull omega from the diagonal of R, fix signs off-diagonal.
        if abs(1.0 + R[2, 2]) > _NEAR_ZERO:
            omega = (1.0 / math.sqrt(2.0 * (1.0 + R[2, 2]))) * np.array(
                [R[0, 2], R[1, 2], 1.0 + R[2, 2]]
            )
        elif abs(1.0 + R[1, 1]) > _NEAR_ZERO:
            omega = (1.0 / math.sqrt(2.0 * (1.0 + R[1, 1]))) * np.array(
                [R[0, 1], 1.0 + R[1, 1], R[2, 1]]
            )
        else:
            omega = (1.0 / math.sqrt(2.0 * (1.0 + R[0, 0]))) * np.array(
                [1.0 + R[0, 0], R[1, 0], R[2, 0]]
            )
        return vec_to_so3(omega * math.pi)
    theta = math.acos(acosinput)
    return (theta / (2.0 * math.sin(theta))) * (R - linalg.transpose(R))


def matrix_exp_se3(se3mat: np.ndarray) -> np.ndarray:
    """Exponential of [V] theta in se(3).

    Rotation part is matrix_exp_so3. Translation is p = G v_theta with
    G = I theta + (1 - cos theta) [w_hat] + (theta - sin theta) [w_hat]^2.
    """
    se3mat = np.asarray(se3mat, dtype=float)
    omega_theta = so3_to_vec(se3mat[:3, :3])
    theta = math.sqrt(float(np.sum(omega_theta * omega_theta)))
    T = linalg.identity(4)
    if theta < _NEAR_ZERO:
        T[:3, 3] = se3mat[:3, 3]
        return T
    T[:3, :3] = matrix_exp_so3(se3mat[:3, :3])
    omgmat = se3mat[:3, :3] / theta
    omgmat2 = linalg.matmul(omgmat, omgmat)
    c, s = math.cos(theta), math.sin(theta)
    G = linalg.identity(3) * theta + omgmat * (1.0 - c) + omgmat2 * (theta - s)
    T[:3, 3] = _mv(G, se3mat[:3, 3] / theta)
    return T


def matrix_log_se3(T: np.ndarray) -> np.ndarray:
    """Log of a homogeneous transform: the se(3) matrix [V] theta."""
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    so3mat = matrix_log_so3(R)
    se3mat = np.zeros((4, 4))
    se3mat[:3, :3] = so3mat
    if float(np.max(np.abs(so3mat))) < _NEAR_ZERO:
        se3mat[:3, 3] = p
        return se3mat
    omega_theta = so3_to_vec(so3mat)
    theta = math.sqrt(float(np.sum(omega_theta * omega_theta)))
    omgmat = so3mat / theta
    omgmat2 = linalg.matmul(omgmat, omgmat)
    G_inv = (
        linalg.identity(3) / theta
        - omgmat / 2.0
        + omgmat2 * (1.0 / theta - 1.0 / (2.0 * math.tan(theta / 2.0)))
    )
    # G_inv p recovers v; the se(3) matrix stores v theta.
    se3mat[:3, 3] = _mv(G_inv, p) * theta
    return se3mat


def adjoint(T: np.ndarray) -> np.ndarray:
    """6x6 adjoint matrix [Ad_T] mapping twists: V' = [Ad_T] V.

    With (omega, v) ordering this is [[R, 0], [[p] R, R]].
    """
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = linalg.matmul(transforms.skew(p), R)
    Ad[3:, 3:] = R
    return Ad


def fk_space(screw_axes: list, home: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Space-form product of exponentials: T = exp([S1] t1) ... exp([Sn] tn) M."""
    T = linalg.identity(4)
    for S, theta in zip(screw_axes, thetas):
        T = linalg.matmul(T, matrix_exp_se3(vec_to_se3(np.asarray(S, dtype=float)) * theta))
    return linalg.matmul(T, home)


def fk_body(screw_axes: list, home: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Body-form product of exponentials: T = M exp([B1] t1) ... exp([Bn] tn)."""
    T = np.asarray(home, dtype=float)
    for B, theta in zip(screw_axes, thetas):
        T = linalg.matmul(T, matrix_exp_se3(vec_to_se3(np.asarray(B, dtype=float)) * theta))
    return T


def jacobian_space(screw_axes: list, thetas: np.ndarray) -> np.ndarray:
    """Space Jacobian: column i is [Ad_{T_{i-1}}] S_i with T_0 = I."""
    n = len(screw_axes)
    Js = np.zeros((6, n))
    T = linalg.identity(4)
    for i in range(n):
        Js[:, i] = _mv(adjoint(T), np.asarray(screw_axes[i], dtype=float))
        T = linalg.matmul(
            T, matrix_exp_se3(vec_to_se3(np.asarray(screw_axes[i], dtype=float)) * thetas[i])
        )
    return Js


def jacobian_body(screw_axes: list, thetas: np.ndarray) -> np.ndarray:
    """Body Jacobian: column i is [Ad_{T}] B_i with T = exp(-[Bn] tn) ... exp(-[B_{i+1}] t_{i+1})."""
    n = len(screw_axes)
    Jb = np.zeros((6, n))
    T = linalg.identity(4)
    for i in range(n - 1, -1, -1):
        Jb[:, i] = _mv(adjoint(T), np.asarray(screw_axes[i], dtype=float))
        T = linalg.matmul(
            T, matrix_exp_se3(vec_to_se3(np.asarray(screw_axes[i], dtype=float)) * (-thetas[i]))
        )
    return Jb


def _damped_lstsq_step(J: np.ndarray, V: np.ndarray, damping: float) -> np.ndarray:
    """Solve (J^T J + damping I) dtheta = J^T V by hand-built normal equations."""
    J = np.asarray(J, dtype=float)
    n = J.shape[1]
    Jt = linalg.transpose(J)
    A = linalg.matmul(Jt, J)
    for i in range(n):
        A[i, i] += damping
    b = _mv(Jt, V)
    return linalg.solve(A, b)


def _ik(fk, jacobian, screw_axes, home, thetas0, target, eomg, ev, max_iter, damping, space):
    thetas = np.asarray(thetas0, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    for iteration in range(1, max_iter + 1):
        T = fk(screw_axes, home, thetas)
        # Body-frame error twist of T relative to the target.
        Vb = se3_to_vec(matrix_log_se3(linalg.matmul(transforms.invert_transform(T), target)))
        err_ang = math.sqrt(float(np.sum(Vb[:3] * Vb[:3])))
        err_lin = math.sqrt(float(np.sum(Vb[3:] * Vb[3:])))
        if err_ang < eomg and err_lin < ev:
            return {"thetas": thetas, "success": True, "iterations": iteration - 1}
        if space:
            # Same error expressed in the space frame, for the space Jacobian.
            V = _mv(adjoint(T), Vb)
        else:
            V = Vb
        J = jacobian(screw_axes, thetas)
        thetas = thetas + _damped_lstsq_step(J, V, damping)
    return {"thetas": thetas, "success": False, "iterations": max_iter}


def ik_space(
    screw_axes: list,
    home: np.ndarray,
    thetas0: np.ndarray,
    target: np.ndarray,
    eomg: float = 1e-6,
    ev: float = 1e-6,
    max_iter: int = 100,
    damping: float = 1e-4,
) -> dict:
    """Numerical IK in the space form, damped least squares Newton steps.

    Returns a dict with the joint vector, a success flag, and the number of
    Newton iterations taken.
    """
    return _ik(
        fk_space, jacobian_space, screw_axes, home, thetas0, target, eomg, ev, max_iter, damping, True
    )


def ik_body(
    screw_axes: list,
    home: np.ndarray,
    thetas0: np.ndarray,
    target: np.ndarray,
    eomg: float = 1e-6,
    ev: float = 1e-6,
    max_iter: int = 100,
    damping: float = 1e-4,
) -> dict:
    """Numerical IK in the body form. Same return convention as ik_space."""
    return _ik(
        fk_body, jacobian_body, screw_axes, home, thetas0, target, eomg, ev, max_iter, damping, False
    )


def _scaling(method: str, u: float) -> tuple[float, float, float]:
    """Time scaling and its first two derivatives with respect to u = t / Tf."""
    if method == "cubic":
        return 3 * u * u - 2 * u**3, 6 * u - 6 * u * u, 6 - 12 * u
    if method == "quintic":
        return (
            10 * u**3 - 15 * u**4 + 6 * u**5,
            30 * u * u - 60 * u**3 + 30 * u**4,
            60 * u - 180 * u * u + 120 * u**3,
        )
    raise ValueError(f"unknown method {method!r}, expected 'cubic' or 'quintic'")


def cubic_time_scaling(Tf: float, t: float) -> float:
    """Third-order time scaling s(t) = 3 (t/Tf)^2 - 2 (t/Tf)^3."""
    return _scaling("cubic", t / Tf)[0]


def quintic_time_scaling(Tf: float, t: float) -> float:
    """Fifth-order time scaling s(t) = 10 u^3 - 15 u^4 + 6 u^5, u = t/Tf."""
    return _scaling("quintic", t / Tf)[0]


def joint_trajectory(
    thetastart: np.ndarray,
    thetafinal: np.ndarray,
    Tf: float,
    N: int,
    method: str = "cubic",
) -> dict:
    """Point-to-point joint trajectory under cubic or quintic time scaling.

    Returns a dict with times (N,), and positions, velocities, and
    accelerations, each (N, n).
    """
    thetastart = np.asarray(thetastart, dtype=float)
    thetafinal = np.asarray(thetafinal, dtype=float)
    gap = thetafinal - thetastart
    times = np.linspace(0.0, Tf, N)
    positions = np.zeros((N, thetastart.size))
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    for i, t in enumerate(times):
        s, ds, dds = _scaling(method, t / Tf)
        positions[i] = thetastart + s * gap
        velocities[i] = (ds / Tf) * gap
        accelerations[i] = (dds / (Tf * Tf)) * gap
    return {
        "times": times,
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
    }
