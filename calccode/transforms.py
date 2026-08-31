"""2D and 3D rigid body transforms, hand-written on numpy arrays.

Matrices are stored as numpy arrays but every formula (Rodrigues,
quaternion conversion, homogeneous inversion) is written out here.
Matrix products go through linalg.matmul, not numpy.
"""

from __future__ import annotations

import math

import numpy as np

from calccode import linalg


def rot2(theta: float) -> np.ndarray:
    """2D rotation by theta radians, counterclockwise."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotx(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]x such that [v]x w = v cross w."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rot_from_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' formula: R = I cos(t) + sin(t) [k]x + (1 - cos(t)) k k^T."""
    k = np.asarray(axis, dtype=float)
    norm = math.sqrt(float(np.sum(k * k)))
    if norm < 1e-12:
        raise ValueError("axis must be nonzero")
    k = k / norm
    K = skew(k)
    kk = k[:, None] * k[None, :]
    c, s = math.cos(theta), math.sin(theta)
    return linalg.identity(3) * c + K * s + kk * (1.0 - c)


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Homogeneous transform from a 3x3 rotation and a 3-vector translation."""
    T = linalg.identity(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float).ravel()
    return T


def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Apply T2 first, then T1: the product T1 T2."""
    return linalg.matmul(T1, T2)


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform: R^T and -R^T p. No general inverse needed."""
    R = T[:3, :3]
    p = T[:3, 3]
    R_inv = linalg.transpose(R)
    T_inv = linalg.identity(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -linalg.matmul(R_inv, p.reshape(3, 1)).ravel()
    return T_inv


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform one point (3,) or a set of points (N, 3)."""
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = pts.reshape(1, 3) if single else pts
    out = np.empty_like(pts)
    for i in range(pts.shape[0]):
        out[i] = linalg.matmul(T[:3, :3], pts[i].reshape(3, 1)).ravel() + T[:3, 3]
    return out[0] if single else out


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=float)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("quaternion must be nonzero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (w, x, y, z).

    Branches on the largest diagonal entry so the division stays away
    from zero whichever direction the rotation points.
    """
    R = np.asarray(R, dtype=float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array(
        [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    )


def planar_arm_fk(
    theta1: float, theta2: float, l1: float = 1.0, l2: float = 1.0
) -> np.ndarray:
    """Forward kinematics of a 2-link planar arm.

    Returns the joint positions base, elbow, end effector as a (3, 2)
    array. theta2 is relative to the first link, as usual for a revolute
    elbow joint.
    """
    base = np.array([0.0, 0.0])
    elbow = base + l1 * np.array([math.cos(theta1), math.sin(theta1)])
    end = elbow + l2 * np.array(
        [math.cos(theta1 + theta2), math.sin(theta1 + theta2)]
    )
    return np.vstack([base, elbow, end])
