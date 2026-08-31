"""Exercise: rigid body transforms.

Reference implementation: calccode/transforms.py.
"""

import numpy as np


def rot2(theta: float) -> np.ndarray:
    """2D rotation matrix, counterclockwise."""
    raise NotImplementedError


def rotz(theta: float) -> np.ndarray:
    """3D rotation about the z axis."""
    raise NotImplementedError


def rot_from_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' formula: R = I cos(t) + sin(t) [k]x + (1 - cos(t)) k k^T."""
    raise NotImplementedError


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform using R^T and -R^T p."""
    raise NotImplementedError


def planar_arm_fk(theta1: float, theta2: float, l1: float, l2: float) -> np.ndarray:
    """Joint positions of a 2-link planar arm, shape (3, 2)."""
    raise NotImplementedError
