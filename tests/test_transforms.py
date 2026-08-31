import math

import numpy as np

from calccode import linalg, transforms


def test_rot2_quarter_turn():
    R = transforms.rot2(math.pi / 2)
    v = np.array([1.0, 0.0])
    assert np.allclose(R @ v, [0.0, 1.0], atol=1e-12)


def test_rotations_are_orthogonal():
    R = transforms.rotx(0.3) @ transforms.roty(-0.7) @ transforms.rotz(1.1)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert abs(linalg.determinant(R) - 1.0) < 1e-12


def test_rot_from_axis_angle_matches_axis_rotations():
    for axis, R_axis in (([1, 0, 0], transforms.rotx), ([0, 1, 0], transforms.roty), ([0, 0, 1], transforms.rotz)):
        theta = 0.7
        assert np.allclose(transforms.rot_from_axis_angle(np.array(axis), theta), R_axis(theta), atol=1e-12)


def test_quat_round_trip():
    rng = np.random.default_rng(1)
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    R = transforms.quat_to_rot(q)
    q2 = transforms.rot_to_quat(R)
    # q and -q represent the same rotation
    assert np.allclose(q, q2, atol=1e-9) or np.allclose(q, -q2, atol=1e-9)


def test_compose_matches_matrix_product():
    T1 = transforms.make_transform(transforms.rotz(0.5), np.array([1.0, 2.0, 3.0]))
    T2 = transforms.make_transform(transforms.rotx(-0.2), np.array([4.0, 5.0, 6.0]))
    assert np.allclose(transforms.compose(T1, T2), T1 @ T2)


def test_invert_transform_recovers_identity():
    T = transforms.make_transform(transforms.roty(0.9), np.array([1.0, -2.0, 0.5]))
    assert np.allclose(transforms.compose(T, transforms.invert_transform(T)), np.eye(4), atol=1e-12)


def test_apply_moves_points():
    T = transforms.make_transform(transforms.rotz(math.pi / 2), np.array([1.0, 0.0, 0.0]))
    p = np.array([1.0, 0.0, 0.0])
    assert np.allclose(transforms.apply(T, p), [1.0, 1.0, 0.0], atol=1e-12)


def test_planar_arm_fk_end_position():
    fk = transforms.planar_arm_fk(0.0, 0.0, l1=2.0, l2=1.0)
    assert np.allclose(fk[-1], [3.0, 0.0])
    fk = transforms.planar_arm_fk(math.pi / 2, math.pi / 2, l1=2.0, l2=1.0)
    assert np.allclose(fk[-1], [-1.0, 2.0], atol=1e-12)
