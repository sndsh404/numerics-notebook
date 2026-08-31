import math

import numpy as np

from calccode import kinematics, transforms

# Planar 3R arm, link lengths 1, home config pointing up the y axis.
# Screw axes S_i = (omega, v) with omega = (0, 0, 1) and v = -omega x q_i.
L1 = L2 = L3 = 1.0
HOME = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, L1 + L2 + L3],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
SPACE_SCREWS = [
    np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, L1, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, L1 + L2, 0.0, 0.0]),
]
BODY_SCREWS = [
    np.array([0.0, 0.0, 1.0, -(L1 + L2 + L3), 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, -(L2 + L3), 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, -L3, 0.0, 0.0]),
]


def test_so3_vec_round_trip():
    w = np.array([0.3, -0.5, 0.8])
    assert np.allclose(kinematics.so3_to_vec(kinematics.vec_to_so3(w)), w)


def test_se3_vec_round_trip():
    V = np.array([0.2, -0.4, 0.7, 1.0, -0.5, 0.3])
    assert np.allclose(kinematics.se3_to_vec(kinematics.vec_to_se3(V)), V)


def test_exp_log_so3_round_trip():
    w_theta = np.array([0.3, -0.5, 0.8])
    so3mat = kinematics.vec_to_so3(w_theta)
    R = kinematics.matrix_exp_so3(so3mat)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.allclose(kinematics.matrix_log_so3(R), so3mat, atol=1e-10)


def test_exp_log_so3_identity():
    assert np.allclose(kinematics.matrix_exp_so3(np.zeros((3, 3))), np.eye(3))
    assert np.allclose(kinematics.matrix_log_so3(np.eye(3)), np.zeros((3, 3)))


def test_exp_log_se3_round_trip():
    V_theta = np.array([0.2, -0.4, 0.7, 1.0, -0.5, 0.3])
    se3mat = kinematics.vec_to_se3(V_theta)
    T = kinematics.matrix_exp_se3(se3mat)
    assert np.allclose(kinematics.matrix_log_se3(T), se3mat, atol=1e-10)


def test_exp_se3_pure_translation():
    V_theta = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    T = kinematics.matrix_exp_se3(kinematics.vec_to_se3(V_theta))
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])


def test_adjoint_matches_conjugation():
    T = kinematics.matrix_exp_se3(
        kinematics.vec_to_se3(np.array([0.1, 0.2, -0.3, 1.0, -2.0, 0.5]))
    )
    V = np.array([0.4, -0.1, 0.2, 0.7, 0.3, -0.9])
    expected = kinematics.se3_to_vec(
        T @ kinematics.vec_to_se3(V) @ transforms.invert_transform(T)
    )
    assert np.allclose(kinematics.adjoint(T) @ V, expected, atol=1e-10)


def test_fk_space_home_at_zero():
    T = kinematics.fk_space(SPACE_SCREWS, HOME, np.zeros(3))
    assert np.allclose(T, HOME, atol=1e-12)


def test_fk_space_quarter_turn_first_joint():
    # Rotating joint 1 by pi/2 swings the whole arm from +y onto -x.
    T = kinematics.fk_space(SPACE_SCREWS, HOME, np.array([math.pi / 2, 0.0, 0.0]))
    assert np.allclose(T[:3, :3], transforms.rotz(math.pi / 2), atol=1e-12)
    assert np.allclose(T[:3, 3], [-(L1 + L2 + L3), 0.0, 0.0], atol=1e-12)


def test_fk_space_two_quarter_turns():
    # theta = (pi/2, pi/2, 0): home points up +y, so the link directions are
    # 180, 270, 270 degrees and the tip lands at (-1, -2) by hand counting.
    T = kinematics.fk_space(SPACE_SCREWS, HOME, np.array([math.pi / 2, math.pi / 2, 0.0]))
    assert np.allclose(T[:3, :3], transforms.rotz(math.pi), atol=1e-12)
    assert np.allclose(T[:3, 3], [-1.0, -2.0, 0.0], atol=1e-12)


def test_fk_body_matches_fk_space():
    thetas = np.array([0.4, -0.7, 1.1])
    T_space = kinematics.fk_space(SPACE_SCREWS, HOME, thetas)
    T_body = kinematics.fk_body(BODY_SCREWS, HOME, thetas)
    assert np.allclose(T_space, T_body, atol=1e-10)


def test_jacobians_adjoint_relationship():
    # With (omega, v) twist ordering the two Jacobians obey
    # J_b = [Ad_{T_bs}] J_s, with no sign flip; T_bs = T_sb^{-1}.
    thetas = np.array([0.4, -0.7, 1.1])
    Js = kinematics.jacobian_space(SPACE_SCREWS, thetas)
    Jb = kinematics.jacobian_body(BODY_SCREWS, thetas)
    T_bs = transforms.invert_transform(kinematics.fk_space(SPACE_SCREWS, HOME, thetas))
    assert np.allclose(Jb, kinematics.adjoint(T_bs) @ Js, atol=1e-10)


def test_ik_space_recovers_target():
    thetas_true = np.array([0.5, -0.7, 0.9])
    target = kinematics.fk_space(SPACE_SCREWS, HOME, thetas_true)
    result = kinematics.ik_space(SPACE_SCREWS, HOME, thetas_true + 0.1, target)
    assert result["success"]
    assert result["iterations"] < 100
    T = kinematics.fk_space(SPACE_SCREWS, HOME, result["thetas"])
    assert np.allclose(T, target, atol=1e-6)


def test_ik_body_recovers_target():
    thetas_true = np.array([0.5, -0.7, 0.9])
    target = kinematics.fk_body(BODY_SCREWS, HOME, thetas_true)
    result = kinematics.ik_body(BODY_SCREWS, HOME, thetas_true + 0.1, target)
    assert result["success"]
    assert result["iterations"] < 100
    T = kinematics.fk_body(BODY_SCREWS, HOME, result["thetas"])
    assert np.allclose(T, target, atol=1e-6)


def test_ik_reports_failure_on_unreachable_target():
    target = np.array(HOME)
    target[:3, 3] = [10.0, 10.0, 0.0]
    result = kinematics.ik_space(SPACE_SCREWS, HOME, np.zeros(3), target, max_iter=20)
    assert not result["success"]
    assert result["iterations"] == 20


def test_cubic_time_scaling_endpoints():
    assert kinematics.cubic_time_scaling(2.0, 0.0) == 0.0
    assert kinematics.cubic_time_scaling(2.0, 2.0) == 1.0
    assert kinematics.quintic_time_scaling(2.0, 0.0) == 0.0
    assert kinematics.quintic_time_scaling(2.0, 2.0) == 1.0


def test_cubic_trajectory_zero_endpoint_velocity():
    traj = kinematics.joint_trajectory(
        np.array([0.0, 1.0]), np.array([1.5, -0.5]), Tf=2.0, N=101, method="cubic"
    )
    assert np.allclose(traj["velocities"][0], 0.0, atol=1e-12)
    assert np.allclose(traj["velocities"][-1], 0.0, atol=1e-12)
    # Cubic acceleration is nonzero at the endpoints.
    assert not np.allclose(traj["accelerations"][0], 0.0)


def test_quintic_trajectory_zero_endpoint_velocity_and_acceleration():
    traj = kinematics.joint_trajectory(
        np.array([0.0, 1.0]), np.array([1.5, -0.5]), Tf=2.0, N=101, method="quintic"
    )
    assert np.allclose(traj["velocities"][0], 0.0, atol=1e-12)
    assert np.allclose(traj["velocities"][-1], 0.0, atol=1e-12)
    assert np.allclose(traj["accelerations"][0], 0.0, atol=1e-12)
    assert np.allclose(traj["accelerations"][-1], 0.0, atol=1e-12)


def test_trajectory_endpoints_match():
    start = np.array([0.3, -1.2, 0.7])
    final = np.array([-0.4, 0.8, 2.0])
    for method in ("cubic", "quintic"):
        traj = kinematics.joint_trajectory(start, final, Tf=3.0, N=51, method=method)
        assert np.allclose(traj["positions"][0], start, atol=1e-12)
        assert np.allclose(traj["positions"][-1], final, atol=1e-12)
        assert traj["positions"].shape == (51, 3)
