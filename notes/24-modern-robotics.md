# 24 Modern Robotics

`calccode/kinematics.py` picks up where `transforms.py` stopped: screw axes, matrix exponentials and logarithms on so(3) and se(3), the product of exponentials formula for forward kinematics, space and body Jacobians, damped least squares inverse kinematics, and cubic and quintic time scaling for trajectories.

The exponential is the whole theory in one function. matrix_exp_so3 is Rodrigues' formula again, this time reading the angle out of the matrix itself: theta is the norm of the vector packed inside [w] theta. matrix_exp_se3 adds the translation term G v with G = I theta + (1 - cos theta)[w] + (theta - sin theta)[w]^2, and matrix_log_se3 has to invert G in closed form. That inverse cost me a bug. It returns v, but the se(3) matrix stores v theta, and my first version dropped the theta. Every round trip test failed by exactly the factor 1/theta, which made the bug easy to find and slightly embarrassing to have written.

Once exp and log round trip, forward kinematics is a loop. fk_space multiplies the exp([S_i] theta_i) terms left to right and appends M. fk_body starts at M and multiplies rightward. The two forms agree because B_i = [Ad_{M^{-1}}] S_i, and the test that convinced me the Jacobians are right is the identity J_b = [Ad_{Tbs}] J_s. One 6x6 matrix ties the two formalisms together.

Inverse kinematics is Newton's method on the error twist. Compute T, take the log of T^{-1} T_target, and step theta by the damped pseudoinverse of the Jacobian, with J^T J + lambda I assembled by hand and solved by Gaussian elimination. On the 3R test arm it recovers a known configuration from 0.1 rad away in a handful of iterations.

Where this breaks: the space and body convention trap. The error twist lives in the end-effector frame, so ik_space must push it through [Ad_{Tsb}] before touching the space Jacobian. Pick the wrong adjoint direction, or order the twist (v, omega) instead of (omega, v), and the iteration still moves. It just walks confidently to nowhere. The failure is silent, because every intermediate matrix is a perfectly legal 4x4.
