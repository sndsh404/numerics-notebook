# 10 Transforms

`calccode/transforms.py` is the Modern Robotics module: rotations in 2D and 3D, Rodrigues' formula for axis-angle, homogeneous transforms, quaternions, and forward kinematics for a 2-link planar arm.

Rodrigues' formula is the star. Given a unit axis k and an angle, R = I cos(t) + sin(t) [k]x + (1 - cos(t)) k k^T. Three terms, each with a geometric meaning: the part that stays put, the part that rotates in the plane perpendicular to the axis, and the projection along the axis. Writing the skew-symmetric matrix by hand and watching the formula reproduce rotx, roty, rotz to machine precision was the moment axis-angle stopped being notation.

Quaternions earned their reputation. Converting to a rotation matrix is a clean formula. Converting back requires branching on the largest diagonal entry so the division stays away from zero, and then q and -q describe the same rotation so the test has to accept either sign. None of this is hard. All of it is fiddly, and every robotics codebase I have read carries the same fiddly code.

The transform composition test checks associativity indirectly: my compose() must match the raw matrix product, and invert_transform must undo it to the identity. The arm is the payoff: two angles, two link lengths, and the end effector lands where the geometry says it should. At (pi/2, pi/2) with lengths 2 and 1 the tip is at (-1, 2), and seeing that fall out of two cos and two sin calls is the whole point.

Where this breaks: quaternions near 180 degree rotations stress the conversion branches, and Euler angles have gimbal lock, which is why I skipped them. Composition also assumes you keep track of frames. The math will happily compose T_AB with T_CD and give you a meaningless number. The notation in Lynch and Park exists because the bookkeeping is the actual difficulty.
