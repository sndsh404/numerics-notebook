# 26 Dynamics

`calccode/dynamics.py` is where Modern Robotics stops being geometry and starts being physics. The kinematics module can place the end of a 2-link arm anywhere; this module answers what happens when you let go.

The model is the simplest one that is not trivial: point masses m1 at the elbow and m2 at the tip, massless links. Writing down the kinetic energy means differentiating the tip position (l1 cos q1 + l2 cos(q1+q2), l1 sin q1 + l2 sin(q1+q2)) and summing (1/2) m v^2 for both masses. The Euler-Lagrange equation then turns that energy into the equation M(q) qdd + C(q, qd) + g(q) = tau. The mass matrix M depends on q2 alone through cos(q2), which makes physical sense: the arm's apparent inertia depends on how folded it is, not on which way it points. Gravity does not appear in T at all; it enters only through the potential energy derivative.

The Coriolis terms are the part I had to check twice. C collects the velocity products: a -sin(q2)(2 qd1 qd2 + qd2^2) term on joint 1 and a +sin(q2) qd1^2 term on joint 2. The signs matter and they are not symmetric. Getting one wrong does not throw an error; it leaks energy.

Which is exactly what the tests catch. At zero velocity the equation reduces to M qdd = tau - g, so a holding torque equal to gravity_vector should produce zero acceleration. The better test drops gravity and torque entirely and simulates with the RK4 from integrators.py: with nothing adding or removing energy, the kinetic energy (1/2) qd^T M qd should stay constant. It does, to one part in a million over two thousand steps, and that test would fail loudly if the Coriolis signs were flipped.

Where this breaks: point masses are a lie. Real links have distributed mass, which adds an inertia term and shifts the centers of mass inward, and real joints have friction that this model ignores. The energy conservation test passes precisely because the friction is absent. Put this in front of real hardware and the simulated arm will overshoot every trajectory the actual arm executes.
