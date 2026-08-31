# 14 ODE Systems

`calccode/ode_systems.py` runs the existing RK4 on three classic systems, and each one demonstrates a different claim from the theory.

The nonlinear pendulum first. At a release angle of 0.05 rad the measured period matches the small-angle formula 2 pi sqrt(L/g) to within a tenth of a percent. At 1.0 rad the period is about 6% longer. The small-angle model is not wrong, it is a linearization, and the code shows exactly where it stops holding.

Lotka-Volterra is the structure case. The system conserves H = delta x - gamma ln x + beta y - alpha ln y along every orbit, and RK4 at h = 0.005 keeps that invariant flat to one part in 1e4 over 10000 steps. The orbits close. Populations cycle forever with no damping and no forcing, which still feels wrong for a model of real animals, but the math does what it does.

Lorenz is the chaos case. Two trajectories starting 1e-6 apart stay together for about 25 time units, then separate to order-one distance by t = 30. The growth is exponential until it saturates at the size of the attractor. This is the butterfly effect as a measured slope, not a metaphor.

Where this breaks: chaotic systems make long-time pointwise accuracy meaningless. Any integrator on Lorenz produces a trajectory that diverges from the true one at the same exponential rate; only shadowing properties and statistics survive. Also, RK4 does not conserve the Lotka-Volterra invariant exactly, it just drifts slowly. For million-step predator-prey runs I would want a symplectic or invariant-preserving scheme, the same lesson the harmonic oscillator taught in integrators.py.
