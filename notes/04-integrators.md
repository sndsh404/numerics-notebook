# 04 Integrators

`calccode/integrators.py` solves y' = f(t, y) three ways: explicit Euler, semi-implicit Euler, and RK4. Same interface for all three: give me f, a start, a step h, a step count, get back the full time history.

On dx/dt = -kx the ordering is exactly what the theory predicts. With h = 0.05 and k = 2 over one second, Euler lands about 0.014 off the exact answer. RK4 at the same step count lands about 1e-8 off. The test asserts RK4 beats Euler by a factor of at least 1000 at equal cost per step.

The demo I care about is the harmonic oscillator, because that is the one that matters for simulation work. Explicit Euler has update matrix with eigenvalues of modulus sqrt(1 + h^2), so every step multiplies the energy by (1 + h^2). At h = 0.05 over 10000 steps the energy grows by a factor near e^12. The orbit spirals out. Semi-implicit Euler, where you update velocity first and then use the new velocity to move the position, is symplectic: its update matrix has determinant 1. The energy oscillates slightly but stays bounded forever. Same test, 10000 steps, energy stays within a factor of 1.5 of the start while explicit Euler is off the chart.

What surprised me: the fix is one line. Swapping the order of two updates turns a method that destroys the physics into one that preserves it for millions of steps. The accuracy per step is the same first order. What changes is structure, not precision.

Where this breaks: semi-implicit Euler as I wrote it assumes the state is [positions, velocities] with forces independent of velocity. Add damping or a magnetic field and the kick-drift split needs care. Also RK4, for all its accuracy, is not symplectic either. Over long enough horizons it drifts too. My 10000 step test shows RK4 holding energy to 1e-6, but run it ten million steps and it slowly decays. For game physics and robot simulation, people pick symplectic integrators on purpose. Now I know why.
