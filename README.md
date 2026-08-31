# Calculus to Code

I am learning calculus and the math behind robotics and machine learning. This repo is how I study: I implement every topic from scratch in Python. It started as Calculus I prep for the CLEP exam in December, then grew to cover Calc II topics from Stewart and Strang, linear algebra for neural networks, rigid body transforms from Modern Robotics, and basic regression.

The rule of the repo: numpy is allowed for arrays and plotting data, but no library does the math for me. No scipy, no sympy, no numpy.linalg, no numpy.gradient. Every derivative, integral, matrix solve, and gradient step is hand-written.

The repo also doubles as a portfolio piece for robotics and simulation work, so the code has type hints, docstrings where they earn their place, and pytest coverage for every module.

## Layout

Calculus:

- `calccode/limits.py`: numerical limits, one-sided limits, divergence and oscillation detection
- `calccode/derivatives.py`: forward, backward, and central differences plus a convergence study
- `calccode/symbolic.py`: expression trees and a hand-written `diff()` with the power, product, and chain rules
- `calccode/integrals.py`: Riemann sums, trapezoid, and Simpson with empirical order checks
- `calccode/series.py`: Taylor polynomials, partial sums, and a numeric ratio test
- `calccode/integrators.py`: explicit Euler, semi-implicit Euler, and RK4 for ODEs
- `calccode/gradient.py`: gradient descent in 1D and 2D on central differences
- `calccode/autograd.py`: scalar reverse-mode autograd and a small MLP that fits sin(x)

Linear algebra and robotics:

- `calccode/linalg.py`: matmul, determinant, Gaussian elimination solve, and rank, all hand-written
- `calccode/transforms.py`: 2D and 3D rotations, Rodrigues' formula, homogeneous transforms, quaternions, planar arm kinematics

Machine learning:

- `calccode/regression.py`: least squares by normal equations and by gradient descent, plus logistic regression

Also:

- `notes/`: my study notes, one per module
- `scripts/make_plots.py`: regenerates the calculus plots into `plots/`
- `scripts/make_plots_2.py`: regenerates the Calc II and ML plots into `plots/`

## Run it

```bash
python -m pytest tests/ -q
python scripts/make_plots.py
python scripts/make_plots_2.py
```

Plots land in `plots/`, which is gitignored.
