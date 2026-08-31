# Calculus to Code

[![tests](https://github.com/sndsh404/numerics-notebook/actions/workflows/tests.yml/badge.svg)](https://github.com/sndsh404/numerics-notebook/actions/workflows/tests.yml)

A study repository that implements calculus, linear algebra, rigid body transforms, and basic machine learning from scratch in Python. The rule: numpy is allowed for arrays and plotting data, but no library does the math. No scipy, no sympy, no numpy.linalg, no numpy.gradient. Every derivative, integral, matrix solve, and gradient step is hand-written, and every module has pytest coverage.

## Contents

- [Modules](#modules)
- [Exercises](#exercises)
- [Figures](#figures)
- [Quickstart](#quickstart)
- [License](#license)

## Modules

| Module | Topic | What it implements |
| --- | --- | --- |
| `calccode/limits.py` | Limits | Numerical limits from both sides, divergence and oscillation detection |
| `calccode/derivatives.py` | Derivatives | Forward, backward, central differences, convergence order fitting |
| `calccode/symbolic.py` | Symbolic differentiation | Expression trees, `diff()` with power, product, and chain rules |
| `calccode/integrals.py` | Integration | Riemann sums, trapezoid, Simpson, empirical order checks |
| `calccode/symbolic_integrate.py` | Symbolic integration | Rule-based antiderivatives, linear u-substitution, integration by parts, FTC definite integrals, Simpson fallback |
| `calccode/applications.py` | Applications of integration | Arc length (trig substitution for the circle), disk and shell volumes, surface area of revolution |
| `calccode/related_rates.py` | Related rates | Chain rule in time on relation trees, linear solve for one unknown rate |
| `calccode/series.py` | Series | Taylor polynomials, partial sums, numeric ratio test |
| `calccode/convergence.py` | Series convergence | Alternating error bound, integral test, comparison test, p-series verdict |
| `calccode/integrators.py` | ODEs | Explicit Euler, semi-implicit Euler, RK4 |
| `calccode/optimize.py` | Root finding | Bisection, Newton, secant, with iteration histories |
| `calccode/multivar.py` | Multivariable calculus | Partials, gradient, Jacobian, Hessian, gradient checking |
| `calccode/gradient.py` | Gradient descent | 1D and 2D descent on central differences, learning rate study |
| `calccode/autograd.py` | Autodiff | Scalar reverse-mode autograd, MLP that fits sin(x) |
| `calccode/linalg.py` | Linear algebra | Matmul, determinant, Gaussian elimination, rank; no numpy.linalg |
| `calccode/transforms.py` | Robotics transforms | Rotations, Rodrigues' formula, quaternions, planar arm kinematics |
| `calccode/regression.py` | Machine learning | Least squares two ways, logistic regression |
| `calccode/ode_systems.py` | ODE systems | Nonlinear pendulum, Lotka-Volterra, Lorenz attractor on RK4 |
| `calccode/fourier.py` | Fourier analysis | Hand-written O(n^2) DFT, inverse, dominant frequency detection |
| `calccode/montecarlo.py` | Monte Carlo | Xorshift RNG, 1D and n-D integration, pi estimation, importance sampling |
| `calccode/probability.py` | Probability | Box-Muller normals, PDFs and CDFs (normal CDF by Simpson), percentiles, CLT demo |
| `calccode/interpolation.py` | Interpolation | Barycentric Lagrange, divided differences, cubic splines, Chebyshev nodes |
| `calccode/eigen.py` | Eigenvalues | Power iteration, inverse iteration with shift, deflation |

Each module has a matching file in `notes/` with study notes on what the code shows and where it breaks. For a topic-by-topic checklist against a Calculus I/II syllabus, see [docs/study-guide.md](docs/study-guide.md).

![Central limit theorem demo](docs/img/clt_demo.png)

Exponential population on the left, means of 30 draws on the right, with the normal curve the CLT predicts overlaid. From `calccode/probability.py`.

## Exercises

The `exercises/` folder has one file per topic with the function bodies removed. The study loop: read the note, implement the exercise, run its test, compare against the reference in `calccode/`.

```bash
python -m pytest tests/test_exercises.py --run-exercises
```

Exercise tests skip by default so the main suite stays green while exercises are unsolved. See `exercises/README.md` for details.

## Figures

All figures regenerate with the two scripts in `scripts/`.

![Finite difference error vs step size](docs/img/fd_convergence.png)

Forward and central difference errors on sin(x), with the roundoff floor below h = 1e-6.

![ODE integrator comparison](docs/img/integrators.png)

Euler against RK4 on exponential decay, and harmonic oscillator energy over 10000 steps.

![Taylor approximations of sin](docs/img/taylor_sin.png)

Taylor polynomials of growing degree, and how the error grows away from the expansion point.

![Planar arm poses](docs/img/arm_poses.png)

Forward kinematics of a 2-link planar arm at three joint configurations.

![Runge phenomenon](docs/img/runge_phenomenon.png)

Interpolating 1 / (1 + 25 x^2) on 15 equispaced nodes oscillates at the ends; the same degree on Chebyshev nodes tracks the function.

## Quickstart

```bash
git clone https://github.com/sndsh404/numerics-notebook.git
cd numerics-notebook
pip install -r requirements.txt
python -m pytest tests/ -q
python scripts/make_plots.py
python scripts/make_plots_2.py
```

The plot scripts write to `plots/` (gitignored scratch) and `docs/img/` (tracked copies shown above).

## License

MIT. See [LICENSE](LICENSE).
