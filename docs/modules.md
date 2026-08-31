# Module index

One line per module: what it implements, and links to the matching note, test file, and exercise file. Exercise files are stubs; the reference implementation is the module itself. Modules without an exercise have "none" in that column.

| Module | What it implements | Note | Tests | Exercise |
| --- | --- | --- | --- | --- |
| `calccode/limits.py` | Numerical limits from both sides, divergence and oscillation detection | [01](../notes/01-limits.md) | [test_limits.py](../tests/test_limits.py) | [ex_limits.py](../exercises/ex_limits.py) |
| `calccode/derivatives.py` | Forward, backward, central differences, convergence order fitting | [02](../notes/02-derivatives.md) | [test_derivatives.py](../tests/test_derivatives.py) | [ex_derivatives.py](../exercises/ex_derivatives.py) |
| `calccode/symbolic.py` | Expression trees, `diff()` with power, product, and chain rules | [03](../notes/03-symbolic.md) | [test_symbolic.py](../tests/test_symbolic.py) | [ex_derivatives.py](../exercises/ex_derivatives.py) |
| `calccode/integrals.py` | Riemann sums, trapezoid, Simpson, empirical order checks | [07](../notes/07-integrals.md) | [test_integrals.py](../tests/test_integrals.py) | [ex_integrals.py](../exercises/ex_integrals.py) |
| `calccode/symbolic_integrate.py` | Rule-based antiderivatives, linear u-substitution, parts, FTC definite integrals | [19](../notes/19-symbolic-integration.md) | [test_symbolic_integrate.py](../tests/test_symbolic_integrate.py) | none |
| `calccode/applications.py` | Arc length, polar and parametric arc length, disk and shell volumes, surface area | [20](../notes/20-applications.md) | [test_applications.py](../tests/test_applications.py) | [ex_applications.py](../exercises/ex_applications.py) |
| `calccode/related_rates.py` | Chain rule in time on relation trees, linear solve for one unknown rate | [21](../notes/21-related-rates.md) | [test_related_rates.py](../tests/test_related_rates.py) | none |
| `calccode/series.py` | Taylor polynomials, partial sums, numeric ratio test | [08](../notes/08-series.md) | [test_series.py](../tests/test_series.py) | [ex_series.py](../exercises/ex_series.py) |
| `calccode/convergence.py` | Alternating error bound, integral test, comparison test, p-series verdict | [22](../notes/22-series-convergence.md) | [test_convergence.py](../tests/test_convergence.py) | [ex_series.py](../exercises/ex_series.py) |
| `calccode/integrators.py` | Explicit Euler, semi-implicit Euler, RK4 | [04](../notes/04-integrators.md) | [test_integrators.py](../tests/test_integrators.py) | none |
| `calccode/optimize.py` | Bisection, Newton, secant; golden section, Armijo line search, Newton and BFGS minimizers, momentum, Nesterov, Adam | [12](../notes/12-optimize.md), [29](../notes/29-minimization.md) | [test_optimize.py](../tests/test_optimize.py) | [ex_optimize.py](../exercises/ex_optimize.py) |
| `calccode/multivar.py` | Partials, gradient, Jacobian, Hessian, gradient checking | [13](../notes/13-multivariable.md) | [test_multivar.py](../tests/test_multivar.py) | [ex_multivar.py](../exercises/ex_multivar.py) |
| `calccode/gradient.py` | 1D and 2D descent on central differences, learning rate study | [05](../notes/05-gradient-descent.md) | [test_gradient.py](../tests/test_gradient.py) | none |
| `calccode/autograd.py` | Scalar reverse-mode autograd, MLP that fits sin(x) | [06](../notes/06-autograd.md) | [test_autograd.py](../tests/test_autograd.py) | [ex_autograd.py](../exercises/ex_autograd.py) |
| `calccode/linalg.py` | Matmul, determinant, Gaussian elimination, rank; no numpy.linalg | [09](../notes/09-linalg.md) | [test_linalg.py](../tests/test_linalg.py) | [ex_linalg.py](../exercises/ex_linalg.py) |
| `calccode/transforms.py` | Rotations, Rodrigues' formula, quaternions, planar arm kinematics | [10](../notes/10-transforms.md) | [test_transforms.py](../tests/test_transforms.py) | [ex_transforms.py](../exercises/ex_transforms.py) |
| `calccode/kinematics.py` | Screw theory, PoE forward kinematics, Jacobians, damped least squares IK, time scaling | [24](../notes/24-modern-robotics.md) | [test_kinematics.py](../tests/test_kinematics.py) | none |
| `calccode/dynamics.py` | 2-link planar arm Lagrangian dynamics: mass matrix, Coriolis and gravity terms, forward dynamics | [26](../notes/26-dynamics.md) | [test_dynamics.py](../tests/test_dynamics.py) | none |
| `calccode/regression.py` | Least squares two ways, logistic regression, ridge closed form | [11](../notes/11-regression.md) | [test_regression.py](../tests/test_regression.py) | none |
| `calccode/ml.py` | Perceptron, k-NN, Gaussian naive Bayes, train/test split, k-fold, metrics | [25](../notes/25-classical-ml.md) | [test_ml.py](../tests/test_ml.py) | none |
| `calccode/ode_systems.py` | Nonlinear pendulum, Lotka-Volterra, Lorenz attractor on RK4 | [14](../notes/14-ode-systems.md) | [test_ode_systems.py](../tests/test_ode_systems.py) | none |
| `calccode/fourier.py` | Hand-written O(n^2) DFT, inverse, dominant frequency detection | [15](../notes/15-fourier.md) | [test_fourier.py](../tests/test_fourier.py) | [ex_fourier.py](../exercises/ex_fourier.py) |
| `calccode/montecarlo.py` | Xorshift RNG, 1D and n-D integration, pi estimation, importance sampling | [16](../notes/16-monte-carlo.md) | [test_montecarlo.py](../tests/test_montecarlo.py) | [ex_montecarlo.py](../exercises/ex_montecarlo.py) |
| `calccode/probability.py` | Box-Muller normals, PDFs and CDFs, percentiles, CLT demo | [23](../notes/23-probability.md) | [test_probability.py](../tests/test_probability.py) | none |
| `calccode/interpolation.py` | Barycentric Lagrange, divided differences, cubic splines, Chebyshev nodes | [17](../notes/17-interpolation.md) | [test_interpolation.py](../tests/test_interpolation.py) | [ex_interpolation.py](../exercises/ex_interpolation.py) |
| `calccode/eigen.py` | Power iteration, inverse iteration with shift, deflation | [18](../notes/18-eigen.md) | [test_eigen.py](../tests/test_eigen.py) | [ex_eigen.py](../exercises/ex_eigen.py) |
| `calccode/orthogonal.py` | Gram-Schmidt both ways, Householder QR, Jacobi SVD, pseudoinverse, rank, low-rank approximation, least squares | [28](../notes/28-orthogonalization.md) | [test_orthogonal.py](../tests/test_orthogonal.py) | none |
| `calccode/vector_calculus.py` | Double and triple integrals, line and surface integrals, Green, divergence, and Stokes checkers, Lagrange multipliers | [30](../notes/30-vector-calculus.md) | [test_vector_calculus.py](../tests/test_vector_calculus.py) | none |
