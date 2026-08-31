# Study guide

A checklist mapping standard Calculus I and II topics to the code, notes, and exercises in this repo. The study loop: read the note, run the module, do the exercise, compare against the reference in `calccode/`.

## Calculus I and II

| Topic | Module | Note | Exercise |
| --- | --- | --- | --- |
| Limits | `calccode/limits.py` | `notes/01-limits.md` | `exercises/ex_limits.py` |
| Continuity | `calccode/limits.py` (two-sided limit checks) | `notes/01-limits.md` | `exercises/ex_limits.py` |
| Derivative definition | `calccode/derivatives.py` | `notes/02-derivatives.md` | `exercises/ex_derivatives.py` |
| Derivative rules (power, product) | `calccode/symbolic.py` | `notes/03-symbolic.md` | `exercises/ex_derivatives.py` |
| Chain rule | `calccode/symbolic.py` | `notes/03-symbolic.md` | none yet |
| Implicit differentiation | `calccode/symbolic.py` (`implicit_diff`, formula level: -Fx/Fy) | `notes/19-symbolic-integration.md` | none yet |
| Related rates | `calccode/related_rates.py` | `notes/21-related-rates.md` | none yet |
| Optimization (critical points) | `calccode/optimize.py`, `calccode/gradient.py` | `notes/12-optimize.md`, `notes/05-gradient-descent.md` | `exercises/ex_optimize.py` |
| L'Hopital's rule | `calccode/limits.py` (numeric check of 0/0 forms) | `notes/01-limits.md` | `exercises/ex_limits.py` |
| Riemann sums | `calccode/integrals.py` | `notes/07-integrals.md` | `exercises/ex_integrals.py` |
| Fundamental theorem of calculus | `calccode/integrals.py` with `calccode/derivatives.py`; `calccode/symbolic_integrate.py` (FTC part 2 on antiderivative trees) | `notes/07-integrals.md`, `notes/19-symbolic-integration.md` | `exercises/ex_integrals.py` |
| Antiderivatives (power, trig, exp rules) | `calccode/symbolic_integrate.py` | `notes/19-symbolic-integration.md` | none yet |
| Integration by parts | `calccode/symbolic_integrate.py` (x times sin, cos, exp, and ln via 1 * ln x) | `notes/19-symbolic-integration.md` | none yet |
| u-substitution (linear inner function) | `calccode/symbolic_integrate.py` (f(a*x + b) for sin, cos, exp, powers) | `notes/19-symbolic-integration.md` | none yet |
| Arc length and surface area | `calccode/applications.py` | `notes/20-applications.md` | `exercises/ex_applications.py` |
| Volumes of revolution (disk, shell) | `calccode/applications.py` | `notes/20-applications.md` | `exercises/ex_applications.py` |
| Series convergence | `calccode/convergence.py` | `notes/22-series-convergence.md` | `exercises/ex_series.py` |
| Taylor polynomials | `calccode/series.py` | `notes/08-series.md` | `exercises/ex_series.py` |

## Beyond calculus

| Topic | Module | Note | Exercise |
| --- | --- | --- | --- |
| Linear algebra | `calccode/linalg.py` | `notes/09-linalg.md` | `exercises/ex_linalg.py` |
| Robotics transforms | `calccode/transforms.py` | `notes/10-transforms.md` | `exercises/ex_transforms.py` |
| Fourier analysis | `calccode/fourier.py` | `notes/15-fourier.md` | `exercises/ex_fourier.py` |
| Monte Carlo | `calccode/montecarlo.py` | `notes/16-monte-carlo.md` | `exercises/ex_montecarlo.py` |
| Probability and statistics | `calccode/probability.py` | `notes/23-probability.md` | none yet |
| Interpolation | `calccode/interpolation.py` | `notes/17-interpolation.md` | `exercises/ex_interpolation.py` |
| Eigenvalues | `calccode/eigen.py` | `notes/18-eigen.md` | `exercises/ex_eigen.py` |

## Not covered yet

Honest gaps, in the order I would tackle them:

- Implicit differentiation past the formula level. `implicit_diff` builds -Fx/Fy from the partials; it does not solve for higher derivatives or handle curves where Fy = 0.
- Symbolic integration past the narrow patterns in `symbolic_integrate.py`. u-substitution only fires on a linear inner function a*x + b, so sin(x^2) still raises. Integration by parts covers only the single-x set (x times sin, cos, or exp, plus ln(x)), so x^2 e^x is out. No partial fractions, no trig substitution on the symbolic side.
- Improper integrals past the one example in `integrals.py`. The 1/sqrt(x) demo shows a substitution removing a singularity, but there is no general machinery for infinite bounds or singular endpoints.
- Related rates past one unknown. `related_rates.py` solves for a single rate per call and cannot set up the relation itself.
- Sequences and series past the convergence tests in `convergence.py`: no root test or power series radius work.
- Differential equations past RK4: no stiff solvers, no boundary value problems.
- Probability past the basics in `probability.py`: the normal CDF integrates the density per call instead of using an erf approximation, and there is no regression-style inference (no confidence intervals, no hypothesis tests).
