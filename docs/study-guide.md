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
| Optimization (critical points) | `calccode/optimize.py`, `calccode/gradient.py` | `notes/12-optimize.md`, `notes/05-gradient-descent.md` | `exercises/ex_optimize.py` |
| L'Hopital's rule | `calccode/limits.py` (numeric check of 0/0 forms) | `notes/01-limits.md` | `exercises/ex_limits.py` |
| Riemann sums | `calccode/integrals.py` | `notes/07-integrals.md` | `exercises/ex_integrals.py` |
| Fundamental theorem of calculus | `calccode/integrals.py` with `calccode/derivatives.py`; `calccode/symbolic_integrate.py` (FTC part 2 on antiderivative trees) | `notes/07-integrals.md`, `notes/19-symbolic-integration.md` | `exercises/ex_integrals.py` |
| Antiderivatives (power, trig, exp rules) | `calccode/symbolic_integrate.py` | `notes/19-symbolic-integration.md` | none yet |
| Series convergence | `calccode/series.py` (numeric ratio test) | `notes/08-series.md` | `exercises/ex_series.py` |
| Taylor polynomials | `calccode/series.py` | `notes/08-series.md` | `exercises/ex_series.py` |

## Beyond calculus

| Topic | Module | Note | Exercise |
| --- | --- | --- | --- |
| Linear algebra | `calccode/linalg.py` | `notes/09-linalg.md` | `exercises/ex_linalg.py` |
| Robotics transforms | `calccode/transforms.py` | `notes/10-transforms.md` | `exercises/ex_transforms.py` |
| Fourier analysis | `calccode/fourier.py` | `notes/15-fourier.md` | none yet |
| Monte Carlo | `calccode/montecarlo.py` | `notes/16-monte-carlo.md` | none yet |
| Interpolation | `calccode/interpolation.py` | `notes/17-interpolation.md` | none yet |
| Eigenvalues | `calccode/eigen.py` | `notes/18-eigen.md` | none yet |

## Not covered yet

Honest gaps, in the order I would tackle them:

- Related rates. No module sets up two quantities tied by one equation and differentiates both sides with respect to time.
- Integration by parts as a symbolic technique. `symbolic_integrate.py` covers the basic rules but has no product rule in reverse, so x sin(x) is refused.
- Implicit differentiation past the formula level. `implicit_diff` builds -Fx/Fy from the partials; it does not solve for higher derivatives or handle curves where Fy = 0.
- u-substitution as a symbolic technique. `symbolic_integrate.py` raises NotImplementedError the moment a composite argument appears; guessing a substitution needs pattern matching the tree does not attempt.
- Sequences and series past Taylor: no root test, integral test, or power series radius work.
- Differential equations past RK4: no stiff solvers, no boundary value problems.
