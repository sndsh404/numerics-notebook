# 16 Monte Carlo

`calccode/montecarlo.py` starts with its own random numbers: xorshift32, three shift-and-xor rounds, seeded and deterministic. No imports beyond math and numpy. Integration is the mean of f over uniform samples times the volume of the domain. Pi comes from throwing darts at the quarter circle.

The headline result is the error scaling. The standard deviation of pi estimates across 25 seeds falls like 1/sqrt(n), and a log-log fit of spread against sample count comes out near -0.5. That exponent is the deal Monte Carlo offers: slow convergence, but the same exponent in every dimension. Quadrature rules from integrals.py degrade exponentially with dimension; this method does not care. In ten dimensions, random sampling wins.

The importance sampling demo is the clean case. Integrating 1/sqrt(x) on (0, 1] by uniform sampling has awful variance because the singularity hogs the integral. Sample instead from p(x) = 1/(2 sqrt(x)) via the inverse CDF x = u^2 and the ratio f/p becomes the constant 2. A constant integrand has zero variance. The estimate is exact at n = 100.

Where this breaks: everything depends on the proposal. Matched density, zero variance. Bad density, worse than uniform. And the 1/sqrt(n) law means each extra digit of accuracy costs 100 times the samples, so Monte Carlo is for hard problems, not precision work. The other trap is RNG quality. My xorshift32 fails strict statistical test suites; it is fine for these demos, and I would not fly it anywhere near cryptography or a serious simulation. Writing it by hand at least made its structure, and its 2^32 period limit, something I can see.
