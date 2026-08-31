# 08 Series

`calccode/series.py` builds Taylor polynomials two ways. The exact path differentiates an expression tree from symbolic.py repeatedly and evaluates f^(k)(a)/k!. The numeric path applies central differences recursively, which costs 2^n evaluations per coefficient and compounds roundoff at each level. It works for degree 5 or 6. Beyond that the symbolic path is the only sensible option, which was a satisfying confirmation of why the earlier modules exist.

The Taylor tests check what the theory promises. The degree 11 polynomial for sin at 0 matches sin(0.5) to 1e-9. The degree 10 polynomial for exp reproduces e at x = 1 to 1e-7. But move away from the expansion point and the error explodes: at x = 3 the degree 5 sin polynomial is off by more than 100 times the error at x = 1. A Taylor polynomial is a local object. Raising the degree extends the accurate region, and for sin and exp it eventually converges everywhere, but each fixed degree has a horizon.

The ratio test in code is a humbling exercise. On geometric series like 0.5^k the ratio sits exactly at 0.5 and the verdict is immediate. On 1/k the ratios march up toward 1 from below and the test honestly reports inconclusive, which matches the theory: the ratio test cannot decide p-series. One implementation detail bit me. I originally required the ratio to settle to a constant, but for 1/k! the ratios shrink toward 0 forever, so "settled" never happens. The verdict has to look at the trend, not just the value.

Where this breaks: factorial terms overflow a float long before the ratios get small, and recursive finite differences lose a digit or two of accuracy per differentiation level. Both are reminders that infinite processes have to be truncated carefully in code.
