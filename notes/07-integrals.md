# 07 Integrals

`calccode/integrals.py` covers the classic quadrature rules: left and right Riemann sums, midpoint, trapezoid, and Simpson. Each is one loop over panel evaluations. The interesting part is measuring how fast each one converges.

The convergence study runs every rule at n = 8, 16, 32, up to 512 panels on exp over [0, 1], then reuses the log-log slope fit from derivatives.py. The measured orders match the textbook: midpoint and trapezoid come out at 2.0, Simpson at 4.0. Halving the panel width quarters the trapezoid error and divides Simpson's by sixteen. At n = 128 Simpson beats trapezoid by more than three orders of magnitude on a smooth function.

Where this breaks: singular integrands. The integral of 1/sqrt(x) over (0, 1] is 2, and midpoint can attempt it because it never touches the endpoint. But the error decays like 1/sqrt(n) instead of 1/n^2. Going from 100 to 10000 panels, a hundred times more work, buys only one digit of accuracy. The singularity eats the convergence order. The fix is a substitution: x = t^2 turns the integrand into the constant 2, which midpoint nails to machine precision immediately. Same integral, same rule, different parametrization.

What I take from this: quadrature order is a property of the rule and the smoothness of the integrand together. A fourth order method is only fourth order when the function cooperates. Checking the empirical order on a smooth function is how you verify the code; checking it on a nasty function is how you learn the theory.
