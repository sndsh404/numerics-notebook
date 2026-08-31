# 17 Interpolation

`calccode/interpolation.py` builds the interpolating polynomial three ways. The barycentric form of Lagrange interpolation is the workhorse: precompute weights w_j = 1 / prod_{k != j} (x_j - x_k), then each evaluation is one weighted average, O(n) per point. Newton's divided differences give the same polynomial in nested form, and the test suite confirms the two agree to 1e-8 on a shared grid. Piecewise linear and the natural cubic spline cover the cases where one global polynomial is the wrong tool.

The spline is my favorite piece in this repo. The second derivatives M_i at the knots satisfy a tridiagonal system, and I solve it with the Gaussian elimination from `linalg.py`, so the whole chain stays hand-written. The natural boundary pins M to zero at the ends. The tests check that the spline hits every node, that the second derivative matches across each interior knot, and that it vanishes at both ends.

The Runge demo is the lesson. Interpolating f(x) = 1 / (1 + 25 x^2) on 15 equally spaced nodes of [-1, 1] gives a polynomial that fits the nodes and then oscillates with error above 1 near the ends. Same degree, same function, but Chebyshev nodes (cosine-spaced, clustered at the ends) pull the max error under 0.1. The node placement matters more than the degree.

Where this breaks: high-degree equispaced interpolation diverges for the Runge function as n grows. The error near the ends does not shrink, it grows. The fix is not more nodes but better-placed nodes, or giving up on one global polynomial and using a spline. There is a second trap I hit while testing: the barycentric weights overflow for large n because the denominator product gets tiny. Past a few dozen nodes, splines are the answer and global polynomials are a demo.
