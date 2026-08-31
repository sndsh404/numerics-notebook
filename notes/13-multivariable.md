# 13 Multivariable Calculus

`calccode/multivar.py` lifts the one-dimensional derivative machinery to several variables. A partial derivative is a central difference along one coordinate with the others frozen. The gradient stacks the partials. The Jacobian does the same for vector-valued functions, one column per input. The Hessian adds second differences, including the mixed partial (f(x+h, y+k) - f(x+h, y-k) - f(x-h, y+k) + f(x-h, y-k)) / 4hk.

The directional derivative ties it together: project the gradient onto a unit direction and you get the slope in that direction. The test checks this identity directly, grad(f) . u, and it holds to finite difference accuracy.

The gradient check is the piece I will reuse. It takes an expression tree from symbolic.py, computes exact partials with a new eval_multi that looks each variable up by name, and compares them against central differences on the same tree. Largest disagreement on three test functions came in under 1e-5. This is precisely the sanity check people run on hand-written backprop code, and now I have it for free because the earlier modules compose.

Where this breaks: the cost and the noise both scale badly. A gradient in n dimensions costs 2n function evaluations, a Hessian costs O(n^2) of them, and each second difference divides by h^2, so the roundoff valley from the derivatives module gets deeper and narrower. My Hessian tests run at h = 1e-4, not 1e-5, because the h^2 in the denominator punishes small steps. For the autograd module the gradient is one backward pass regardless of n, which is exactly why nobody trains networks with finite differences. Finite differences are the reference you check against, not the tool you compute with.
