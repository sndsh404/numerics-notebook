# 05 Gradient Descent

`calccode/gradient.py` is the shortest module in the repo. The gradient is a central difference per coordinate, borrowed straight from `derivatives.py`. The update is x = x - lr * grad in a loop. Everything interesting is in the learning rate.

The test bed is the bowl f(x, y) = (x - 1)^2 + 3(y + 2)^2, minimum at (1, -2), with curvatures 2 and 6 along the axes. Three regimes, all visible in the plots. lr = 0.01 crawls: after 100 steps it is still far from the bottom. lr = 0.1 converges cleanly, faster in the steep direction. lr = 0.5 diverges, and the test asserts the loss grows by six orders of magnitude in 50 steps.

The divergence threshold is not magic. For a quadratic with Hessian eigenvalues up to L, gradient descent is stable only when lr < 2/L. My bowl has L = 6, so the cutoff is 1/3. lr = 0.5 exceeds it and the iterates zigzag apart along the stiff axis while the shallow axis barely moves. That asymmetry is the annoying part: one learning rate has to serve every direction, and the stiffest direction sets the limit while the flattest one sets the speed. The ratio of those curvatures is the condition number, and it is why plain gradient descent is slow on long narrow valleys.

What surprised me: how violent divergence is. I expected overshoot and oscillation. What actually happens is the error doubles or triples per step along the stiff direction, so 50 steps is enough to overflow any reasonable scale. There is no gentle failure mode past the stability boundary.

Where this breaks: the numerical gradient costs 2n function evaluations per step in n dimensions, and each one carries the h valley problem from the derivatives module. Fine for 2D homework. Useless for a neural net with a million weights, which is the entire motivation for the next module. Also, real loss surfaces are not quadratic bowls. Learning rate schedules, momentum, and Adam exist because of everything this simple loop cannot do.
