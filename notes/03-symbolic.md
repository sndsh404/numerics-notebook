# 03 Symbolic Differentiation

`calccode/symbolic.py` builds expressions as trees: Const, Var, Add, Mul, Pow, Sin, Cos, Exp, Log. Each node knows three things: how to evaluate itself, how to differentiate itself, and how to simplify. `diff()` just asks the root node for its derivative and simplifies what comes back.

The rules map one to one onto the class. Power rule lives in Pow, product rule in Mul, chain rule falls out for free because every composite node multiplies by the derivative of its argument. There is no quotient rule. A quotient sin(x)/x is a product with Pow(x, -1), and the product and power rules handle it. I checked all of these against central differences from the previous module at random points, and they agree to finite difference accuracy.

The simplifier does constant folding and strips identities: x + 0, 1 * x, 0 * anything, x^1. So d/dx x^2 prints as (2 * x), not ((2 * (x^1)) * 1).

What surprised me: how fast the trees blow up without simplification. Differentiate a product of three terms and the product rule hands you a sum where each term contains the full original tree again. Differentiate that and it happens recursively. The size grows exponentially in the number of diffs if you do not fold constants aggressively. And even with folding, my simplifier is weak. It does not combine 2x + 3x, does not cancel x/x, does not sort terms. A real computer algebra system spends most of its code on simplification, not on the rules themselves.

This is where the capstone starts to make sense. Symbolic differentiation gives you a formula, exact but potentially enormous. Finite differences give you a number, cheap but approximate. The thing I actually want for training a model is exact gradients at one point, computed without ever building a formula. That is reverse-mode autograd, and it only took writing this module to see why someone had to invent it.
