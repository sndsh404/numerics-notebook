# 12 Root Finding and Optimization

`calccode/optimize.py` has the three classic root finders: bisection, Newton, and secant. Newton gets its derivative from the central difference in derivatives.py, so none of the methods need a formula for f'. Each one returns its full x history, which turns convergence order from a claim into a measurement.

The iteration counts tell the story. On x^2 - 2 with a [1, 2] bracket, bisection needs about 33 halvings to reach 1e-10 because the interval only shrinks by half each step. Newton gets there in 4 iterations. Fitting log(e_{n+1})/log(e_n) on the error history comes out at 2.1, the quadratic convergence the theory promises: each step roughly doubles the number of correct digits. The secant method lands in between at about 1.6, which is exactly the golden ratio, one of my favorite constants showing up uninvited.

Where this breaks: Newton from a bad start. On arctan(x), starting past about 1.39 makes each iterate larger than the last because the tangent line overshoots and the shallow slope at large x sends the next guess even farther out. Starting at 3.0 the iterates blow past 30 within a few steps and the method reports converged = False. The same method that doubles digits near a root amplifies error far from one. Bisection never does this, because the bracket is a cage. That tradeoff, speed with no guarantee against slowness with a cage, is the whole lesson of the module.

The other surprise is how the failure is visible in the history. A converging run shows errors collapsing; a diverging run shows them doubling. Logging the iterates costs nothing and turns debugging from guessing into reading.
