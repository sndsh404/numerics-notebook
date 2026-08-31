# 02 Derivatives

`calccode/derivatives.py` has the three finite differences: forward, backward, central. Each is one line. The module earns its keep in `convergence_study`, which measures the error against a known derivative over a range of h values, and `fit_order`, which fits the log-log slope.

The theory says forward difference error is O(h) and central is O(h^2), because the central version's odd error terms cancel. The test suite checks this empirically: on sin(x) at x = 1, the fitted slopes come out at 1.0 and 2.0 within a tenth. Halving h quarters the central difference error. That part works exactly as the textbook says.

What surprised me: h too small is worse, not better. I assumed smaller steps always help. They do until about 1e-6 for exp at x = 1, and then the error starts growing again. The reason is cancellation. f(x + h) and f(x - h) agree to about 15 digits when h is 1e-12, so their difference keeps maybe 3 significant digits, and dividing by 2h amplifies the noise. The error curve is a valley: truncation error falls as h^2, roundoff error rises as 1/h, and the sweet spot for a central difference sits near h = eps^(1/3), around 1e-5 for doubles.

This changed how I think about the definition of the derivative. The limit h to 0 is a statement about exact arithmetic. Floating point never gets there. So a numerical derivative is always a tradeoff, and "smaller h" is only right on the way down the valley.

The convergence test is also the first place I used a log-log slope as an assertion instead of an eyeball check. Fitting the order and asserting it lands in (1.9, 2.1) is a much stronger statement than checking one error value, and it would catch a subtle bug like a missing factor of 2 that a single-point test could miss.
