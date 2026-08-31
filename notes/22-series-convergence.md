# 22 Series convergence

`calccode/convergence.py` picks up where the ratio test in series.py gives up. The ratio test is silent whenever the ratio of consecutive terms tends to 1, which is every p-series, so this module implements the tests that decide those cases.

The alternating series bound is the most satisfying one to code. The theorem says the error after n terms is at most the size of the first omitted term, so the code is one line: return abs(term(n + 1)). The test sums the alternating harmonic series and checks the error against the true value ln 2 at n = 10, 100, 1000. The bound holds every time, with room to spare.

The integral test needed more care. Comparing sum 1/k^p against the integral of x^-p sounds easy, but Simpson's rule over [1, 100000] with a fixed panel count wastes most of its panels on the flat tail and loses accuracy near x = 1 where the function dives. My fix was to integrate decade by decade, [1, 10], [10, 100], and so on, with the same panel count in each decade. The verdict logic then reads the growth: if the last decade adds less than 1% of the running total, the integral has settled and the series converges. For p = 2 the last decade adds about 9e-6 against a total near 1. For p = 1 each decade adds another ln 10 forever. The verdicts come out exactly where the theory puts them, including the boundary at p = 1.

The comparison test is a glorified loop. It checks 0 <= term(k) <= benchmark(k) over a range of k and reports the worst ratio. For 1/(n^2 + 1) against 1/n^2 the ratio stays under 1 and creeps up toward it, which is the whole story: the smaller series inherits convergence from the bigger one.

Where this breaks: numerical partial sums can suggest divergence but never prove it. The harmonic series is the painful case. Its partial sums grow like ln n, so after a million terms the sum is only about 14, and doubling that takes a squared term count. Worse, once 1/k falls below half the ulp of the running total, float64 stops registering new terms at all; the sum freezes at a finite value even though the real series is unbounded. A computer sees a finite sum at every finite n, always.
