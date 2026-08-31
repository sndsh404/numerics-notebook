# 01 Limits

A limit asks what f does near a point, not at it. In `calccode/limits.py` I do the dumbest honest version of that: evaluate f at a geometric sequence of offsets, 0.1, 0.025, and so on down to about 1e-15, and watch the last few values.

The classifier has four outcomes. If the tail of the sequence settles within a small relative spread, I call it converged and take the median of the last five samples as the estimate. If the magnitudes grow monotonically by a big factor, it diverges. If the values stay bounded but never settle, like sin(1/x) at 0, I call it oscillating. And if any sample hits a domain error, the result is undefined, which is what sqrt(x) does from the left of 0.

The two-sided wrapper just runs both directions and compares. 1/x at 0 diverges to plus infinity from the right and minus infinity from the left, so the two-sided limit does not exist. sin(x)/x at 0 converges to 1 from both sides, and the estimate lands within 1e-3 of 1 without ever evaluating at 0 itself.

What surprised me: log(1/x) near 0 is divergent, but it grows so slowly that my classifier reads it as oscillating. Over the full sample range it only climbs from about 2 to about 32, never fast enough to trip the growth rule. The fix would be more samples or a weaker growth threshold, but both make genuinely oscillating functions harder to catch. A numeric limit is a claim about a finite set of samples, and any function pathological enough can fool it. That is exactly why the epsilon-delta definition exists: it says something about every neighborhood, not just the ones I happened to sample.

The other takeaway is practical. One-sided limits are cheap and honest. Checking both sides separately caught the 1/x sign flip for free, which a single blind sample run would have averaged into nonsense.
