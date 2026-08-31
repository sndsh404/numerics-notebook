# 23 Probability

`calccode/probability.py` builds probability on top of two pieces I already had: the xorshift32 generator from montecarlo.py and Simpson's rule from integrals.py. Everything else is a loop or a formula. Uniforms come straight from the generator. Exponentials come from the inverse CDF, x = -ln(1 - u) / lam. Normals come from Box-Muller: two uniforms in, two normals out, via a radius from -2 ln(u1) and an angle from 2 pi u2. Fifty thousand seeded draws land within a few hundredths of the target mean and standard deviation, which is all I ask of a teaching RNG.

The densities are one-liners. The cumulative functions are where the work is. The exponential CDF has a closed form, 1 - e^(-lam x), so that one is exact. The normal CDF has no closed form at all, so I compute it the honest way: standardize to z, then 0.5 plus Simpson's rule on the standard density from 0 to z. Phi(0) comes out 0.5 to machine precision, Phi(1) hits 0.8413, and the symmetry F(-x) = 1 - F(x) holds to 1e-10 because I integrate the short side.

Expected value and variance of a continuous distribution are quadrature too: integrate x f(x) and (x - mu)^2 f(x) over a wide enough interval. On an exponential with lam = 2 the code returns 0.5 and 0.25 to six figures, which is the definition of mean and variance made executable.

The central limit demo is my favorite part. Take 30 exponential draws, average them, repeat a few thousand times. The population is a hard right skew; the means pile up into a bell centered at 1 / lam with spread 1 / (lam sqrt(n)), and the predicted normal curve sits on the histogram with no fitting.

Where this breaks: the normal CDF by quadrature is accurate but slow, roughly a thousand density evaluations per call where a polynomial approximation costs a dozen flops. This is exactly why real libraries ship erf approximations instead of integrating on demand. My version is for reading, not for production.
