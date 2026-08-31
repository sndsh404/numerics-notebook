# 31 FFT

`calccode/fourier.py` now has the transform the DFT note promised: `fft` computes the exact same sum as `dft`, but factors out the redundancy instead of re-evaluating every root of unity. The layout is iterative. One pass permutes the input into bit-reversed order, then butterflies of width 2, 4, and so on up to n combine pairs in place. I went iterative on purpose. The recursive version is prettier, but note 06's autograd lesson applies: recursion in Python is a depth budget, and a large n should not depend on it. `ifft` is one line of conjugation on top: conj(fft(conj(X))) / N.

The payoff is measured, not asserted. `fft_vs_dft_sizes` times both on Xorshift32-drawn signals. At n = 2048 the direct DFT takes 1.8 seconds; the FFT takes 7 milliseconds. That is a factor of about 250 at a toy size, and the gap widens with n because the curves are n^2 against n log n. The figure `docs/img/fft_vs_dft.png` shows both hugging their reference slopes. Correctness checks run the FFT against the plain dft to 1e-10 on sizes from 1 to 1024, complex input included.

The bit reversal is the part I had to slow down for. Output position i reads input position i with its bits reversed, because each butterfly stage splits even from odd indices, recursively. Writing the two-loop reversal by hand is what made the in-place structure click.

Where this breaks: radix-2 only. A signal of length 1000 is refused outright; `pad_to_power_of_2` zero-pads it to 1024 first. Padding gives the exact DFT of the padded sequence, but that is a different sequence: the zeros change the spectrum and can smear tones across bins. The honest fix for arbitrary n is a mixed-radix or Bluestein transform, and this module has neither. For the sizes I actually study, padding is fine.
