# 15 Fourier

`calccode/fourier.py` is the DFT as its definition: X_k equals the sum of x_n exp(-2 pi i k n / N), one complex multiply-accumulate at a time. O(n^2), no numpy.fft. The inverse flips the sign and divides by N. A round trip through both returns the input to 1e-10.

The tests lean on exact structure. A pure tone of amplitude 2 at bin 4 puts magnitude N in that bin and zeros (to 1e-8) in its neighbors. The amplitude spectrum 2|X_k|/N reads off 1.5 and 0.7 for a two-tone signal, matching the amplitudes I put in. Dominant frequency detection works even with seeded noise on top, because the tones own their bins and the noise spreads thin over all of them.

Writing the double loop makes the FFT feel inevitable. Every output reuses the same roots of unity, and half the work is repeated with a sign flip. The radix-2 FFT is this same sum with the redundancy factored out, and after doing it the slow way I can see exactly where the savings come from.

Where this breaks: resolution and leakage. Frequency resolution is sample_rate / N, full stop. A tone that lands between bins smears across all of them, and no amount of code fixes a short record. The O(n^2) cost is also real: N = 512 is instant, N = 100000 is 1e10 operations and a coffee break. For anything past toy signals the FFT exists for a reason. There is also a subtlety I tripped on: the DC term must not be doubled in the one-sided spectrum, or every DC offset reads twice its true value. Small convention, wrong numbers.
