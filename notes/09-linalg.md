# 09 Linear Algebra

`calccode/linalg.py` is matrix code with training wheels removed. matmul is a triple loop. The determinant comes from Gaussian elimination: eliminate to upper triangular form, multiply the diagonal, flip the sign for each row swap. Solve is elimination with partial pivoting followed by back substitution. Rank counts nonzero pivots. No numpy.linalg anywhere.

Partial pivoting turned out to matter for a reason I had not appreciated. My first pass worked on every test matrix I threw at it, then failed on an almost singular system where the natural pivot was tiny. Dividing by a near zero pivot amplifies rounding error catastrophically. Swapping in the largest available pivot before each elimination step is a one line change and it makes the solver stable in practice. The theory says Gaussian elimination without pivoting is unstable; the code says it too, loudly, once you test the right matrix.

The determinant identity det(AB) = det(A) det(B) makes a great test. Two random 4x4 matrices, compute the product by hand with matmul, take determinants three ways, and they agree to 1e-8. That kind of structural check catches bugs that individual value checks miss.

Where this breaks: elimination based rank is fuzzy by nature. A pivot of 1e-13 on a matrix whose entries are order 1 is zero for practical purposes, but the cutoff is a judgment call. Singular values would give a cleaner answer, and computing those by hand is a much bigger project. Also, my solve is O(n^3) with a big constant because every row update is a Python loop over a numpy slice. Fine for n = 20. Not for n = 20000. numpy.linalg exists for a reason, and now I know exactly what it is doing inside.
