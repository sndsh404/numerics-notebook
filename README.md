# Calculus to Code

I am learning Calculus I and sitting the CLEP calculus exam in December. This repo is how I study: I implement every topic from scratch in Python.

The rule of the repo: numpy is allowed for arrays and plotting data, but no library does the math for me. No scipy, no sympy, no numpy.gradient. Every derivative, integral, and gradient step is hand-written.

The repo also doubles as a portfolio piece for robotics and simulation work, so the code has type hints, docstrings where they earn their place, and pytest coverage for every module.

## Layout

- `calccode/limits.py`: numerical limits, one-sided limits, divergence and oscillation detection
- `calccode/derivatives.py`: forward, backward, and central differences plus a convergence study
- `calccode/symbolic.py`: expression trees and a hand-written `diff()` with the power, product, and chain rules
- `calccode/integrators.py`: explicit Euler, semi-implicit Euler, and RK4 for ODEs
- `calccode/gradient.py`: gradient descent in 1D and 2D on central differences
- `calccode/autograd.py`: scalar reverse-mode autograd and a small MLP that fits sin(x)
- `notes/`: my study notes, one per module
- `scripts/make_plots.py`: regenerates every plot into `plots/`

## Run it

```bash
python -m pytest tests/ -q
python scripts/make_plots.py
```

Plots land in `plots/`, which is gitignored.
