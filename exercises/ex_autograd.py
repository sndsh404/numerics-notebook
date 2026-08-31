"""Exercise: gradients by hand, the ideas behind autograd.

Reference implementation: calccode/autograd.py.
"""

import math


def relu(x: float) -> float:
    """Rectified linear unit."""
    raise NotImplementedError


def tanh_grad(x: float) -> float:
    """Derivative of tanh at x, in terms of tanh(x)."""
    raise NotImplementedError


def neuron_output(w: list[float], x: list[float], b: float) -> float:
    """tanh(w . x + b), computed with a plain loop."""
    raise NotImplementedError


def mse_loss(preds: list[float], targets: list[float]) -> float:
    """Mean squared error."""
    raise NotImplementedError


def mse_grad(preds: list[float], targets: list[float]) -> list[float]:
    """Gradient of the MSE with respect to each prediction."""
    raise NotImplementedError


def gradient_step(params: list[float], grads: list[float], lr: float) -> list[float]:
    """One descent step: params - lr * grads."""
    raise NotImplementedError
