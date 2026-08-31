"""Scalar reverse-mode autograd, micrograd style, plus a tiny MLP.

Every Value records how it was produced. Calling backward() walks the
graph in reverse topological order and applies the chain rule one local
gradient at a time. This is what symbolic.py cannot do cheaply: the graph
is the derivative, so nothing has to be expanded and simplified.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np


class Value:
    """A scalar with a link to the operations that produced it."""

    def __init__(self, data: float, _children: tuple["Value", ...] = ()):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)

    def __add__(self, other: "Value | float") -> "Value":
        other = _wrap(other)
        out = Value(self.data + other.data, (self, other))

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = _wrap(other)
        out = Value(self.data * other.data, (self, other))

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, n: float) -> "Value":
        if not isinstance(n, (int, float)):
            raise TypeError("only int and float exponents are supported")
        out = Value(self.data**n, (self,))

        def _backward() -> None:
            self.grad += n * self.data ** (n - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,))

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        e = math.exp(self.data)
        out = Value(e, (self,))

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,))

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-_wrap(other))

    def __truediv__(self, other: "Value | float") -> "Value":
        return self * _wrap(other) ** -1.0

    def __radd__(self, other: float) -> "Value":
        return self + other

    def __rmul__(self, other: float) -> "Value":
        return self * other

    def __rsub__(self, other: float) -> "Value":
        return _wrap(other) + (-self)

    def backward(self) -> None:
        """Populate .grad on every ancestor via reverse topological order."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def _wrap(value: "Value | float") -> "Value":
    return value if isinstance(value, Value) else Value(float(value))


class Neuron:
    def __init__(self, n_in: int, rng: random.Random):
        self.w = [Value(rng.uniform(-1.0, 1.0)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, xs: list[Value]) -> Value:
        total = self.b
        for wi, xi in zip(self.w, xs):
            total = total + wi * xi
        return total

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Layer:
    def __init__(self, n_in: int, n_out: int, rng: random.Random, nonlin: bool = True):
        self.neurons = [Neuron(n_in, rng) for _ in range(n_out)]
        self.nonlin = nonlin

    def __call__(self, xs: list[Value]) -> list[Value]:
        raw = [n(xs) for n in self.neurons]
        return [v.tanh() for v in raw] if self.nonlin else raw

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Fully connected net; every layer uses tanh except the linear output."""

    def __init__(self, sizes: list[int], seed: int = 42):
        rng = random.Random(seed)
        self.layers = [
            Layer(sizes[i], sizes[i + 1], rng, nonlin=i < len(sizes) - 2)
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: float | list[Value]) -> Value:
        xs = [Value(float(x))] if not isinstance(x, list) else x
        for layer in self.layers:
            xs = layer(xs)
        return xs[0]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


def fit(
    mlp: MLP,
    xs: np.ndarray,
    ys: np.ndarray,
    epochs: int,
    lr: float,
) -> list[float]:
    """Full-batch gradient descent on mean squared error. Returns the loss history."""
    history: list[float] = []
    for _ in range(epochs):
        loss = Value(0.0)
        for xi, yi in zip(xs, ys):
            pred = mlp(float(xi))
            loss = loss + (pred - float(yi)) ** 2
        loss = loss / float(len(xs))
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= lr * p.grad
        history.append(loss.data)
    return history
