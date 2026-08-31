import math

import numpy as np

from calccode.autograd import MLP, Value, fit
from calccode.derivatives import central_diff


def test_add_mul_gradients():
    x, y = Value(3.0), Value(-2.0)
    z = x * y + x
    z.backward()
    assert z.data == -3.0
    assert x.grad == -1.0  # y + 1
    assert y.grad == 3.0  # x


def test_power_rule_gradient():
    x = Value(4.0)
    y = x**3
    y.backward()
    assert x.grad == 48.0


def test_known_expression_matches_finite_difference():
    def f(t: float) -> float:
        return math.tanh(t) * math.exp(t) + math.log(t * t + 1.0)

    x = Value(0.8)
    out = x.tanh() * x.exp() + (x * x + 1.0).log()
    out.backward()
    numeric = central_diff(f, 0.8, 1e-6)
    assert abs(x.grad - numeric) < 1e-5


def test_chain_rule_through_nested_ops():
    x = Value(1.3)
    y = ((x * 2.0 + 1.0) ** 2).relu()
    y.backward()
    numeric = central_diff(lambda t: max(0.0, (2.0 * t + 1.0) ** 2), 1.3, 1e-6)
    assert abs(x.grad - numeric) < 1e-4


def test_gradients_accumulate_across_uses():
    x = Value(5.0)
    y = x * x
    y.backward()
    assert x.grad == 10.0  # x is used twice, both contributions add up


def test_mlp_forward_shape_and_parameters():
    mlp = MLP([1, 8, 8, 1], seed=1)
    out = mlp(0.5)
    assert isinstance(out, Value)
    # Layer sizes (in+1)*out: 2*8 + 9*8 + 9*1 weights and biases.
    assert len(mlp.parameters()) == 97


def test_mlp_training_reduces_loss_on_sin():
    xs = np.linspace(-math.pi, math.pi, 32)
    ys = np.sin(xs)
    mlp = MLP([1, 10, 10, 1], seed=3)
    history = fit(mlp, xs, ys, epochs=200, lr=0.05)
    assert history[-1] < 0.5 * history[0]
    assert math.isfinite(history[-1])


def _deep_chain(n: int) -> tuple[Value, Value]:
    x = Value(0.5)
    out = x
    for _ in range(n):
        out = out * 0.999 + 0.001
    return x, out


def _branched_chain(n: int) -> tuple[Value, Value]:
    # Every step reads out three times, so gradient accumulation order
    # decides the last bits of x.grad.
    x = Value(0.7)
    out = x
    for _ in range(n):
        out = out * 0.11 + out * 0.13 + out * 0.07
    return x, out


def test_backward_handles_a_deep_chain():
    x, out = _deep_chain(5000)
    out.backward()
    assert math.isfinite(x.grad)


def test_backward_is_bit_reproducible():
    x1, out1 = _branched_chain(200)
    x2, out2 = _branched_chain(200)
    out1.backward()
    out2.backward()
    assert out1.data == out2.data
    assert x1.grad == x2.grad  # bit-identical, not just close
