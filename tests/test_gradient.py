import numpy as np

from calccode.gradient import gradient_descent, gradient_descent_1d, numerical_gradient


def bowl(v):
    # Minimum value 0 at (1, -2). Curvatures 2 and 6.
    return (v[0] - 1.0) ** 2 + 3.0 * (v[1] + 2.0) ** 2


def test_numerical_gradient_on_bowl():
    g = numerical_gradient(bowl, np.array([3.0, 4.0]))
    assert np.allclose(g, [4.0, 36.0], atol=1e-4)


def test_gradient_descent_converges_on_2d_bowl():
    path = gradient_descent(bowl, np.array([5.0, 5.0]), lr=0.1, n_iter=200)
    assert np.linalg.norm(path[-1] - np.array([1.0, -2.0])) < 1e-3
    assert bowl(path[-1]) < 1e-6


def test_gradient_descent_1d_quadratic():
    xs = gradient_descent_1d(lambda x: (x - 3.0) ** 2, x0=-4.0, lr=0.1, n_iter=100)
    assert abs(xs[-1] - 3.0) < 1e-3


def test_loss_decreases_monotonically_with_good_lr():
    path = gradient_descent(bowl, np.array([5.0, 5.0]), lr=0.1, n_iter=50)
    losses = np.array([bowl(p) for p in path])
    assert np.all(np.diff(losses) < 0.0)


def test_small_lr_converges_slower_than_good_lr():
    start = np.array([5.0, 5.0])
    slow = gradient_descent(bowl, start, lr=0.01, n_iter=100)
    good = gradient_descent(bowl, start, lr=0.1, n_iter=100)
    assert bowl(slow[-1]) > 100.0 * bowl(good[-1])


def test_oversized_lr_diverges():
    # 2 / max curvature = 1/3, so 0.5 is unstable on this bowl.
    path = gradient_descent(bowl, np.array([1.5, -1.0]), lr=0.5, n_iter=50)
    assert bowl(path[-1]) > 1e6 * bowl(path[0])


def test_history_shape_includes_starting_point():
    path = gradient_descent(bowl, np.array([0.0, 0.0]), lr=0.1, n_iter=10)
    assert path.shape == (11, 2)
