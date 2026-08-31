
import numpy as np

from calccode import regression


def make_line_data(n=80, slope=2.5, intercept=-1.0, noise=0.3, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=n)
    y = slope * X + intercept + noise * rng.normal(size=n)
    return X, y


def test_closed_form_recovers_known_line():
    X = np.linspace(-3.0, 3.0, 50)
    y = 2.5 * X - 1.0  # no noise
    w = regression.ols_closed_form(X, y)
    assert np.allclose(w, [-1.0, 2.5], atol=1e-9)


def test_closed_form_matches_numpy_reference():
    X, y = make_line_data()
    w = regression.ols_closed_form(X, y)
    ref = np.polyfit(X, y, 1)  # [slope, intercept]
    assert np.allclose(w, [ref[1], ref[0]], atol=1e-8)


def test_gradient_descent_agrees_with_closed_form():
    X, y = make_line_data()
    w_exact = regression.ols_closed_form(X, y)
    w_gd, history = regression.ols_gradient_descent(X, y, lr=0.5, n_iter=3000)
    assert np.allclose(w_gd, w_exact, atol=1e-3)
    # Loss floors at the noise variance, so it cannot shrink without bound.
    assert history[-1] < 0.1 * history[0]


def test_gradient_descent_loss_decreases_monotonically():
    X, y = make_line_data()
    _, history = regression.ols_gradient_descent(X, y, lr=0.3, n_iter=500)
    # Tiny increases at the 1e-17 level are float noise near convergence.
    assert np.all(np.diff(history) <= 1e-12)


def test_multivariate_closed_form():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 2))
    y = 4.0 - 2.0 * X[:, 0] + 0.5 * X[:, 1] + 0.05 * rng.normal(size=120)
    w = regression.ols_closed_form(X, y)
    assert np.allclose(w, [4.0, -2.0, 0.5], atol=0.1)


def test_sigmoid_bounds_and_midpoint():
    z = np.array([-100.0, 0.0, 100.0])
    s = regression.sigmoid(z)
    assert s[0] < 1e-40
    assert s[1] == 0.5
    assert 1.0 - s[2] < 1e-40


def test_logistic_regression_separates_two_clusters():
    rng = np.random.default_rng(11)
    X0 = rng.normal(loc=-2.0, scale=0.7, size=(60, 2))
    X1 = rng.normal(loc=2.0, scale=0.7, size=(60, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(60), np.ones(60)])
    w, history = regression.logistic_gradient_descent(X, y, lr=0.5, n_iter=2000)
    preds = regression.predict_class(X, w)
    accuracy = float(np.mean(preds == y))
    assert accuracy > 0.95
    assert history[-1] < history[0]
    assert history[-1] < 0.2
