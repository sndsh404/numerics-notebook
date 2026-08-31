"""Ordinary least squares and logistic regression, from scratch.

The closed form solves the normal equations with the hand-written
Gaussian elimination from linalg.py. The iterative fit is batch
gradient descent with hand-written gradients of the mean squared error.
Logistic regression fits the cross entropy loss the same way. No
numpy.linalg, no sklearn.
"""

from __future__ import annotations

import math

import numpy as np

from calccode import linalg


def _as_design(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y disagree on the number of samples")
    return X, y


def _augment(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the intercept is just another weight."""
    return np.column_stack([np.ones(X.shape[0]), X])


def ols_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights from the normal equations (X^T X) w = X^T y."""
    X, y = _as_design(X, y)
    Xa = _augment(X)
    XtX = linalg.matmul(linalg.transpose(Xa), Xa)
    Xty = linalg.matmul(linalg.transpose(Xa), y.reshape(-1, 1)).ravel()
    return linalg.solve(XtX, Xty)


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Ridge weights from (X^T X + lam I) w = X^T y on the augmented design.

    The intercept sits in the augmented column, so it gets penalized along
    with the slopes. lam = 0 reduces to the plain normal equations, and
    larger lam pulls every weight toward zero, which is what rescues the
    solve when the columns of X are nearly collinear.
    """
    if lam < 0.0:
        raise ValueError("lam must be nonnegative")
    X, y = _as_design(X, y)
    Xa = _augment(X)
    XtX = linalg.matmul(linalg.transpose(Xa), Xa)
    Xty = linalg.matmul(linalg.transpose(Xa), y.reshape(-1, 1)).ravel()
    p = XtX.shape[0]
    A = XtX + lam * linalg.identity(p)
    return linalg.solve(A, Xty)


def _mse_grad(Xa: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = Xa.shape[0]
    residual = linalg.matmul(Xa, w.reshape(-1, 1)).ravel() - y
    return (2.0 / n) * linalg.matmul(linalg.transpose(Xa), residual.reshape(-1, 1)).ravel()


def ols_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.5,
    n_iter: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch gradient descent on mean squared error. Returns (weights, loss history)."""
    X, y = _as_design(X, y)
    Xa = _augment(X)
    w = np.zeros(Xa.shape[1])
    history = np.empty(n_iter)
    for i in range(n_iter):
        w = w - lr * _mse_grad(Xa, y, w)
        residual = linalg.matmul(Xa, w.reshape(-1, 1)).ravel() - y
        history[i] = float(residual @ residual) / Xa.shape[0]
    return w, history


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def logistic_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.5,
    n_iter: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch gradient descent on cross entropy. Returns (weights, loss history)."""
    X, y = _as_design(X, y)
    Xa = _augment(X)
    n = Xa.shape[0]
    w = np.zeros(Xa.shape[1])
    history = np.empty(n_iter)
    for i in range(n_iter):
        p = sigmoid(linalg.matmul(Xa, w.reshape(-1, 1)).ravel())
        w = w - (lr / n) * linalg.matmul(linalg.transpose(Xa), (p - y).reshape(-1, 1)).ravel()
        eps = 1e-12  # keeps log away from 0 when a prediction saturates
        history[i] = -float(np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
    return w, history


def predict_probability(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return sigmoid(linalg.matmul(_augment(X), w.reshape(-1, 1)).ravel())


def predict_class(X: np.ndarray, w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (predict_probability(X, w) >= threshold).astype(int)
