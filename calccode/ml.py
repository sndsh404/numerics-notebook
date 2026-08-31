"""Classical machine learning, from scratch.

Perceptron, k-nearest neighbors, and Gaussian naive Bayes, plus the data
utilities and metrics they need. Shuffles come from the xorshift32 RNG in
montecarlo.py, so every split is reproducible from a seed. Distances,
votes, means, variances, and log-likelihoods are plain loops. The ridge
closed form lives in regression.py next to OLS, since it is the same
normal equations with lambda * I added to the diagonal.
"""

from __future__ import annotations

import math

import numpy as np

from calccode import linalg, regression
from calccode.montecarlo import Xorshift32
from calccode.probability import normal_samples


def _as_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y disagree on the number of samples")
    return X, y


# Data utilities


def _shuffle_indices(n: int, seed: int) -> list[int]:
    """Fisher-Yates shuffle driven by the hand-written xorshift32."""
    rng = Xorshift32(seed)
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.next_uint() % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def train_test_split(
    X: np.ndarray, y: np.ndarray, ratio: float = 0.25, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffled split into (X_train, X_test, y_train, y_test).

    ratio is the test fraction. The shuffle is Fisher-Yates on the
    xorshift32 stream, so the same seed gives the same split every time.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must lie in (0, 1)")
    X, y = _as_xy(X, y)
    n = X.shape[0]
    n_test = max(1, int(round(n * ratio)))
    idx = _shuffle_indices(n, seed)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def k_fold_indices(n: int, k: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """k (train, test) index pairs that partition 0..n-1 exactly.

    After a seeded shuffle the indices are cut into k contiguous folds;
    the first n % k folds get one extra sample. Fold i is the test set,
    everything else is the train set.
    """
    if k < 2 or k > n:
        raise ValueError("k must lie in [2, n]")
    idx = _shuffle_indices(n, seed)
    base, extra = divmod(n, k)
    folds: list[list[int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        folds.append(idx[start : start + size])
        start += size
    out = []
    for i in range(k):
        test = np.array(sorted(folds[i]), dtype=int)
        train = np.array(sorted(j for f, fold in enumerate(folds) if f != i for j in fold), dtype=int)
        out.append((train, test))
    return out


def make_blobs(
    centers: list[tuple[float, ...]], spread: float, n_per: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian blobs, one per center, drawn from the xorshift32 stream.

    Each blob gets its own seed offset so the per-class sample counts can
    change without shifting the other blobs. Labels are 0, 1, ... in the
    order the centers are given.
    """
    dim = len(centers[0])
    X = np.zeros((n_per * len(centers), dim))
    y = np.zeros(n_per * len(centers), dtype=int)
    for label, center in enumerate(centers):
        draws = normal_samples(n_per * dim, seed=seed + 1000 * label + 1)
        for j in range(dim):
            X[label * n_per : (label + 1) * n_per, j] = center[j] + spread * draws[j::dim]
        y[label * n_per : (label + 1) * n_per] = label
    return X, y


# Perceptron


class Perceptron:
    """Classic online perceptron on labels {0, 1}.

    One pass over the data per epoch; every misclassified point updates
    w += lr * y_pm * x with y_pm in {-1, +1}. Converged means a full
    epoch with zero mistakes. On non-separable data that never happens,
    so max_epochs is the only exit and converged stays False.
    """

    def __init__(self, lr: float = 1.0, max_epochs: int = 100):
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        self.lr = lr
        self.max_epochs = max_epochs
        self.w_: np.ndarray | None = None
        self.converged = False
        self.epochs_used = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X, y = _as_xy(X, y)
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("perceptron expects labels 0 and 1")
        Xa = np.column_stack([np.ones(X.shape[0]), X])
        y_pm = 2.0 * y.astype(float) - 1.0
        w = np.zeros(Xa.shape[1])
        self.converged = False
        self.epochs_used = self.max_epochs
        for epoch in range(1, self.max_epochs + 1):
            mistakes = 0
            for i in range(Xa.shape[0]):
                score = 0.0
                for j in range(w.size):
                    score += Xa[i, j] * w[j]
                if y_pm[i] * score <= 0.0:
                    w = w + self.lr * y_pm[i] * Xa[i]
                    mistakes += 1
            if mistakes == 0:
                self.converged = True
                self.epochs_used = epoch
                break
        self.w_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise ValueError("call fit first")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xa = np.column_stack([np.ones(X.shape[0]), X])
        scores = linalg.matmul(Xa, self.w_.reshape(-1, 1)).ravel()
        return (scores >= 0.0).astype(int)


# k-nearest neighbors


class KNNClassifier:
    """Brute-force k-NN with Euclidean distance.

    Every prediction loops over the whole training set. Votes are label
    counts among the k nearest; ties go to the lowest label, so the
    result never depends on dict ordering.
    """

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_, self.y_ = _as_xy(X, y)
        if self.k > self.X_.shape[0]:
            raise ValueError("k cannot exceed the number of training samples")
        return self

    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        for j in range(a.size):
            d = a[j] - b[j]
            total += d * d
        return math.sqrt(total)

    def predict_one(self, x: np.ndarray) -> int:
        if self.X_ is None or self.y_ is None:
            raise ValueError("call fit first")
        dists = sorted(
            (self._euclidean(x, self.X_[i]), int(self.y_[i])) for i in range(self.X_.shape[0])
        )
        counts: dict[int, int] = {}
        for _, label in dists[: self.k]:
            counts[label] = counts.get(label, 0) + 1
        return min(counts, key=lambda label: (-counts[label], label))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.array([self.predict_one(X[i]) for i in range(X.shape[0])], dtype=int)


# Gaussian naive Bayes


class GaussianNaiveBayes:
    """Per-class Gaussians on each feature, scored by log-likelihood.

    fit estimates mean and variance per (class, feature) with plain
    loops; predict sums log priors and Gaussian log densities and takes
    the argmax. The independence assumption is the "naive" part: features
    are treated as uncorrelated inside a class, which is wrong in general
    and still works surprisingly often.
    """

    def __init__(self, var_floor: float = 1e-9):
        self.var_floor = var_floor
        self.classes_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.log_priors_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X, y = _as_xy(X, y)
        self.classes_ = np.array(sorted(set(int(v) for v in y)), dtype=int)
        n_classes = self.classes_.size
        n_features = X.shape[1]
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.log_priors_ = np.zeros(n_classes)
        n = X.shape[0]
        for c, label in enumerate(self.classes_):
            rows = X[y == label]
            m = rows.shape[0]
            self.log_priors_[c] = math.log(m / n)
            for j in range(n_features):
                mu = 0.0
                for i in range(m):
                    mu += rows[i, j]
                mu /= m
                var = 0.0
                for i in range(m):
                    var += (rows[i, j] - mu) ** 2
                var /= m
                self.means_[c, j] = mu
                self.vars_[c, j] = max(var, self.var_floor)
        return self

    def _log_likelihood(self, x: np.ndarray, c: int) -> float:
        total = float(self.log_priors_[c])
        for j in range(x.size):
            var = self.vars_[c, j]
            total += -0.5 * math.log(2.0 * math.pi * var) - (x[j] - self.means_[c, j]) ** 2 / (2.0 * var)
        return total

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("call fit first")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        out = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            scores = [self._log_likelihood(X[i], c) for c in range(self.classes_.size)]
            out[i] = int(self.classes_[int(np.argmax(scores))])
        return out


# Ridge helpers (the closed form itself is in regression.py)


def _linear_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    return linalg.matmul(Xa, w.reshape(-1, 1)).ravel()


def ridge_lambda_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Test MSE of ridge fits across a sweep of lambda values."""
    out = np.empty(len(lambdas))
    for i, lam in enumerate(lambdas):
        w = regression.ridge_closed_form(X_train, y_train, lam)
        out[i] = mean_squared_error(y_test, _linear_predict(X_test, w))
    return out


# Metrics


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that match."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    correct = sum(1 for i in range(y_true.size) if y_true[i] == y_pred[i])
    return correct / y_true.size


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None
) -> np.ndarray:
    """Counts with true labels on rows, predicted labels on columns."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = np.array(sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred)))
    index = {int(label): i for i, label in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for i in range(y_true.size):
        M[index[int(y_true[i])], index[int(y_pred[i])]] += 1
    return M


def precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1
) -> tuple[float, float]:
    """Binary precision and recall for the given positive label.

    A zero denominator returns 0.0 rather than raising: a model that
    never predicts positive has no precision to speak of.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = fp = fn = 0
    for i in range(y_true.size):
        if y_pred[i] == positive and y_true[i] == positive:
            tp += 1
        elif y_pred[i] == positive:
            fp += 1
        elif y_true[i] == positive:
            fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    total = 0.0
    for i in range(y_true.size):
        total += (y_true[i] - y_pred[i]) ** 2
    return total / y_true.size
