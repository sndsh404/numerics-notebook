import math

import numpy as np

from calccode import ml, regression


def make_blobs(seed=7, n_per=60, spread=0.6):
    centers = [(-2.0, -2.0), (2.0, 2.0)]
    return ml.make_blobs(centers, spread=spread, n_per=n_per, seed=seed)


# Data utilities


def test_train_test_split_sizes_and_disjointness():
    X, y = make_blobs()
    X_train, X_test, y_train, y_test = ml.train_test_split(X, y, ratio=0.25, seed=3)
    assert X_train.shape[0] == 90
    assert X_test.shape[0] == 30
    assert y_train.shape[0] == 90
    assert y_test.shape[0] == 30
    # Reassembling the split recovers every row exactly once.
    counts = np.zeros(X.shape[0], dtype=int)
    for row in np.vstack([X_train, X_test]):
        hits = np.where(np.all(np.isclose(X, row), axis=1))[0]
        counts[hits] += 1
    assert np.all(counts == 1)


def test_train_test_split_is_seeded():
    X, y = make_blobs()
    a = ml.train_test_split(X, y, ratio=0.3, seed=11)
    b = ml.train_test_split(X, y, ratio=0.3, seed=11)
    c = ml.train_test_split(X, y, ratio=0.3, seed=12)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_k_fold_indices_partition_exactly():
    n, k = 103, 5
    folds = ml.k_fold_indices(n, k, seed=5)
    assert len(folds) == k
    tests = [set(test.tolist()) for _, test in folds]
    # No overlaps between test folds.
    for i in range(k):
        for j in range(i + 1, k):
            assert tests[i] & tests[j] == set()
    # Full coverage: every index appears in exactly one test fold.
    assert set().union(*tests) == set(range(n))
    for train, test in folds:
        assert set(train.tolist()) & set(test.tolist()) == set()
        assert train.size + test.size == n


# Perceptron


def test_perceptron_learns_separable_data():
    # Two well separated 2D blobs: a line exists, so the perceptron must find one.
    X, y = make_blobs(spread=0.4)
    clf = ml.Perceptron(lr=1.0, max_epochs=100).fit(X, y)
    assert clf.converged
    assert ml.accuracy(y, clf.predict(X)) == 1.0


def test_perceptron_hits_epoch_cap_on_xor():
    # XOR is not linearly separable, so no weight vector separates it.
    # The perceptron cycles forever; max_epochs is the only exit.
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    clf = ml.Perceptron(lr=1.0, max_epochs=50).fit(X, y)
    assert not clf.converged
    assert clf.epochs_used == 50
    assert ml.accuracy(y, clf.predict(X)) <= 0.75


# k-nearest neighbors


def test_knn_k1_memorizes_training_set():
    X, y = make_blobs()
    clf = ml.KNNClassifier(k=1).fit(X, y)
    assert ml.accuracy(y, clf.predict(X)) == 1.0


def test_knn_k3_on_two_blobs():
    X, y = make_blobs(seed=21)
    X_train, X_test, y_train, y_test = ml.train_test_split(X, y, ratio=0.3, seed=9)
    clf = ml.KNNClassifier(k=3).fit(X_train, y_train)
    acc = ml.accuracy(y_test, clf.predict(X_test))
    assert acc > 0.9


def test_knn_tie_break_picks_lowest_label():
    # k=2 with one neighbor of each class: the vote ties 1-1 and the
    # deterministic rule gives it to the lower label, 0.
    X = np.array([[-1.0], [1.0]])
    y = np.array([0, 1])
    clf = ml.KNNClassifier(k=2).fit(X, y)
    assert clf.predict(np.array([[0.0]]))[0] == 0


# Gaussian naive Bayes


def test_naive_bayes_beats_chance_on_blobs():
    X, y = make_blobs(seed=13, spread=0.8)
    X_train, X_test, y_train, y_test = ml.train_test_split(X, y, ratio=0.3, seed=4)
    clf = ml.GaussianNaiveBayes().fit(X_train, y_train)
    acc = ml.accuracy(y_test, clf.predict(X_test))
    assert acc > 0.85  # chance is 0.5 on balanced two-class data


def test_confusion_matrix_rows_sum_to_class_counts():
    X, y = make_blobs(seed=13, spread=0.8)
    clf = ml.GaussianNaiveBayes().fit(X, y)
    M = ml.confusion_matrix(y, clf.predict(X))
    assert M.shape == (2, 2)
    assert np.all(M.sum(axis=1) == np.array([60, 60]))
    assert M.sum() == y.size


def test_precision_recall_perfect_and_degenerate():
    y_true = np.array([0, 0, 1, 1, 1])
    perfect = np.array([0, 0, 1, 1, 1])
    assert ml.precision_recall(y_true, perfect) == (1.0, 1.0)
    never_positive = np.zeros(5, dtype=int)
    precision, recall = ml.precision_recall(y_true, never_positive)
    assert precision == 0.0
    assert recall == 0.0
    half = np.array([0, 0, 1, 1, 0])  # misses one positive
    precision, recall = ml.precision_recall(y_true, half)
    assert precision == 1.0
    assert math.isclose(recall, 2 / 3)


# Ridge regression


def make_collinear_data(n=60, seed=17):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    x2 = x1 + 1e-4 * rng.normal(size=n)  # nearly a copy of x1
    X = np.column_stack([x1, x2])
    y = 1.0 + 2.0 * x1 - x2 + 0.05 * rng.normal(size=n)
    return X, y


def test_ridge_lambda_zero_matches_ols():
    X, y = make_collinear_data()
    w_ols = regression.ols_closed_form(X, y)
    w_ridge = regression.ridge_closed_form(X, y, lam=0.0)
    assert np.allclose(w_ridge, w_ols, atol=1e-6)


def test_ridge_larger_lambda_shrinks_norms():
    X, y = make_collinear_data()
    w_small = regression.ridge_closed_form(X, y, lam=1e-6)
    w_large = regression.ridge_closed_form(X, y, lam=10.0)
    norm_small = math.sqrt(sum(v * v for v in w_small))
    norm_large = math.sqrt(sum(v * v for v in w_large))
    assert norm_large < norm_small


def test_ridge_generalizes_better_on_collinear_data():
    # X^T X is nearly singular here, so OLS swings the coefficients around
    # to fit the training noise. Ridge trades a little bias for a much
    # smaller variance and wins on held-out points.
    X, y = make_collinear_data()
    X_train, X_test, y_train, y_test = ml.train_test_split(X, y, ratio=0.3, seed=8)
    w_ols = regression.ols_closed_form(X_train, y_train)
    w_ridge = regression.ridge_closed_form(X_train, y_train, lam=1.0)
    mse_ols = ml.mean_squared_error(y_test, ml._linear_predict(X_test, w_ols))
    mse_ridge = ml.mean_squared_error(y_test, ml._linear_predict(X_test, w_ridge))
    assert mse_ridge < mse_ols


def test_ridge_lambda_curve_shape():
    X, y = make_collinear_data()
    X_train, X_test, y_train, y_test = ml.train_test_split(X, y, ratio=0.3, seed=8)
    lambdas = np.array([0.0, 0.01, 0.1, 1.0, 10.0])
    curve = ml.ridge_lambda_curve(X_train, y_train, X_test, y_test, lambdas)
    assert curve.shape == (5,)
    assert np.all(np.isfinite(curve))
    assert curve[3] < curve[0]


# Metrics


def test_accuracy_and_mse():
    assert ml.accuracy(np.array([1, 0, 1]), np.array([1, 1, 1])) == 2 / 3
    assert ml.mean_squared_error(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == 2.0
