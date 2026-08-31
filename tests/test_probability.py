import math

import numpy as np
import pytest

from calccode import probability


def test_uniform_samples_are_seeded_and_in_range():
    a = probability.uniform_samples(1000, seed=3, a=2.0, b=5.0)
    b = probability.uniform_samples(1000, seed=3, a=2.0, b=5.0)
    assert np.array_equal(a, b)
    assert float(np.min(a)) >= 2.0
    assert float(np.max(a)) < 5.0
    assert abs(probability.sample_mean(a) - 3.5) < 0.1


def test_exponential_samples_match_mean_and_std():
    xs = probability.exponential_samples(50000, lam=2.0, seed=11)
    assert abs(probability.sample_mean(xs) - 0.5) < 0.02
    assert abs(probability.sample_std(xs) - 0.5) < 0.02


def test_normal_samples_have_right_mean_and_std():
    xs = probability.normal_samples(50000, seed=7, mu=3.0, sigma=2.0)
    assert abs(probability.sample_mean(xs) - 3.0) < 0.04
    assert abs(probability.sample_std(xs) - 2.0) < 0.04


def test_normal_samples_deterministic_per_seed():
    a = probability.normal_samples(100, seed=9)
    b = probability.normal_samples(100, seed=9)
    c = probability.normal_samples(100, seed=10)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_normal_cdf_known_values():
    assert abs(probability.normal_cdf(0.0) - 0.5) < 1e-8
    assert abs(probability.normal_cdf(1.0) - 0.8413) < 1e-4
    assert abs(probability.normal_cdf(-1.0) - 0.1587) < 1e-4
    assert abs(probability.normal_cdf(2.0) - 0.9772) < 1e-4


def test_normal_cdf_symmetry():
    assert abs(probability.normal_cdf(1.5) + probability.normal_cdf(-1.5) - 1.0) < 1e-10


def test_normal_pdf_integrates_to_one():
    # Simpson on the density itself over a wide range.
    from calccode.integrals import simpson

    area = simpson(lambda x: probability.normal_pdf(x, 1.0, 2.0), -15.0, 17.0, 4000)
    assert abs(area - 1.0) < 1e-6


def test_uniform_pdf_and_cdf():
    assert probability.uniform_pdf(0.5, 0.0, 2.0) == 0.5
    assert probability.uniform_pdf(3.0, 0.0, 2.0) == 0.0
    assert probability.uniform_cdf(1.0, 0.0, 2.0) == 0.5
    assert probability.uniform_cdf(-1.0) == 0.0
    assert probability.uniform_cdf(2.0) == 1.0


def test_exponential_pdf_and_cdf_agree():
    lam = 1.5
    for x in (0.1, 0.7, 2.3):
        # CDF is the running integral of the PDF; check a numeric derivative.
        h = 1e-6
        slope = (probability.exponential_cdf(x + h, lam) - probability.exponential_cdf(x - h, lam)) / (2 * h)
        assert abs(slope - probability.exponential_pdf(x, lam)) < 1e-5


def test_exponential_mean_and_variance_by_integration():
    lam = 2.0
    pdf = lambda x: probability.exponential_pdf(x, lam)
    mu = probability.expected_value(pdf, 0.0, 25.0)
    var = probability.distribution_variance(pdf, 0.0, 25.0)
    assert abs(mu - 1.0 / lam) < 1e-6
    assert abs(var - 1.0 / lam**2) < 1e-5


def test_normal_mean_and_variance_by_integration():
    pdf = lambda x: probability.normal_pdf(x, 1.5, 0.75)
    mu = probability.expected_value(pdf, -8.0, 11.0)
    var = probability.distribution_variance(pdf, -8.0, 11.0)
    assert abs(mu - 1.5) < 1e-6
    assert abs(var - 0.75**2) < 1e-5


def test_variance_needs_two_samples():
    with pytest.raises(ValueError):
        probability.sample_variance(np.array([1.0]))


def test_median_odd_and_even():
    assert probability.median(np.array([5.0, 1.0, 3.0])) == 3.0
    assert probability.median(np.array([4.0, 1.0, 3.0, 2.0])) == 2.5


def test_percentile_exact_on_sorted_list():
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert probability.percentile(xs, 0) == 1.0
    assert probability.percentile(xs, 25) == 2.0
    assert probability.percentile(xs, 50) == 3.0
    assert probability.percentile(xs, 75) == 4.0
    assert probability.percentile(xs, 100) == 5.0


def test_percentile_interpolates_between_values():
    xs = np.array([0.0, 10.0])
    assert probability.percentile(xs, 50) == 5.0
    assert probability.percentile(xs, 10) == 1.0


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        probability.percentile(np.array([1.0, 2.0]), 101.0)


def test_clt_means_have_predicted_spread():
    lam = 1.0
    n = 30
    data = probability.clt_demo(lam=lam, sample_size=n, n_means=3000, seed=21)
    assert abs(data["mu"] - 1.0 / lam) < 1e-12
    assert abs(data["sigma"] - 1.0 / (lam * math.sqrt(n))) < 1e-12
    # Theory: mean of means is mu, std of means is sigma / ... exactly sigma.
    assert abs(probability.sample_mean(data["means"]) - data["mu"]) < 0.02
    assert abs(probability.sample_std(data["means"]) - data["sigma"]) < 0.02 * data["sigma"] * 10


def test_clt_means_look_normal_against_curve():
    data = probability.clt_demo(lam=1.0, sample_size=30, n_means=3000, seed=5)
    means = data["means"]
    # One-sigma mass of a normal is about 0.6827.
    lo = data["mu"] - data["sigma"]
    hi = data["mu"] + data["sigma"]
    frac = float(np.mean((means >= lo) & (means <= hi)))
    assert abs(frac - 0.6827) < 0.04
