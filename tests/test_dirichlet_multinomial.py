"""
Property tests and unit tests for Dirichlet-Multinomial inference.

Tests cover:
- Mathematical properties (e.g., probabilities sum to 1)
- Monotonicity and consistency checks
- Edge cases and error handling
- Reproducibility
"""

import pytest
import numpy as np
from src.dirichlet_multinomial import (
    DirichletMultinomialModel,
    compute_p_best_dirichlet,
    compute_expected_regret_dirichlet,
    log_marginal_likelihood_multinomial,
    compute_concentration,
)


class TestDirichletMultinomialModel:
    """Test suite for DirichletMultinomialModel class."""

    def test_initialization_valid(self):
        """Test valid model initialization."""
        alphas = [2.0, 3.0, 1.0]
        model = DirichletMultinomialModel(alphas=alphas)

        np.testing.assert_array_equal(model.alphas, alphas)
        assert model.n_categories == 3
        assert model.n_samples == 10000
        assert model.seed == 42

    def test_initialization_invalid_params(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            DirichletMultinomialModel(alphas=[1.0, 0.0, 2.0])

        with pytest.raises(ValueError, match="must be positive"):
            DirichletMultinomialModel(alphas=[1.0, -1.0, 2.0])

    def test_initialization_empty_alphas(self):
        """Test empty alphas raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DirichletMultinomialModel(alphas=[])

    def test_update_valid(self):
        """Test Bayesian update with valid data."""
        model = DirichletMultinomialModel(alphas=[1.0, 1.0, 1.0])
        posterior = model.update(counts=[10, 5, 8])

        expected_alphas = [11.0, 6.0, 9.0]
        np.testing.assert_array_almost_equal(posterior.alphas, expected_alphas)

    def test_update_mismatched_counts(self):
        """Test update with wrong number of counts raises ValueError."""
        model = DirichletMultinomialModel(alphas=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="must match"):
            model.update(counts=[10, 5])  # Only 2 counts, need 3

    def test_update_negative_counts(self):
        """Test update with negative counts raises ValueError."""
        model = DirichletMultinomialModel(alphas=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="non-negative"):
            model.update(counts=[10, -5, 8])

    def test_posterior_mean_sums_to_one(self):
        """Property: Posterior means sum to 1."""
        model = DirichletMultinomialModel(alphas=[5.0, 3.0, 2.0])
        posterior = model.update(counts=[10, 15, 20])

        means = posterior.posterior_mean()
        assert np.isclose(np.sum(means), 1.0)

    def test_posterior_mean_uniform_prior(self):
        """Test posterior mean with uniform prior."""
        model = DirichletMultinomialModel(alphas=[1.0, 1.0, 1.0])
        posterior = model.update(counts=[10, 5, 15])

        # E[p_i] = alpha_i / sum(alphas) = [11, 6, 16] / 33
        expected_means = np.array([11, 6, 16]) / 33.0
        np.testing.assert_array_almost_equal(posterior.posterior_mean(), expected_means)

    def test_posterior_variance_all_positive(self):
        """Property: All posterior variances are positive."""
        model = DirichletMultinomialModel(alphas=[10.0, 5.0, 8.0])
        variances = model.posterior_variance()

        assert np.all(variances > 0)

    def test_posterior_covariance_symmetric(self):
        """Property: Covariance matrix is symmetric."""
        model = DirichletMultinomialModel(alphas=[10.0, 5.0, 8.0])
        cov = model.posterior_covariance()

        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_posterior_covariance_shape(self):
        """Test covariance matrix has correct shape."""
        model = DirichletMultinomialModel(alphas=[1.0, 1.0, 1.0])
        cov = model.posterior_covariance()

        assert cov.shape == (3, 3)

    def test_credible_interval_valid_category(self):
        """Test credible interval for valid category."""
        model = DirichletMultinomialModel(alphas=[50.0, 30.0, 20.0])
        lower, upper = model.credible_interval(category=0, confidence=0.95)

        assert 0.0 <= lower <= upper <= 1.0

    def test_credible_interval_invalid_category(self):
        """Test invalid category index raises ValueError."""
        model = DirichletMultinomialModel(alphas=[10.0, 5.0, 8.0])

        with pytest.raises(ValueError, match="Category must be in"):
            model.credible_interval(category=5, confidence=0.95)

    def test_credible_interval_invalid_confidence(self):
        """Test invalid confidence raises ValueError."""
        model = DirichletMultinomialModel(alphas=[10.0, 5.0, 8.0])

        with pytest.raises(ValueError, match="Confidence must be in"):
            model.credible_interval(category=0, confidence=1.5)

    def test_sample_shape(self):
        """Test posterior sampling returns correct shape."""
        model = DirichletMultinomialModel(alphas=[10.0, 5.0, 8.0], n_samples=1000)
        samples = model.sample()

        assert samples.shape == (1000, 3)
        # Each sample should sum to 1 (property of Dirichlet)
        assert np.allclose(np.sum(samples, axis=1), 1.0)

    def test_sample_rows_sum_to_one(self):
        """Property: Each Dirichlet sample sums to 1."""
        model = DirichletMultinomialModel(alphas=[5.0, 3.0, 2.0])
        samples = model.sample(n_samples=100)

        row_sums = np.sum(samples, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(100))

    def test_reproducibility_with_seed(self):
        """Property: Same seed produces identical samples."""
        model1 = DirichletMultinomialModel(alphas=[10.0, 5.0, 8.0], seed=123)
        model2 = DirichletMultinomialModel(alphas=[10.0, 5.0, 8.0], seed=123)

        samples1 = model1.sample(n_samples=50)
        samples2 = model2.sample(n_samples=50)

        np.testing.assert_array_equal(samples1, samples2)

    def test_to_dict_structure(self):
        """Test model serialization to dictionary."""
        model = DirichletMultinomialModel(alphas=[10.0, 5.0, 8.0], n_samples=5000, seed=99)
        data = model.to_dict()

        assert data["alphas"] == [10.0, 5.0, 8.0]
        assert data["n_categories"] == 3
        assert data["n_samples"] == 5000
        assert data["seed"] == 99
        assert "posterior_mean" in data
        assert "posterior_variance" in data


class TestPBestDirichlet:
    """Test suite for P(best) calculation with Dirichlet."""

    def test_p_best_sums_to_one(self):
        """Property: P(best) probabilities sum to 1."""
        model = DirichletMultinomialModel(alphas=[10, 5, 8], seed=42)
        posterior = model.update(counts=[10, 15, 20])

        p_best = compute_p_best_dirichlet(posterior)

        assert np.isclose(np.sum(p_best), 1.0, atol=1e-6)

    def test_p_best_all_in_range(self):
        """Property: All P(best) values are in [0, 1]."""
        model = DirichletMultinomialModel(alphas=[5, 5, 5], seed=42)
        posterior = model.update(counts=[10, 12, 8])

        p_best = compute_p_best_dirichlet(posterior)

        assert np.all(p_best >= 0)
        assert np.all(p_best <= 1)

    def test_p_best_clear_winner(self):
        """Test P(best) identifies clear winner."""
        # Category with much higher count should have high P(best)
        model = DirichletMultinomialModel(alphas=[1, 1, 1], seed=42)
        posterior = model.update(counts=[10, 90, 10])

        p_best = compute_p_best_dirichlet(posterior)

        assert p_best[1] > 0.99  # Second category should be clearly best

    def test_p_best_uniform_case(self):
        """Test P(best) with equal evidence for all categories."""
        model = DirichletMultinomialModel(alphas=[1, 1, 1], seed=42)
        posterior = model.update(counts=[50, 50, 50])

        p_best = compute_p_best_dirichlet(posterior)

        # All categories should have roughly equal P(best)
        assert np.allclose(p_best, 1/3, atol=0.05)


class TestExpectedRegretDirichlet:
    """Test suite for expected regret with Dirichlet."""

    def test_regret_all_non_negative(self):
        """Property: Expected regret is always non-negative."""
        model = DirichletMultinomialModel(alphas=[10, 5, 8], seed=42)
        posterior = model.update(counts=[10, 15, 20])

        regret = compute_expected_regret_dirichlet(posterior)

        assert np.all(regret >= 0)

    def test_regret_best_category_lowest(self):
        """Property: Best category has lowest expected regret."""
        model = DirichletMultinomialModel(alphas=[1, 1, 1], seed=42)
        posterior = model.update(counts=[10, 50, 10])

        regret = compute_expected_regret_dirichlet(posterior)
        p_best = compute_p_best_dirichlet(posterior)

        best_idx = np.argmax(p_best)
        assert regret[best_idx] == np.min(regret)

    def test_regret_consistent_with_p_best(self):
        """Property: Lower regret correlates with higher P(best)."""
        model = DirichletMultinomialModel(alphas=[1, 1, 1], seed=42)
        posterior = model.update(counts=[20, 50, 30])

        regret = compute_expected_regret_dirichlet(posterior)
        p_best = compute_p_best_dirichlet(posterior)

        # Category with highest P(best) should have lowest regret
        assert np.argmax(p_best) == np.argmin(regret)


class TestLogMarginalLikelihoodMultinomial:
    """Test suite for log marginal likelihood."""

    def test_log_marginal_likelihood_finite(self):
        """Test log marginal likelihood returns finite value."""
        log_ml = log_marginal_likelihood_multinomial(
            counts=[10, 15, 20],
            alphas_prior=[1.0, 1.0, 1.0]
        )

        assert np.isfinite(log_ml)

    def test_log_marginal_likelihood_mismatched_lengths(self):
        """Test mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            log_marginal_likelihood_multinomial(
                counts=[10, 15, 20],
                alphas_prior=[1.0, 1.0]
            )


class TestConcentration:
    """Test suite for concentration parameter."""

    def test_concentration_sum_of_alphas(self):
        """Test concentration equals sum of alphas."""
        model = DirichletMultinomialModel(alphas=[10.0, 5.0, 8.0])
        concentration = compute_concentration(model)

        assert concentration == 23.0

    def test_concentration_increases_with_data(self):
        """Property: Concentration increases with more data."""
        model = DirichletMultinomialModel(alphas=[1.0, 1.0, 1.0])
        prior_concentration = compute_concentration(model)

        posterior = model.update(counts=[10, 15, 20])
        posterior_concentration = compute_concentration(posterior)

        assert posterior_concentration > prior_concentration
        assert posterior_concentration == prior_concentration + 45  # 10 + 15 + 20


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_observations(self):
        """Test model with zero observations equals prior."""
        model = DirichletMultinomialModel(alphas=[2.0, 3.0, 1.0])
        posterior = model.update(counts=[0, 0, 0])

        np.testing.assert_array_equal(posterior.alphas, [2.0, 3.0, 1.0])

    def test_all_observations_one_category(self):
        """Test model with all observations in one category."""
        model = DirichletMultinomialModel(alphas=[1.0, 1.0, 1.0])
        posterior = model.update(counts=[100, 0, 0])

        # First category should dominate
        means = posterior.posterior_mean()
        assert means[0] > 0.98
        assert means[1] < 0.02
        assert means[2] < 0.02

    def test_two_categories(self):
        """Test model with only two categories (reduces to Beta-Binomial)."""
        model = DirichletMultinomialModel(alphas=[1.0, 1.0])
        posterior = model.update(counts=[10, 5])

        # Should behave like Dirichlet(11, 6), means sum to 1
        means = posterior.posterior_mean()
        np.testing.assert_array_almost_equal(means, [11/17, 6/17])

    def test_many_categories(self):
        """Test model with many categories."""
        n_categories = 10
        model = DirichletMultinomialModel(alphas=np.ones(n_categories))
        posterior = model.update(counts=np.random.randint(0, 50, size=n_categories))

        # Should still satisfy basic properties
        means = posterior.posterior_mean()
        assert np.isclose(np.sum(means), 1.0)
        assert np.all(means >= 0)
        assert np.all(means <= 1)

    def test_informative_prior_influence(self):
        """Test that informative prior affects posterior."""
        # Weak prior (uniform)
        weak_prior = DirichletMultinomialModel(alphas=[1.0, 1.0, 1.0])
        weak_posterior = weak_prior.update(counts=[10, 10, 10])

        # Strong prior (believes first category is best)
        strong_prior = DirichletMultinomialModel(alphas=[100.0, 10.0, 10.0])
        strong_posterior = strong_prior.update(counts=[10, 10, 10])

        # Strong prior should pull first category mean higher
        assert strong_posterior.posterior_mean()[0] > weak_posterior.posterior_mean()[0]
