"""
Property tests and unit tests for Beta-Binomial inference.

Tests cover:
- Mathematical properties (e.g., probabilities sum to 1)
- Monotonicity and consistency checks
- Edge cases and error handling
- Reproducibility
"""

import pytest
import numpy as np
from src.beta_binomial import (
    BetaBinomialModel,
    compute_p_best,
    compute_expected_regret,
    log_marginal_likelihood,
)


class TestBetaBinomialModel:
    """Test suite for BetaBinomialModel class."""

    def test_initialization_valid(self):
        """Test valid model initialization."""
        model = BetaBinomialModel(alpha=2.0, beta=3.0)
        assert model.alpha == 2.0
        assert model.beta == 3.0
        assert model.n_samples == 10000
        assert model.seed == 42

    def test_initialization_invalid_params(self):
        """Test that negative or zero parameters raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            BetaBinomialModel(alpha=0.0, beta=1.0)

        with pytest.raises(ValueError, match="must be positive"):
            BetaBinomialModel(alpha=1.0, beta=-1.0)

    def test_update_valid(self):
        """Test Bayesian update with valid data."""
        model = BetaBinomialModel(alpha=1.0, beta=1.0)
        posterior = model.update(successes=10, failures=5)

        assert posterior.alpha == 11.0  # 1 + 10
        assert posterior.beta == 6.0    # 1 + 5

    def test_update_invalid_data(self):
        """Test update with negative counts raises ValueError."""
        model = BetaBinomialModel(alpha=1.0, beta=1.0)

        with pytest.raises(ValueError, match="non-negative"):
            model.update(successes=-1, failures=5)

        with pytest.raises(ValueError, match="non-negative"):
            model.update(successes=10, failures=-2)

    def test_posterior_mean_uniform_prior(self):
        """Test posterior mean with uniform prior Beta(1,1)."""
        model = BetaBinomialModel(alpha=1.0, beta=1.0)
        posterior = model.update(successes=10, failures=5)

        # E[p] = alpha / (alpha + beta) = 11 / 17
        expected_mean = 11.0 / 17.0
        assert np.isclose(posterior.posterior_mean(), expected_mean)

    def test_posterior_variance(self):
        """Test posterior variance calculation."""
        model = BetaBinomialModel(alpha=10.0, beta=10.0)
        variance = model.posterior_variance()

        # Var[p] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        expected_var = (10 * 10) / ((20 ** 2) * 21)
        assert np.isclose(variance, expected_var)

    def test_credible_interval_bounds(self):
        """Test credible interval returns valid probability bounds."""
        model = BetaBinomialModel(alpha=50.0, beta=50.0)
        lower, upper = model.credible_interval(confidence=0.95)

        assert 0.0 <= lower <= upper <= 1.0
        assert lower < model.posterior_mean() < upper

    def test_credible_interval_invalid_confidence(self):
        """Test invalid confidence levels raise ValueError."""
        model = BetaBinomialModel(alpha=10.0, beta=10.0)

        with pytest.raises(ValueError, match="Confidence must be in"):
            model.credible_interval(confidence=0.0)

        with pytest.raises(ValueError, match="Confidence must be in"):
            model.credible_interval(confidence=1.5)

    def test_sample_shape(self):
        """Test posterior sampling returns correct shape."""
        model = BetaBinomialModel(alpha=10.0, beta=5.0, n_samples=1000)
        samples = model.sample()

        assert samples.shape == (1000,)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_reproducibility_with_seed(self):
        """Property: Same seed produces identical samples."""
        model1 = BetaBinomialModel(alpha=10.0, beta=5.0, seed=123)
        model2 = BetaBinomialModel(alpha=10.0, beta=5.0, seed=123)

        samples1 = model1.sample(n_samples=100)
        samples2 = model2.sample(n_samples=100)

        np.testing.assert_array_equal(samples1, samples2)

    def test_different_seeds_different_samples(self):
        """Test different seeds produce different samples."""
        model1 = BetaBinomialModel(alpha=10.0, beta=5.0, seed=123)
        model2 = BetaBinomialModel(alpha=10.0, beta=5.0, seed=456)

        samples1 = model1.sample(n_samples=100)
        samples2 = model2.sample(n_samples=100)

        # Extremely unlikely to be identical
        assert not np.allclose(samples1, samples2)

    def test_to_dict_structure(self):
        """Test model serialization to dictionary."""
        model = BetaBinomialModel(alpha=10.0, beta=5.0, n_samples=5000, seed=99)
        data = model.to_dict()

        assert data["alpha"] == 10.0
        assert data["beta"] == 5.0
        assert data["n_samples"] == 5000
        assert data["seed"] == 99
        assert "posterior_mean" in data
        assert "posterior_variance" in data


class TestPBest:
    """Test suite for P(best) calculation."""

    def test_p_best_sums_to_one(self):
        """Property: P(best) probabilities sum to 1."""
        models = [
            BetaBinomialModel(alpha=10, beta=5, seed=42).update(10, 5),
            BetaBinomialModel(alpha=10, beta=5, seed=42).update(12, 8),
            BetaBinomialModel(alpha=10, beta=5, seed=42).update(8, 12),
        ]

        p_best = compute_p_best(models)

        assert np.isclose(np.sum(p_best), 1.0, atol=1e-6)

    def test_p_best_all_positive(self):
        """Property: All P(best) values are non-negative."""
        models = [
            BetaBinomialModel(alpha=10, beta=10, seed=42).update(5, 5),
            BetaBinomialModel(alpha=10, beta=10, seed=42).update(6, 4),
        ]

        p_best = compute_p_best(models)

        assert np.all(p_best >= 0)
        assert np.all(p_best <= 1)

    def test_p_best_clear_winner(self):
        """Test P(best) identifies clear winner."""
        # Variant with much higher success rate should have high P(best)
        models = [
            BetaBinomialModel(alpha=1, beta=1, seed=42).update(10, 90),   # ~10% success
            BetaBinomialModel(alpha=1, beta=1, seed=42).update(90, 10),   # ~90% success
        ]

        p_best = compute_p_best(models)

        assert p_best[1] > 0.99  # Second variant should be clearly best

    def test_p_best_empty_models(self):
        """Test P(best) with empty models list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_p_best([])

    def test_p_best_mismatched_samples(self):
        """Test P(best) with different n_samples raises error."""
        models = [
            BetaBinomialModel(alpha=10, beta=5, n_samples=1000),
            BetaBinomialModel(alpha=10, beta=5, n_samples=2000),
        ]

        with pytest.raises(ValueError, match="same n_samples"):
            compute_p_best(models)


class TestExpectedRegret:
    """Test suite for expected regret calculation."""

    def test_regret_all_non_negative(self):
        """Property: Expected regret is always non-negative."""
        models = [
            BetaBinomialModel(alpha=10, beta=5, seed=42).update(10, 5),
            BetaBinomialModel(alpha=10, beta=5, seed=42).update(12, 3),
        ]

        regret = compute_expected_regret(models)

        assert np.all(regret >= 0)

    def test_regret_best_variant_lowest(self):
        """Property: Best variant has lowest expected regret."""
        models = [
            BetaBinomialModel(alpha=1, beta=1, seed=42).update(20, 80),
            BetaBinomialModel(alpha=1, beta=1, seed=42).update(80, 20),
        ]

        regret = compute_expected_regret(models)
        p_best = compute_p_best(models)

        best_idx = np.argmax(p_best)
        assert regret[best_idx] == np.min(regret)

    def test_regret_empty_models(self):
        """Test expected regret with empty models list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_expected_regret([])

    def test_regret_monotonicity(self):
        """Property: Higher conversion rate leads to lower expected regret."""
        # Create variants with clearly different success rates
        models = [
            BetaBinomialModel(alpha=1, beta=1, seed=42).update(10, 90),  # Low success
            BetaBinomialModel(alpha=1, beta=1, seed=42).update(50, 50),  # Medium success
            BetaBinomialModel(alpha=1, beta=1, seed=42).update(90, 10),  # High success
        ]

        regret = compute_expected_regret(models)

        # Regret should decrease as success rate increases
        assert regret[0] > regret[1] > regret[2]


class TestLogMarginalLikelihood:
    """Test suite for log marginal likelihood."""

    def test_log_marginal_likelihood_finite(self):
        """Test log marginal likelihood returns finite value."""
        log_ml = log_marginal_likelihood(
            successes=10,
            failures=5,
            alpha_prior=1.0,
            beta_prior=1.0
        )

        assert np.isfinite(log_ml)

    def test_log_marginal_likelihood_depends_on_data(self):
        """Test log marginal likelihood varies with different data."""
        # Different data should give different log marginal likelihoods
        log_ml_1 = log_marginal_likelihood(
            successes=5,
            failures=5,
            alpha_prior=1.0,
            beta_prior=1.0
        )

        log_ml_2 = log_marginal_likelihood(
            successes=10,
            failures=5,
            alpha_prior=1.0,
            beta_prior=1.0
        )

        # Different data should yield different evidence
        assert log_ml_1 != log_ml_2
        assert np.isfinite(log_ml_1) and np.isfinite(log_ml_2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_observations(self):
        """Test model with zero observations equals prior."""
        model = BetaBinomialModel(alpha=2.0, beta=3.0)
        posterior = model.update(successes=0, failures=0)

        assert posterior.alpha == 2.0
        assert posterior.beta == 3.0

    def test_all_successes(self):
        """Test model with only successes."""
        model = BetaBinomialModel(alpha=1.0, beta=1.0)
        posterior = model.update(successes=100, failures=0)

        assert posterior.alpha == 101.0
        assert posterior.beta == 1.0
        assert posterior.posterior_mean() > 0.99  # Should be very high

    def test_all_failures(self):
        """Test model with only failures."""
        model = BetaBinomialModel(alpha=1.0, beta=1.0)
        posterior = model.update(successes=0, failures=100)

        assert posterior.alpha == 1.0
        assert posterior.beta == 101.0
        assert posterior.posterior_mean() < 0.01  # Should be very low

    def test_informative_prior_influence(self):
        """Test that informative prior affects posterior."""
        # Weak prior (uniform)
        weak_prior = BetaBinomialModel(alpha=1.0, beta=1.0)
        weak_posterior = weak_prior.update(successes=5, failures=5)

        # Strong prior (believes high success rate)
        strong_prior = BetaBinomialModel(alpha=100.0, beta=10.0)
        strong_posterior = strong_prior.update(successes=5, failures=5)

        # Strong prior should pull posterior mean toward its belief
        assert strong_posterior.posterior_mean() > weak_posterior.posterior_mean()
