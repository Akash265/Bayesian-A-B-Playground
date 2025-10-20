"""
Beta-Binomial Bayesian inference for A/B testing.

This module implements conjugate Bayesian updates for binary outcomes using the
Beta-Binomial model, along with decision metrics like P(best) and expected regret.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
from scipy import stats
from scipy.special import betaln


class BetaBinomialModel:
    """
    Beta-Binomial conjugate model for Bayesian A/B testing.

    The Beta distribution is a conjugate prior for the Binomial likelihood,
    making posterior updates analytically tractable.

    Attributes:
        alpha: Beta distribution alpha parameter (prior successes + 1)
        beta: Beta distribution beta parameter (prior failures + 1)
        n_samples: Number of posterior samples for Monte Carlo estimation
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        n_samples: int = 10000,
        seed: int = 42
    ) -> None:
        """
        Initialize Beta prior.

        Args:
            alpha: Prior alpha (default 1.0 for uniform prior)
            beta: Prior beta (default 1.0 for uniform prior)
            n_samples: Number of Monte Carlo samples for P(best) estimation
            seed: Random seed for reproducibility

        Raises:
            ValueError: If alpha or beta are not positive
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Alpha and beta must be positive, got alpha={alpha}, beta={beta}")

        self.alpha = alpha
        self.beta = beta
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def update(self, successes: int, failures: int) -> "BetaBinomialModel":
        """
        Perform Bayesian update with new observations.

        Posterior: Beta(alpha + successes, beta + failures)

        Args:
            successes: Number of successful trials
            failures: Number of failed trials

        Returns:
            New BetaBinomialModel with updated posterior parameters

        Raises:
            ValueError: If successes or failures are negative
        """
        if successes < 0 or failures < 0:
            raise ValueError(
                f"Successes and failures must be non-negative, "
                f"got successes={successes}, failures={failures}"
            )

        return BetaBinomialModel(
            alpha=self.alpha + successes,
            beta=self.beta + failures,
            n_samples=self.n_samples,
            seed=self.seed
        )

    def posterior_mean(self) -> float:
        """
        Compute posterior expectation E[p | data].

        Returns:
            Posterior mean
        """
        return self.alpha / (self.alpha + self.beta)

    def posterior_variance(self) -> float:
        """
        Compute posterior variance Var[p | data].

        Returns:
            Posterior variance
        """
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def credible_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute equal-tailed credible interval.

        Args:
            confidence: Credible level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)

        Raises:
            ValueError: If confidence not in (0, 1)
        """
        if not 0 < confidence < 1:
            raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

        tail_prob = (1 - confidence) / 2
        lower = stats.beta.ppf(tail_prob, self.alpha, self.beta)
        upper = stats.beta.ppf(1 - tail_prob, self.alpha, self.beta)
        return (float(lower), float(upper))

    def sample(self, n_samples: int | None = None) -> np.ndarray:
        """
        Draw samples from the posterior distribution.

        Args:
            n_samples: Number of samples (uses self.n_samples if None)

        Returns:
            Array of posterior samples
        """
        if n_samples is None:
            n_samples = self.n_samples
        return self.rng.beta(self.alpha, self.beta, size=n_samples)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model state to dictionary for serialization.

        Returns:
            Dictionary with model parameters
        """
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "posterior_mean": self.posterior_mean(),
            "posterior_variance": self.posterior_variance(),
            "n_samples": self.n_samples,
            "seed": self.seed
        }


def compute_p_best(models: List[BetaBinomialModel]) -> np.ndarray:
    """
    Compute P(best) for each variant using Monte Carlo sampling.

    P(best_i) = P(p_i > p_j for all j != i)

    This is estimated by:
    1. Drawing samples from each posterior
    2. Counting how often each variant has the highest sample
    3. Normalizing counts to probabilities

    Args:
        models: List of BetaBinomialModel instances

    Returns:
        Array of P(best) probabilities, one per model

    Raises:
        ValueError: If models list is empty or models have different n_samples
    """
    if not models:
        raise ValueError("Models list cannot be empty")

    n_samples = models[0].n_samples
    if not all(m.n_samples == n_samples for m in models):
        raise ValueError("All models must have the same n_samples")

    # Draw samples from all posteriors
    samples = np.array([model.sample() for model in models])  # shape: (n_variants, n_samples)

    # Find winner for each sample
    winners = np.argmax(samples, axis=0)  # shape: (n_samples,)

    # Count wins for each variant
    n_variants = len(models)
    p_best = np.zeros(n_variants)
    for i in range(n_variants):
        p_best[i] = np.mean(winners == i)

    return p_best


def compute_expected_regret(models: List[BetaBinomialModel]) -> np.ndarray:
    """
    Compute expected regret for choosing each variant.

    Expected regret for choosing variant i:
    E[max_j(p_j) - p_i | data]

    This is the expected loss in conversion rate from choosing variant i
    instead of the best variant.

    Args:
        models: List of BetaBinomialModel instances

    Returns:
        Array of expected regret values, one per model
    """
    if not models:
        raise ValueError("Models list cannot be empty")

    # Draw samples from all posteriors
    samples = np.array([model.sample() for model in models])  # shape: (n_variants, n_samples)

    # For each sample draw, find the maximum across variants
    max_values = np.max(samples, axis=0)  # shape: (n_samples,)

    # Expected regret = E[max - p_i]
    regret = np.mean(max_values - samples, axis=1)

    return regret


def log_marginal_likelihood(
    successes: int,
    failures: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0
) -> float:
    """
    Compute log marginal likelihood (evidence) for Beta-Binomial model.

    log P(data | alpha_prior, beta_prior) = log B(alpha_prior + s, beta_prior + f)
                                             - log B(alpha_prior, beta_prior)

    where B is the Beta function and s, f are successes and failures.

    This is useful for model comparison and computing Bayes factors.

    Args:
        successes: Number of successful trials
        failures: Number of failed trials
        alpha_prior: Prior alpha parameter
        beta_prior: Prior beta parameter

    Returns:
        Log marginal likelihood
    """
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures

    # Use log Beta function for numerical stability
    log_evidence = betaln(alpha_post, beta_post) - betaln(alpha_prior, beta_prior)

    return float(log_evidence)
