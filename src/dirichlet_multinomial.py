"""
Dirichlet-Multinomial Bayesian inference for multi-armed bandit problems.

This module implements conjugate Bayesian updates for categorical outcomes using the
Dirichlet-Multinomial model, generalizing the Beta-Binomial to multiple categories.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats
from scipy.special import gammaln


class DirichletMultinomialModel:
    """
    Dirichlet-Multinomial conjugate model for Bayesian inference with categorical data.

    The Dirichlet distribution is a conjugate prior for the Multinomial likelihood,
    making posterior updates analytically tractable.

    Attributes:
        alphas: Dirichlet concentration parameters (pseudo-counts + prior)
        n_samples: Number of posterior samples for Monte Carlo estimation
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        alphas: np.ndarray | List[float],
        n_samples: int = 10000,
        seed: int = 42
    ) -> None:
        """
        Initialize Dirichlet prior.

        Args:
            alphas: Concentration parameters (length = number of categories)
                   alphas = [1, 1, ..., 1] gives uniform prior
            n_samples: Number of Monte Carlo samples for P(best) estimation
            seed: Random seed for reproducibility

        Raises:
            ValueError: If any alpha is not positive or alphas is empty
        """
        self.alphas = np.atleast_1d(alphas).astype(float)

        if len(self.alphas) == 0:
            raise ValueError("Alphas array cannot be empty")

        if np.any(self.alphas <= 0):
            raise ValueError(f"All alphas must be positive, got {self.alphas}")

        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_categories = len(self.alphas)

    def update(self, counts: np.ndarray | List[int]) -> "DirichletMultinomialModel":
        """
        Perform Bayesian update with new observations.

        Posterior: Dirichlet(alpha_1 + count_1, ..., alpha_k + count_k)

        Args:
            counts: Observed counts for each category

        Returns:
            New DirichletMultinomialModel with updated posterior parameters

        Raises:
            ValueError: If counts length doesn't match alphas or contains negative values
        """
        counts_array = np.atleast_1d(counts).astype(float)

        if len(counts_array) != self.n_categories:
            raise ValueError(
                f"Counts length ({len(counts_array)}) must match "
                f"number of categories ({self.n_categories})"
            )

        if np.any(counts_array < 0):
            raise ValueError(f"All counts must be non-negative, got {counts_array}")

        return DirichletMultinomialModel(
            alphas=self.alphas + counts_array,
            n_samples=self.n_samples,
            seed=self.seed
        )

    def posterior_mean(self) -> np.ndarray:
        """
        Compute posterior expectation E[p | data] for each category.

        For Dirichlet(alpha_1, ..., alpha_k):
        E[p_i] = alpha_i / sum_j(alpha_j)

        Returns:
            Array of posterior means, one per category
        """
        return self.alphas / np.sum(self.alphas)

    def posterior_variance(self) -> np.ndarray:
        """
        Compute posterior variance Var[p_i | data] for each category.

        For Dirichlet(alpha_1, ..., alpha_k):
        Var[p_i] = (alpha_i * (alpha_0 - alpha_i)) / (alpha_0^2 * (alpha_0 + 1))
        where alpha_0 = sum_j(alpha_j)

        Returns:
            Array of posterior variances, one per category
        """
        alpha_0 = np.sum(self.alphas)
        numerator = self.alphas * (alpha_0 - self.alphas)
        denominator = (alpha_0 ** 2) * (alpha_0 + 1)
        return numerator / denominator

    def posterior_covariance(self) -> np.ndarray:
        """
        Compute posterior covariance matrix Cov[p_i, p_j | data].

        For i != j:
        Cov[p_i, p_j] = -alpha_i * alpha_j / (alpha_0^2 * (alpha_0 + 1))

        Returns:
            Covariance matrix of shape (n_categories, n_categories)
        """
        alpha_0 = np.sum(self.alphas)
        cov = -np.outer(self.alphas, self.alphas)
        cov /= (alpha_0 ** 2) * (alpha_0 + 1)

        # Set diagonal to variances
        np.fill_diagonal(cov, self.posterior_variance())

        return cov

    def credible_interval(self, category: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute marginal credible interval for a specific category.

        The marginal distribution of p_i is Beta(alpha_i, alpha_0 - alpha_i).

        Args:
            category: Category index (0-indexed)
            confidence: Credible level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)

        Raises:
            ValueError: If category index is out of bounds or confidence not in (0, 1)
        """
        if not 0 <= category < self.n_categories:
            raise ValueError(
                f"Category must be in [0, {self.n_categories-1}], got {category}"
            )

        if not 0 < confidence < 1:
            raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

        alpha_i = self.alphas[category]
        alpha_0 = np.sum(self.alphas)
        beta_i = alpha_0 - alpha_i

        tail_prob = (1 - confidence) / 2
        lower = stats.beta.ppf(tail_prob, alpha_i, beta_i)
        upper = stats.beta.ppf(1 - tail_prob, alpha_i, beta_i)

        return (float(lower), float(upper))

    def sample(self, n_samples: int | None = None) -> np.ndarray:
        """
        Draw samples from the posterior Dirichlet distribution.

        Args:
            n_samples: Number of samples (uses self.n_samples if None)

        Returns:
            Array of shape (n_samples, n_categories) with posterior samples
        """
        if n_samples is None:
            n_samples = self.n_samples

        return self.rng.dirichlet(self.alphas, size=n_samples)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model state to dictionary for serialization.

        Returns:
            Dictionary with model parameters and summary statistics
        """
        return {
            "alphas": self.alphas.tolist(),
            "n_categories": self.n_categories,
            "posterior_mean": self.posterior_mean().tolist(),
            "posterior_variance": self.posterior_variance().tolist(),
            "n_samples": self.n_samples,
            "seed": self.seed
        }


def compute_p_best_dirichlet(model: DirichletMultinomialModel) -> np.ndarray:
    """
    Compute P(best) for each category using Monte Carlo sampling.

    P(best_i) = P(p_i > p_j for all j != i)

    Args:
        model: DirichletMultinomialModel instance

    Returns:
        Array of P(best) probabilities, one per category
    """
    # Draw samples from posterior
    samples = model.sample()  # shape: (n_samples, n_categories)

    # Find winner for each sample
    winners = np.argmax(samples, axis=1)  # shape: (n_samples,)

    # Count wins for each category
    p_best = np.zeros(model.n_categories)
    for i in range(model.n_categories):
        p_best[i] = np.mean(winners == i)

    return p_best


def compute_expected_regret_dirichlet(model: DirichletMultinomialModel) -> np.ndarray:
    """
    Compute expected regret for choosing each category.

    Expected regret for choosing category i:
    E[max_j(p_j) - p_i | data]

    Args:
        model: DirichletMultinomialModel instance

    Returns:
        Array of expected regret values, one per category
    """
    # Draw samples from posterior
    samples = model.sample()  # shape: (n_samples, n_categories)

    # For each sample draw, find the maximum across categories
    max_values = np.max(samples, axis=1, keepdims=True)  # shape: (n_samples, 1)

    # Expected regret = E[max - p_i]
    regret = np.mean(max_values - samples, axis=0)

    return regret


def log_marginal_likelihood_multinomial(
    counts: np.ndarray | List[int],
    alphas_prior: np.ndarray | List[float]
) -> float:
    """
    Compute log marginal likelihood (evidence) for Dirichlet-Multinomial model.

    log P(data | alphas_prior) uses the multivariate Beta function (Dirichlet integral).

    This is useful for model comparison and computing Bayes factors.

    Args:
        counts: Observed counts for each category
        alphas_prior: Prior concentration parameters

    Returns:
        Log marginal likelihood

    Raises:
        ValueError: If counts and alphas_prior have different lengths
    """
    counts_array = np.atleast_1d(counts).astype(float)
    alphas_array = np.atleast_1d(alphas_prior).astype(float)

    if len(counts_array) != len(alphas_array):
        raise ValueError("Counts and alphas_prior must have the same length")

    alphas_post = alphas_array + counts_array

    # Log multivariate Beta function = sum(log Gamma(alpha_i)) - log Gamma(sum(alpha_i))
    def log_mvbeta(alphas: np.ndarray) -> float:
        return float(np.sum(gammaln(alphas)) - gammaln(np.sum(alphas)))

    log_evidence = log_mvbeta(alphas_post) - log_mvbeta(alphas_array)

    return log_evidence


def compute_concentration(model: DirichletMultinomialModel) -> float:
    """
    Compute total concentration parameter (sum of alphas).

    Higher concentration means more concentrated/peaked distribution.

    Args:
        model: DirichletMultinomialModel instance

    Returns:
        Total concentration (sum of all alphas)
    """
    return float(np.sum(model.alphas))
