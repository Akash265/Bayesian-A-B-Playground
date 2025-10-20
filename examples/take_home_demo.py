"""
Demonstration of the Olumi take-home task requirements.

This script demonstrates:
1. Beta-Binomial update for 3-option toy problem
2. P(best) calculation
3. Sensitivity analysis (modify one input and show metric changes)
4. Property tests for validation
5. Description of Markovian sampling methods

Task Description:
- Implement a Beta-Binomial update and P(best) on a 3-option toy
- Modify one input and report how key metrics change
- Include two property tests
- Describe how to use Markovian methods to sample from posterior distributions
"""

import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.beta_binomial import (
    BetaBinomialModel,
    compute_p_best,
    compute_expected_regret,
)


def run_baseline_scenario() -> Tuple[List[BetaBinomialModel], np.ndarray, np.ndarray]:
    """
    Run baseline 3-option A/B test.

    Returns:
        Tuple of (models, p_best, expected_regret)
    """
    print("=" * 80)
    print("BASELINE SCENARIO: 3-Option A/B Test")
    print("=" * 80)

    # Define 3 options with observed data
    options = [
        {"name": "Option A", "successes": 50, "failures": 50},
        {"name": "Option B", "successes": 60, "failures": 40},
        {"name": "Option C", "successes": 55, "failures": 45},
    ]

    print("\nObserved Data:")
    print("-" * 80)
    for opt in options:
        total = opt["successes"] + opt["failures"]
        rate = opt["successes"] / total
        print(f"{opt['name']:12} | Successes: {opt['successes']:3}/{total:3} | "
              f"Observed Rate: {rate:.3f}")

    # Perform Bayesian update with uniform prior Beta(1, 1)
    models = []
    print("\n\nBayesian Posterior Results:")
    print("-" * 80)

    for opt in options:
        model = BetaBinomialModel(alpha=1.0, beta=1.0, n_samples=10000, seed=42)
        posterior = model.update(
            successes=opt["successes"],
            failures=opt["failures"]
        )
        models.append(posterior)

        mean = posterior.posterior_mean()
        ci_lower, ci_upper = posterior.credible_interval(confidence=0.95)

        print(f"\n{opt['name']}:")
        print(f"  Posterior: Beta({posterior.alpha:.0f}, {posterior.beta:.0f})")
        print(f"  Mean:      {mean:.4f}")
        print(f"  95% CI:    [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Compute decision metrics
    p_best = compute_p_best(models)
    regret = compute_expected_regret(models)

    print("\n\nDecision Metrics:")
    print("-" * 80)
    print(f"{'Option':<12} | {'P(best)':<10} | {'Expected Regret':<20}")
    print("-" * 80)

    for i, opt in enumerate(options):
        print(f"{opt['name']:<12} | {p_best[i]:<10.4f} | {regret[i]:<20.6f}")

    best_idx = np.argmax(p_best)
    print(f"\n{'='*80}")
    print(f"BEST OPTION: {options[best_idx]['name']} (P(best) = {p_best[best_idx]:.4f})")
    print(f"{'='*80}\n")

    return models, p_best, regret


def run_sensitivity_analysis():
    """
    Modify one input and show how metrics change.

    Sensitivity analysis: Increase Option B successes from 60 to 70.
    """
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: Modify Option B (60 → 70 successes)")
    print("=" * 80)

    # Modified data: Increase Option B performance
    options = [
        {"name": "Option A", "successes": 50, "failures": 50},
        {"name": "Option B", "successes": 70, "failures": 40},  # Changed!
        {"name": "Option C", "successes": 55, "failures": 45},
    ]

    print("\nModified Data:")
    print("-" * 80)
    for opt in options:
        total = opt["successes"] + opt["failures"]
        rate = opt["successes"] / total
        marker = " ← MODIFIED" if opt["name"] == "Option B" else ""
        print(f"{opt['name']:12} | Successes: {opt['successes']:3}/{total:3} | "
              f"Observed Rate: {rate:.3f}{marker}")

    # Perform Bayesian update
    models = []
    print("\n\nBayesian Posterior Results:")
    print("-" * 80)

    for opt in options:
        model = BetaBinomialModel(alpha=1.0, beta=1.0, n_samples=10000, seed=42)
        posterior = model.update(
            successes=opt["successes"],
            failures=opt["failures"]
        )
        models.append(posterior)

        mean = posterior.posterior_mean()
        ci_lower, ci_upper = posterior.credible_interval(confidence=0.95)

        print(f"\n{opt['name']}:")
        print(f"  Posterior: Beta({posterior.alpha:.0f}, {posterior.beta:.0f})")
        print(f"  Mean:      {mean:.4f}")
        print(f"  95% CI:    [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Compute decision metrics
    p_best = compute_p_best(models)
    regret = compute_expected_regret(models)

    print("\n\nDecision Metrics (After Modification):")
    print("-" * 80)
    print(f"{'Option':<12} | {'P(best)':<10} | {'Expected Regret':<20}")
    print("-" * 80)

    for i, opt in enumerate(options):
        print(f"{opt['name']:<12} | {p_best[i]:<10.4f} | {regret[i]:<20.6f}")

    best_idx = np.argmax(p_best)
    print(f"\n{'='*80}")
    print(f"BEST OPTION: {options[best_idx]['name']} (P(best) = {p_best[best_idx]:.4f})")
    print(f"{'='*80}\n")

    return models, p_best, regret


def run_property_tests():
    """
    Run two property tests as required by the take-home task.
    """
    print("=" * 80)
    print("PROPERTY TESTS")
    print("=" * 80)

    # Test 1: P(best) sums to 1
    print("\nProperty Test 1: P(best) probabilities sum to 1")
    print("-" * 80)

    models = [
        BetaBinomialModel(alpha=1.0, beta=1.0, seed=42).update(50, 50),
        BetaBinomialModel(alpha=1.0, beta=1.0, seed=42).update(60, 40),
        BetaBinomialModel(alpha=1.0, beta=1.0, seed=42).update(55, 45),
    ]

    p_best = compute_p_best(models)
    p_best_sum = np.sum(p_best)

    print(f"P(best) values: {p_best}")
    print(f"Sum of P(best): {p_best_sum:.10f}")
    print(f"Expected sum:   1.0")
    print(f"Difference:     {abs(p_best_sum - 1.0):.2e}")

    assert np.isclose(p_best_sum, 1.0, atol=1e-6), "P(best) must sum to 1"
    print("✓ PASSED: P(best) sums to 1 within tolerance\n")

    # Test 2: Expected regret is non-negative
    print("\nProperty Test 2: Expected regret is always non-negative")
    print("-" * 80)

    regret = compute_expected_regret(models)

    print(f"Expected regret values: {regret}")
    print(f"All non-negative? {np.all(regret >= 0)}")

    assert np.all(regret >= 0), "Expected regret must be non-negative"
    print("✓ PASSED: All expected regret values are non-negative\n")

    print("=" * 80)
    print("All Property Tests Passed!")
    print("=" * 80 + "\n")


def explain_markovian_sampling():
    """
    Explain how to use Markovian methods to sample from posterior distributions.
    """
    print("=" * 80)
    print("MARKOVIAN SAMPLING METHODS FOR POSTERIOR DISTRIBUTIONS")
    print("=" * 80)

    explanation = """
For Beta-Binomial and Dirichlet-Multinomial models, we use conjugate priors,
so posterior distributions have closed-form solutions. However, for general
Bayesian inference, we need Markovian sampling methods.

1. MARKOV CHAIN MONTE CARLO (MCMC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MCMC constructs a Markov chain whose stationary distribution is the target
posterior distribution. After a "burn-in" period, samples from the chain
approximate samples from the posterior.

Key Markov Property: P(θₜ₊₁ | θₜ, θₜ₋₁, ..., θ₀) = P(θₜ₊₁ | θₜ)
(Future state depends only on current state, not past history)


2. METROPOLIS-HASTINGS ALGORITHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Most general MCMC method:

Algorithm:
1. Initialize θ₀ arbitrarily
2. For t = 0, 1, 2, ..., T:
   a. Propose θ* ~ q(θ* | θₜ)  [proposal distribution]
   b. Compute acceptance ratio:
      α = min(1, [P(data|θ*) P(θ*) q(θₜ|θ*)] / [P(data|θₜ) P(θₜ) q(θ*|θₜ)])
   c. Accept θₜ₊₁ = θ* with probability α, else θₜ₊₁ = θₜ

The chain satisfies detailed balance and converges to P(θ | data).

Example (Python pseudocode):
    def metropolis_hastings(data, prior, likelihood, n_samples, proposal_std):
        theta = initial_value
        samples = []

        for i in range(n_samples):
            # Propose new value
            theta_star = theta + np.random.normal(0, proposal_std)

            # Compute acceptance probability
            log_alpha = (
                likelihood(data, theta_star) + prior(theta_star)
                - likelihood(data, theta) - prior(theta)
            )

            # Accept/reject
            if np.log(np.random.rand()) < log_alpha:
                theta = theta_star

            samples.append(theta)

        return np.array(samples[burnin:])


3. GIBBS SAMPLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Special case of Metropolis-Hastings for multivariate distributions.
Sample each parameter conditional on all others.

Algorithm:
1. Initialize θ = (θ₁⁽⁰⁾, θ₂⁽⁰⁾, ..., θₖ⁽⁰⁾)
2. For t = 1, 2, ..., T:
   θ₁⁽ᵗ⁾ ~ P(θ₁ | θ₂⁽ᵗ⁻¹⁾, θ₃⁽ᵗ⁻¹⁾, ..., θₖ⁽ᵗ⁻¹⁾, data)
   θ₂⁽ᵗ⁾ ~ P(θ₂ | θ₁⁽ᵗ⁾,   θ₃⁽ᵗ⁻¹⁾, ..., θₖ⁽ᵗ⁻¹⁾, data)
   ...
   θₖ⁽ᵗ⁾ ~ P(θₖ | θ₁⁽ᵗ⁾,   θ₂⁽ᵗ⁾,   ..., θₖ₋₁⁽ᵗ⁾,   data)

Particularly efficient when conditional distributions have closed forms.


4. HAMILTONIAN MONTE CARLO (HMC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Uses gradient information and physics-inspired dynamics to propose moves.
More efficient than random walk for high-dimensional problems.

Introduces auxiliary momentum variables and simulates Hamiltonian dynamics.

Algorithm:
1. Define potential energy: U(θ) = -log P(θ | data)
2. Sample momentum: p ~ N(0, M)
3. Simulate Hamiltonian dynamics using leapfrog integration:
   - Update momentum: p ← p - (ε/2) ∇U(θ)
   - Update position: θ ← θ + ε M⁻¹ p
   - Update momentum: p ← p - (ε/2) ∇U(θ)
4. Accept/reject using Metropolis criterion

Used in modern tools like Stan and PyMC.


5. APPLICATION TO BETA-BINOMIAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For Beta-Binomial, we DON'T need MCMC (closed-form posterior), but if we did:

Target posterior: P(p | s, f) ∝ p^(α+s-1) (1-p)^(β+f-1)

Metropolis-Hastings Implementation:
    def sample_beta_binomial_mcmc(alpha, beta, successes, failures, n_samples):
        p = 0.5  # Initialize
        samples = []

        for i in range(n_samples):
            # Propose using normal random walk on logit scale
            logit_p = np.log(p / (1 - p))
            logit_p_star = logit_p + np.random.normal(0, 0.5)
            p_star = 1 / (1 + np.exp(-logit_p_star))

            # Compute log-posterior ratio
            log_ratio = (
                (alpha + successes - 1) * np.log(p_star / p)
                + (beta + failures - 1) * np.log((1 - p_star) / (1 - p))
            )

            # Accept/reject
            if np.log(np.random.rand()) < log_ratio:
                p = p_star

            samples.append(p)

        return np.array(samples[1000:])  # Discard burn-in


6. CONVERGENCE DIAGNOSTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key diagnostics for MCMC:
- Trace plots: Visual inspection of chain behavior
- Autocorrelation: Check for slow mixing
- Effective Sample Size (ESS): n_samples / (1 + 2∑ρₖ)
- Gelman-Rubin statistic (R̂): Compare multiple chains
- Acceptance rate: Target ~0.234 for random walk, ~0.65 for HMC


7. SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When to use MCMC:
✓ Non-conjugate priors
✓ Complex likelihood functions
✓ High-dimensional parameter spaces
✓ Hierarchical models

When NOT needed:
✗ Conjugate priors (Beta-Binomial, Dirichlet-Multinomial)
✗ Simple models with closed-form posteriors

This project uses direct sampling from Beta/Dirichlet distributions for
efficiency. For more complex models (e.g., mixture models, hierarchical
priors), we would implement Metropolis-Hastings or HMC.
    """

    print(explanation)
    print("=" * 80 + "\n")


def main():
    """Run complete take-home demonstration."""
    print("\n" + "=" * 80)
    print("OLUMI TAKE-HOME TASK DEMONSTRATION")
    print("=" * 80)
    print()

    # Part 1: Baseline scenario
    baseline_models, baseline_p_best, baseline_regret = run_baseline_scenario()

    input("\nPress Enter to continue to sensitivity analysis...\n")

    # Part 2: Sensitivity analysis
    modified_models, modified_p_best, modified_regret = run_sensitivity_analysis()

    print("=" * 80)
    print("COMPARISON: Baseline vs. Modified")
    print("=" * 80)
    print("\nP(best) Changes:")
    print("-" * 80)
    print(f"{'Option':<12} | {'Baseline':<12} | {'Modified':<12} | {'Change':<12}")
    print("-" * 80)

    option_names = ["Option A", "Option B", "Option C"]
    for i, name in enumerate(option_names):
        change = modified_p_best[i] - baseline_p_best[i]
        print(f"{name:<12} | {baseline_p_best[i]:<12.4f} | "
              f"{modified_p_best[i]:<12.4f} | {change:+12.4f}")

    print("\n" + "=" * 80)
    print(f"KEY INSIGHT: Increasing Option B from 60→70 successes increased its")
    print(f"P(best) from {baseline_p_best[1]:.3f} to {modified_p_best[1]:.3f} "
          f"(Δ = +{modified_p_best[1] - baseline_p_best[1]:.3f})")
    print("=" * 80 + "\n")

    input("Press Enter to run property tests...\n")

    # Part 3: Property tests
    run_property_tests()

    input("Press Enter to see Markovian sampling explanation...\n")

    # Part 4: Markovian methods explanation
    explain_markovian_sampling()

    print("=" * 80)
    print("TAKE-HOME DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll requirements demonstrated:")
    print("  ✓ Beta-Binomial update for 3-option problem")
    print("  ✓ P(best) calculation")
    print("  ✓ Sensitivity analysis (modified input + metric comparison)")
    print("  ✓ Two property tests included")
    print("  ✓ Markovian sampling methods explained")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
