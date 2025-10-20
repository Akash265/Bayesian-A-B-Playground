"""
Basic usage examples for Bayesian A/B Testing Playground.

This script demonstrates:
1. Beta-Binomial inference for A/B testing
2. P(best) and expected regret calculation
3. Visualization of results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.beta_binomial import (
    BetaBinomialModel,
    compute_p_best,
    compute_expected_regret,
)
from src.visualize import (
    plot_beta_posteriors,
    plot_p_best_comparison,
    plot_expected_regret,
    plot_decision_summary,
)


def main():
    """Run basic A/B testing example."""
    print("=" * 70)
    print("Bayesian A/B Testing Example")
    print("=" * 70)

    # Define experiment data
    variants = [
        {"name": "Control", "successes": 120, "failures": 80},
        {"name": "Treatment A", "successes": 140, "failures": 60},
        {"name": "Treatment B", "successes": 135, "failures": 65},
    ]

    print("\nExperiment Data:")
    print("-" * 70)
    for variant in variants:
        total = variant["successes"] + variant["failures"]
        obs_rate = variant["successes"] / total
        print(f"{variant['name']:15} | Conversions: {variant['successes']:3}/{total:3} | "
              f"Observed Rate: {obs_rate:.3f}")

    # Create posterior models
    print("\n\nBayesian Inference Results:")
    print("-" * 70)

    models = []
    for variant in variants:
        # Uniform prior: Beta(1, 1)
        model = BetaBinomialModel(alpha=1.0, beta=1.0, n_samples=10000, seed=42)

        # Update with observed data
        posterior = model.update(
            successes=variant["successes"],
            failures=variant["failures"]
        )
        models.append(posterior)

        # Display results
        mean = posterior.posterior_mean()
        variance = posterior.posterior_variance()
        ci_lower, ci_upper = posterior.credible_interval(confidence=0.95)

        print(f"\n{variant['name']}:")
        print(f"  Posterior Mean:     {mean:.4f}")
        print(f"  Posterior Std Dev:  {variance**0.5:.4f}")
        print(f"  95% Credible Int:   [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Posterior Params:   α={posterior.alpha:.1f}, β={posterior.beta:.1f}")

    # Compute decision metrics
    print("\n\nDecision Metrics:")
    print("-" * 70)

    p_best = compute_p_best(models)
    regret = compute_expected_regret(models)

    variant_names = [v["name"] for v in variants]

    for i, name in enumerate(variant_names):
        print(f"{name:15} | P(best): {p_best[i]:.4f} | Expected Regret: {regret[i]:.6f}")

    # Identify best variant
    best_idx = p_best.argmax()
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDATION: {variant_names[best_idx]} has the highest P(best) = {p_best[best_idx]:.4f}")
    print(f"{'=' * 70}")

    # Create visualizations
    print("\n\nGenerating Visualizations...")
    print("-" * 70)

    # Posterior distributions
    fig1 = plot_beta_posteriors(models, variant_names, confidence=0.95)
    fig1.savefig("posterior_distributions.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: posterior_distributions.png")

    # P(best) comparison
    fig2 = plot_p_best_comparison(p_best, variant_names)
    fig2.savefig("p_best_comparison.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: p_best_comparison.png")

    # Expected regret
    fig3 = plot_expected_regret(regret, variant_names)
    fig3.savefig("expected_regret.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: expected_regret.png")

    # Comprehensive summary
    fig4 = plot_decision_summary(models, variant_names, p_best, regret, confidence=0.95)
    fig4.savefig("decision_summary.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: decision_summary.png")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
