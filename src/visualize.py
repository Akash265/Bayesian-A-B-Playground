"""
Visualization utilities for Bayesian inference results.

Provides functions to plot posterior distributions, credible intervals,
P(best) comparisons, and expected regret.
"""

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats

from .beta_binomial import BetaBinomialModel
from .dirichlet_multinomial import DirichletMultinomialModel


def plot_beta_posteriors(
    models: List[BetaBinomialModel],
    variant_names: List[str],
    confidence: float = 0.95,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Plot posterior distributions for Beta-Binomial models.

    Args:
        models: List of BetaBinomialModel instances
        variant_names: Names for each variant
        confidence: Confidence level for credible intervals
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.linspace(0, 1, 1000)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, (model, name) in enumerate(zip(models, variant_names)):
        # Plot posterior PDF
        pdf = stats.beta.pdf(x, model.alpha, model.beta)
        ax.plot(x, pdf, label=name, color=colors[i], linewidth=2)

        # Add mean line
        mean = model.posterior_mean()
        ax.axvline(mean, color=colors[i], linestyle='--', alpha=0.5)

        # Shade credible interval
        ci_lower, ci_upper = model.credible_interval(confidence=confidence)
        mask = (x >= ci_lower) & (x <= ci_upper)
        ax.fill_between(x, 0, pdf, where=mask, alpha=0.2, color=colors[i])

    ax.set_xlabel('Conversion Rate', fontsize=12)
    ax.set_ylabel('Posterior Density', fontsize=12)
    ax.set_title(f'Posterior Distributions ({confidence*100:.0f}% Credible Intervals)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_p_best_comparison(
    p_best: np.ndarray,
    variant_names: List[str],
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot P(best) comparison as a bar chart.

    Args:
        p_best: Array of P(best) probabilities
        variant_names: Names for each variant
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(p_best)))
    bars = ax.bar(variant_names, p_best, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, prob in zip(bars, p_best):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{prob:.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_ylabel('P(Best)', fontsize=12)
    ax.set_title('Probability Each Variant is Best', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_expected_regret(
    regret: np.ndarray,
    variant_names: List[str],
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot expected regret comparison as a bar chart.

    Args:
        regret: Array of expected regret values
        variant_names: Names for each variant
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(regret)))
    bars = ax.bar(variant_names, regret, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, reg in zip(bars, regret):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{reg:.4f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_ylabel('Expected Regret', fontsize=12)
    ax.set_title('Expected Regret for Choosing Each Variant', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_credible_intervals(
    models: List[BetaBinomialModel],
    variant_names: List[str],
    confidence: float = 0.95,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot credible intervals with posterior means.

    Args:
        models: List of BetaBinomialModel instances
        variant_names: Names for each variant
        confidence: Confidence level for credible intervals
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    means = []
    lowers = []
    uppers = []

    for model in models:
        mean = model.posterior_mean()
        ci_lower, ci_upper = model.credible_interval(confidence=confidence)
        means.append(mean)
        lowers.append(ci_lower)
        uppers.append(ci_upper)

    y_pos = np.arange(len(variant_names))
    errors = np.array([[m - l for m, l in zip(means, lowers)],
                       [u - m for m, u in zip(means, uppers)]])

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    ax.errorbar(
        means,
        y_pos,
        xerr=errors,
        fmt='o',
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=2,
        ecolor=colors,
        markerfacecolor=colors,
        markeredgecolor='black',
        markeredgewidth=1.5
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variant_names)
    ax.set_xlabel('Conversion Rate', fontsize=12)
    ax.set_title(f'{confidence*100:.0f}% Credible Intervals', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig


def plot_dirichlet_posteriors(
    model: DirichletMultinomialModel,
    category_names: List[str],
    n_samples: int = 10000,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Plot marginal posterior distributions for Dirichlet-Multinomial model.

    Each category's marginal distribution is Beta(alpha_i, alpha_0 - alpha_i).

    Args:
        model: DirichletMultinomialModel instance
        category_names: Names for each category
        n_samples: Number of samples to draw
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.linspace(0, 1, 1000)
    colors = plt.cm.Set2(np.linspace(0, 1, len(category_names)))
    alpha_0 = np.sum(model.alphas)

    for i, (alpha_i, name) in enumerate(zip(model.alphas, category_names)):
        # Marginal distribution is Beta(alpha_i, alpha_0 - alpha_i)
        beta_i = alpha_0 - alpha_i
        pdf = stats.beta.pdf(x, alpha_i, beta_i)
        ax.plot(x, pdf, label=name, color=colors[i], linewidth=2)

        # Add mean line
        mean = model.posterior_mean()[i]
        ax.axvline(mean, color=colors[i], linestyle='--', alpha=0.5)

    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Marginal Posterior Density', fontsize=12)
    ax.set_title('Marginal Posterior Distributions (Dirichlet)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_decision_summary(
    models: List[BetaBinomialModel],
    variant_names: List[str],
    p_best: np.ndarray,
    regret: np.ndarray,
    confidence: float = 0.95,
    figsize: Tuple[int, int] = (16, 10)
) -> Figure:
    """
    Create comprehensive decision summary with multiple subplots.

    Args:
        models: List of BetaBinomialModel instances
        variant_names: Names for each variant
        p_best: Array of P(best) probabilities
        regret: Array of expected regret values
        confidence: Confidence level for credible intervals
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object with 4 subplots
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: Posterior distributions
    ax1 = fig.add_subplot(gs[0, :])
    x = np.linspace(0, 1, 1000)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, (model, name) in enumerate(zip(models, variant_names)):
        pdf = stats.beta.pdf(x, model.alpha, model.beta)
        ax1.plot(x, pdf, label=name, color=colors[i], linewidth=2)

        mean = model.posterior_mean()
        ax1.axvline(mean, color=colors[i], linestyle='--', alpha=0.5)

        ci_lower, ci_upper = model.credible_interval(confidence=confidence)
        mask = (x >= ci_lower) & (x <= ci_upper)
        ax1.fill_between(x, 0, pdf, where=mask, alpha=0.2, color=colors[i])

    ax1.set_xlabel('Conversion Rate', fontsize=11)
    ax1.set_ylabel('Posterior Density', fontsize=11)
    ax1.set_title('Posterior Distributions', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: P(best)
    ax2 = fig.add_subplot(gs[1, 0])
    bars2 = ax2.bar(variant_names, p_best, color=colors, edgecolor='black', linewidth=1.5)
    for bar, prob in zip(bars2, p_best):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height, f'{prob:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('P(Best)', fontsize=11)
    ax2.set_title('Probability of Being Best', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, axis='y', alpha=0.3)

    # Subplot 3: Expected regret
    ax3 = fig.add_subplot(gs[1, 1])
    regret_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(regret)))
    bars3 = ax3.bar(variant_names, regret, color=regret_colors, edgecolor='black', linewidth=1.5)
    for bar, reg in zip(bars3, regret):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height, f'{reg:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Expected Regret', fontsize=11)
    ax3.set_title('Expected Regret', fontsize=13, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Bayesian A/B Test Decision Summary', fontsize=16, fontweight='bold', y=0.995)

    return fig


def save_figure(fig: Figure, filepath: str, dpi: int = 300) -> None:
    """
    Save figure to file.

    Args:
        fig: Matplotlib Figure object
        filepath: Path to save file
        dpi: Dots per inch for output resolution
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
