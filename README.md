# Bayesian A/B Testing Playground

A production-ready probabilistic decision engine implementing **Beta-Binomial** and **Dirichlet-Multinomial** Bayesian inference for A/B testing and multi-armed bandit problems.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Beta-Binomial Conjugate Inference**: Analytically compute posterior distributions for binary outcomes (e.g., conversion rates, click-through rates)
- **Dirichlet-Multinomial Inference**: Handle categorical outcomes with multiple choices
- **Decision Metrics**:
  - P(best): Probability each variant is the best performer
  - Expected regret: Expected loss from choosing each variant
  - Credible intervals: Bayesian confidence intervals for parameters
- **RESTful API**: FastAPI endpoints with strict schema validation
- **Reproducible**: Seeded random number generation for full reproducibility
- **Tested**: Comprehensive property tests and unit tests (>95% coverage)
- **Visualizations**: Publication-ready plots for posteriors, credible intervals, and decision metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Akash265/bayesian-ab-playground.git
cd bayesian-ab-playground

# Install dependencies
pip install -r requirements.txt

# Or install with pip
pip install -e .
```

### Run the API Server

```bash
# Start the FastAPI server
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

### Example: Beta-Binomial A/B Test

```python
from src.beta_binomial import BetaBinomialModel, compute_p_best, compute_expected_regret

# Define variants with observed data
variants = [
    {"name": "Control", "successes": 120, "failures": 80},
    {"name": "Treatment A", "successes": 140, "failures": 60},
    {"name": "Treatment B", "successes": 135, "failures": 65}
]

# Create posterior models
models = []
for variant in variants:
    model = BetaBinomialModel(alpha=1.0, beta=1.0, seed=42)
    posterior = model.update(
        successes=variant["successes"],
        failures=variant["failures"]
    )
    models.append(posterior)

    print(f"{variant['name']}:")
    print(f"  Posterior mean: {posterior.posterior_mean():.3f}")
    print(f"  95% CI: {posterior.credible_interval()}")

# Compute decision metrics
p_best = compute_p_best(models)
regret = compute_expected_regret(models)

for i, variant in enumerate(variants):
    print(f"{variant['name']}: P(best)={p_best[i]:.3f}, Regret={regret[i]:.4f}")
```

### Example: API Request

```bash
curl -X POST "http://localhost:8000/infer/beta-binomial" \
  -H "Content-Type: application/json" \
  -d '{
    "variants": [
      {"name": "Control", "successes": 120, "failures": 80},
      {"name": "Treatment", "successes": 140, "failures": 60}
    ],
    "prior_alpha": 1.0,
    "prior_beta": 1.0,
    "confidence": 0.95,
    "n_samples": 10000,
    "seed": 42
  }'
```

## Project Structure

```
bayesian-ab-playground/
├── src/
│   ├── __init__.py
│   ├── beta_binomial.py         # Beta-Binomial inference engine
│   ├── dirichlet_multinomial.py # Dirichlet-Multinomial inference
│   ├── schemas.py                # Pydantic validation schemas
│   ├── api.py                    # FastAPI application
│   └── visualize.py              # Visualization utilities
├── tests/
│   ├── __init__.py
│   ├── test_beta_binomial.py    # Beta-Binomial property tests
│   ├── test_dirichlet_multinomial.py
│   └── test_api.py               # API integration tests
├── notebooks/                     # Example Jupyter notebooks
├── docs/                          # Additional documentation
├── requirements.txt
├── pyproject.toml
└── README.md
```

## API Endpoints

### Health Check

```
GET /
```

Returns API status and version.

### Beta-Binomial Inference

```
POST /infer/beta-binomial
```

**Request Body:**

```json
{
  "variants": [
    {"name": "Control", "successes": 120, "failures": 80},
    {"name": "Treatment", "successes": 140, "failures": 60}
  ],
  "prior_alpha": 1.0,
  "prior_beta": 1.0,
  "confidence": 0.95,
  "n_samples": 10000,
  "seed": 42
}
```

**Response:**

```json
{
  "results": [
    {
      "name": "Control",
      "posterior_mean": 0.600,
      "posterior_variance": 0.0012,
      "credible_interval_lower": 0.532,
      "credible_interval_upper": 0.667,
      "p_best": 0.123,
      "expected_regret": 0.075,
      "posterior_alpha": 121.0,
      "posterior_beta": 81.0
    }
  ],
  "best_variant": "Treatment",
  "total_observations": 400,
  "run_log": {...}
}
```

### Dirichlet-Multinomial Inference

```
POST /infer/dirichlet-multinomial
```

**Request Body:**

```json
{
  "categories": [
    {"name": "Option A", "count": 45},
    {"name": "Option B", "count": 38},
    {"name": "Option C", "count": 52}
  ],
  "prior_alphas": null,
  "confidence": 0.95,
  "n_samples": 10000,
  "seed": 42
}
```

## Mathematical Background

### Beta-Binomial Model

The **Beta-Binomial** model is a conjugate Bayesian framework for binary outcomes:

- **Prior**: Beta(α, β)
- **Likelihood**: Binomial(n, p)
- **Posterior**: Beta(α + successes, β + failures)

**Key Properties:**
- Analytically tractable (no MCMC needed)
- Posterior mean: E[p|data] = (α + s) / (α + β + s + f)
- Conjugacy allows sequential updates

### Dirichlet-Multinomial Model

The **Dirichlet-Multinomial** model extends Beta-Binomial to categorical data:

- **Prior**: Dirichlet(α₁, ..., αₖ)
- **Likelihood**: Multinomial(n, p₁, ..., pₖ)
- **Posterior**: Dirichlet(α₁ + c₁, ..., αₖ + cₖ)

**Key Properties:**
- Marginal distributions are Beta: p_i ~ Beta(αᵢ, α₀ - αᵢ) where α₀ = Σαⱼ
- Posterior means sum to 1: Σ E[pᵢ] = 1

### Decision Metrics

**P(best)**: Monte Carlo estimate of P(variant i is best)

```
P(best_i) = P(pᵢ > pⱼ for all j ≠ i)
```

**Expected Regret**: Expected loss from choosing variant i

```
Regret(i) = E[max_j(pⱼ) - pᵢ | data]
```

## Running Tests

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_beta_binomial.py -v

# Run property tests only
pytest tests/ -v -k "property"
```

### Example Property Tests

The test suite includes rigorous property-based tests:

```python
def test_p_best_sums_to_one(self):
    """Property: P(best) probabilities sum to 1."""
    models = [...]
    p_best = compute_p_best(models)
    assert np.isclose(np.sum(p_best), 1.0, atol=1e-6)

def test_regret_all_non_negative(self):
    """Property: Expected regret is always non-negative."""
    models = [...]
    regret = compute_expected_regret(models)
    assert np.all(regret >= 0)

def test_reproducibility_with_seed(self):
    """Property: Same seed produces identical samples."""
    model1 = BetaBinomialModel(seed=123)
    model2 = BetaBinomialModel(seed=123)
    samples1 = model1.sample()
    samples2 = model2.sample()
    np.testing.assert_array_equal(samples1, samples2)
```

## Visualization Examples

```python
from src.visualize import (
    plot_beta_posteriors,
    plot_p_best_comparison,
    plot_credible_intervals,
    plot_decision_summary
)

# Plot posterior distributions with credible intervals
fig = plot_beta_posteriors(
    models=models,
    variant_names=["Control", "Treatment A", "Treatment B"],
    confidence=0.95
)
fig.savefig("posteriors.png", dpi=300, bbox_inches='tight')

# Plot P(best) comparison
fig = plot_p_best_comparison(p_best, variant_names)
fig.savefig("p_best.png", dpi=300, bbox_inches='tight')

# Comprehensive decision summary
fig = plot_decision_summary(
    models=models,
    variant_names=variant_names,
    p_best=p_best,
    regret=regret,
    confidence=0.95
)
fig.savefig("decision_summary.png", dpi=300, bbox_inches='tight')
```

## Use Cases

1. **A/B Testing**: Compare conversion rates between control and treatment groups
2. **Multi-Armed Bandits**: Select the best option among multiple choices
3. **Clinical Trials**: Bayesian analysis of treatment efficacy
4. **Marketing Optimization**: Compare email subject lines, ad creatives, landing pages
5. **Product Analytics**: Evaluate feature variants or UI/UX changes

## Design Principles

- **Type Safety**: Full type hints with Pydantic validation
- **Reproducibility**: All random operations seeded for deterministic results
- **Testability**: Property-based tests ensure mathematical correctness
- **Clarity**: Extensive docstrings and code comments
- **Performance**: NumPy vectorization for fast Monte Carlo sampling
- **API-First**: RESTful design with OpenAPI/Swagger documentation

## Performance Considerations

- **Monte Carlo Sampling**: Default 10,000 samples balances accuracy and speed
- **Vectorization**: NumPy operations avoid Python loops
- **Log-Space Numerics**: Marginal likelihood computed in log-space for numerical stability
- **Conjugacy**: Closed-form posteriors (no MCMC overhead)

Typical inference times on a standard laptop:
- Beta-Binomial with 3 variants: ~50ms
- Dirichlet-Multinomial with 5 categories: ~100ms

## Extending the Project

### Adding New Models

To add a new conjugate model (e.g., Gamma-Poisson):

1. Create `src/gamma_poisson.py` with model class
2. Implement `update()`, `posterior_mean()`, `credible_interval()`
3. Add decision metrics (P(best), expected regret)
4. Create corresponding tests in `tests/`
5. Add API endpoint in `src/api.py`

### Custom Priors

Informative priors can encode domain knowledge:

```python
# Weakly informative prior centered at 20% conversion
model = BetaBinomialModel(alpha=2.0, beta=8.0)  # Mode at 0.2

# Strong prior (100 pseudo-observations)
model = BetaBinomialModel(alpha=20.0, beta=80.0)
```

## References

### Academic Papers
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*

### Online Resources
- [Conjugate Prior - Wikipedia](https://en.wikipedia.org/wiki/Conjugate_prior)
- [Bayesian A/B Testing - Evan Miller](https://www.evanmiller.org/bayesian-ab-testing.html)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Akash Kaushik**
- Email: akash265457k@gmail.com
- GitHub: [@Akash265](https://github.com/Akash265)
- LinkedIn: [Akash Kaushik](https://www.linkedin.com/in/akash-kaushik-9100b0238/)

## Acknowledgments

Built as a demonstration project showcasing:
- Bayesian probabilistic inference
- FastAPI backend development
- Property-based testing
- Mathematical rigor in software engineering

---

**Keywords**: Bayesian inference, A/B testing, conjugate priors, Beta-Binomial, Dirichlet-Multinomial, decision theory, FastAPI, Python, statistics, machine learning
