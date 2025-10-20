# Quick Start Guide

## Installation

```bash
cd bayesian-ab-playground
pip install -r requirements.txt
```

## 1. Run Tests (Recommended First Step)

```bash
# Run all tests
pytest tests/ -v

# Or use the provided script
./run_tests.sh
```

Expected output: **84 passed** with **~67% coverage**

## 2. Run the Take-Home Demo

This demonstrates exactly what Olumi is looking for:

```bash
python examples/take_home_demo.py
```

This will show:
- ✓ Beta-Binomial update for 3-option problem
- ✓ P(best) calculation
- ✓ Sensitivity analysis (modify input + compare metrics)
- ✓ Two property tests
- ✓ Explanation of Markovian sampling methods

## 3. Try the Basic Example

```bash
python examples/basic_usage.py
```

This creates visualization plots:
- `posterior_distributions.png`
- `p_best_comparison.png`
- `expected_regret.png`
- `decision_summary.png`

## 4. Start the API Server

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Then visit:
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## 5. Test the API

In another terminal:

```bash
python examples/api_client_example.py
```

Or use curl:

```bash
curl -X POST "http://localhost:8000/infer/beta-binomial" \
  -H "Content-Type: application/json" \
  -d '{
    "variants": [
      {"name": "Control", "successes": 120, "failures": 80},
      {"name": "Treatment", "successes": 140, "failures": 60}
    ],
    "seed": 42
  }'
```

## 6. Python API Usage

```python
from src.beta_binomial import BetaBinomialModel, compute_p_best

# Create and update model
model = BetaBinomialModel(alpha=1.0, beta=1.0, seed=42)
posterior = model.update(successes=50, failures=50)

# Get results
print(f"Mean: {posterior.posterior_mean():.3f}")
print(f"95% CI: {posterior.credible_interval()}")

# Compare multiple variants
models = [
    BetaBinomialModel(alpha=1, beta=1, seed=42).update(50, 50),
    BetaBinomialModel(alpha=1, beta=1, seed=42).update(60, 40),
]
p_best = compute_p_best(models)
print(f"P(best): {p_best}")
```

## Key Files to Review

For the Olumi application, these files demonstrate the required skills:

1. **`examples/take_home_demo.py`** - Addresses the take-home task exactly
2. **`src/beta_binomial.py`** - Core Bayesian inference implementation
3. **`tests/test_beta_binomial.py`** - Property tests and validation
4. **`src/api.py`** - FastAPI endpoint with strict validation
5. **`src/schemas.py`** - Pydantic schema validation

## Project Highlights

✅ **Beta-Binomial & Dirichlet-Multinomial** conjugate inference
✅ **P(best)** and **expected regret** decision metrics
✅ **FastAPI** with strict schema validation
✅ **Property-based tests** (84 tests, all passing)
✅ **Reproducible** with seeded RNG
✅ **Visualizations** for decision support
✅ **Full documentation** with docstrings and comments

## Next Steps

- Review the comprehensive [README.md](README.md)
- Check the [API documentation](http://localhost:8000/docs) (after starting server)
- Explore the test suite in `tests/`
- Try modifying `examples/take_home_demo.py` with different scenarios
