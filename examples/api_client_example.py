"""
Example API client usage for Bayesian A/B Testing Playground.

This script demonstrates how to interact with the FastAPI endpoints.
"""

import requests
import json
from pprint import pprint


def test_health_check(base_url: str = "http://localhost:8000"):
    """Test health check endpoint."""
    print("=" * 70)
    print("Testing Health Check Endpoint")
    print("=" * 70)

    response = requests.get(f"{base_url}/")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())
    print()


def test_beta_binomial_inference(base_url: str = "http://localhost:8000"):
    """Test Beta-Binomial inference endpoint."""
    print("=" * 70)
    print("Testing Beta-Binomial Inference")
    print("=" * 70)

    payload = {
        "variants": [
            {"name": "Control", "successes": 120, "failures": 80},
            {"name": "Treatment A", "successes": 140, "failures": 60},
            {"name": "Treatment B", "successes": 135, "failures": 65}
        ],
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "confidence": 0.95,
        "n_samples": 10000,
        "seed": 42
    }

    print("\nRequest Payload:")
    print(json.dumps(payload, indent=2))

    response = requests.post(
        f"{base_url}/infer/beta-binomial",
        json=payload
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()

        print("\nInference Results:")
        print("-" * 70)

        for result in data["results"]:
            print(f"\n{result['name']}:")
            print(f"  Posterior Mean:     {result['posterior_mean']:.4f}")
            print(f"  95% CI:             [{result['credible_interval_lower']:.4f}, "
                  f"{result['credible_interval_upper']:.4f}]")
            print(f"  P(best):            {result['p_best']:.4f}")
            print(f"  Expected Regret:    {result['expected_regret']:.6f}")

        print(f"\nBest Variant: {data['best_variant']}")
        print(f"Total Observations: {data['total_observations']}")
    else:
        print("Error:")
        pprint(response.json())

    print()


def test_dirichlet_multinomial_inference(base_url: str = "http://localhost:8000"):
    """Test Dirichlet-Multinomial inference endpoint."""
    print("=" * 70)
    print("Testing Dirichlet-Multinomial Inference")
    print("=" * 70)

    payload = {
        "categories": [
            {"name": "Option A", "count": 45},
            {"name": "Option B", "count": 38},
            {"name": "Option C", "count": 52},
            {"name": "Option D", "count": 41}
        ],
        "confidence": 0.95,
        "n_samples": 10000,
        "seed": 42
    }

    print("\nRequest Payload:")
    print(json.dumps(payload, indent=2))

    response = requests.post(
        f"{base_url}/infer/dirichlet-multinomial",
        json=payload
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()

        print("\nInference Results:")
        print("-" * 70)

        for result in data["results"]:
            print(f"\n{result['name']}:")
            print(f"  Posterior Mean:     {result['posterior_mean']:.4f}")
            print(f"  95% CI:             [{result['credible_interval_lower']:.4f}, "
                  f"{result['credible_interval_upper']:.4f}]")
            print(f"  P(best):            {result['p_best']:.4f}")
            print(f"  Expected Regret:    {result['expected_regret']:.6f}")

        print(f"\nBest Category: {data['best_category']}")
        print(f"Total Observations: {data['total_observations']}")
        print(f"Concentration: {data['concentration']:.2f}")
    else:
        print("Error:")
        pprint(response.json())

    print()


def main():
    """Run all API examples."""
    base_url = "http://localhost:8000"

    print("\n" + "=" * 70)
    print("Bayesian A/B Testing API Client Examples")
    print("=" * 70)
    print(f"Base URL: {base_url}")
    print("=" * 70 + "\n")

    try:
        # Test endpoints
        test_health_check(base_url)
        test_beta_binomial_inference(base_url)
        test_dirichlet_multinomial_inference(base_url)

        print("=" * 70)
        print("All API Tests Completed Successfully!")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 70)
        print("ERROR: Could not connect to API server")
        print("=" * 70)
        print("\nPlease start the server first:")
        print("  uvicorn src.api:app --reload --host 0.0.0.0 --port 8000")
        print()


if __name__ == "__main__":
    main()
