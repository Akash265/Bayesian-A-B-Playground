"""
Integration tests for FastAPI endpoints.

Tests cover:
- Request/response validation
- End-to-end inference workflows
- Error handling
- API schema compliance
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestBetaBinomialEndpoint:
    """Test Beta-Binomial inference endpoint."""

    def test_valid_request(self, client):
        """Test valid Beta-Binomial request."""
        payload = {
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

        response = client.post("/infer/beta-binomial", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "best_variant" in data
        assert "run_log" in data
        assert len(data["results"]) == 2

    def test_response_p_best_sums_to_one(self, client):
        """Property: P(best) values sum to 1."""
        payload = {
            "variants": [
                {"name": "A", "successes": 50, "failures": 50},
                {"name": "B", "successes": 60, "failures": 40},
                {"name": "C", "successes": 55, "failures": 45}
            ]
        }

        response = client.post("/infer/beta-binomial", json=payload)
        data = response.json()

        p_best_sum = sum(r["p_best"] for r in data["results"])
        assert abs(p_best_sum - 1.0) < 1e-6

    def test_response_regret_non_negative(self, client):
        """Property: Expected regret is non-negative."""
        payload = {
            "variants": [
                {"name": "A", "successes": 50, "failures": 50},
                {"name": "B", "successes": 60, "failures": 40}
            ]
        }

        response = client.post("/infer/beta-binomial", json=payload)
        data = response.json()

        for result in data["results"]:
            assert result["expected_regret"] >= 0

    def test_response_credible_intervals_valid(self, client):
        """Property: Credible intervals are valid probability ranges."""
        payload = {
            "variants": [
                {"name": "A", "successes": 50, "failures": 50},
                {"name": "B", "successes": 60, "failures": 40}
            ]
        }

        response = client.post("/infer/beta-binomial", json=payload)
        data = response.json()

        for result in data["results"]:
            assert 0 <= result["credible_interval_lower"] <= result["credible_interval_upper"] <= 1
            assert result["credible_interval_lower"] < result["posterior_mean"] < result["credible_interval_upper"]

    def test_insufficient_variants(self, client):
        """Test request with fewer than 2 variants fails."""
        payload = {
            "variants": [
                {"name": "A", "successes": 50, "failures": 50}
            ]
        }

        response = client.post("/infer/beta-binomial", json=payload)
        assert response.status_code == 422  # Validation error

    def test_duplicate_variant_names(self, client):
        """Test duplicate variant names are rejected."""
        payload = {
            "variants": [
                {"name": "A", "successes": 50, "failures": 50},
                {"name": "A", "successes": 60, "failures": 40}
            ]
        }

        response = client.post("/infer/beta-binomial", json=payload)
        assert response.status_code == 422

    def test_negative_counts(self, client):
        """Test negative counts are rejected."""
        payload = {
            "variants": [
                {"name": "A", "successes": -10, "failures": 50},
                {"name": "B", "successes": 60, "failures": 40}
            ]
        }

        response = client.post("/infer/beta-binomial", json=payload)
        assert response.status_code == 422

    def test_invalid_confidence(self, client):
        """Test invalid confidence level is rejected."""
        payload = {
            "variants": [
                {"name": "A", "successes": 50, "failures": 50},
                {"name": "B", "successes": 60, "failures": 40}
            ],
            "confidence": 1.5
        }

        response = client.post("/infer/beta-binomial", json=payload)
        assert response.status_code == 422

    def test_reproducibility_with_seed(self, client):
        """Property: Same seed produces identical results."""
        payload = {
            "variants": [
                {"name": "A", "successes": 50, "failures": 50},
                {"name": "B", "successes": 60, "failures": 40}
            ],
            "seed": 123
        }

        response1 = client.post("/infer/beta-binomial", json=payload)
        response2 = client.post("/infer/beta-binomial", json=payload)

        data1 = response1.json()
        data2 = response2.json()

        # P(best) should be identical with same seed
        for r1, r2 in zip(data1["results"], data2["results"]):
            assert abs(r1["p_best"] - r2["p_best"]) < 1e-10

    def test_run_log_completeness(self, client):
        """Test run log contains all parameters for reproducibility."""
        payload = {
            "variants": [
                {"name": "A", "successes": 50, "failures": 50},
                {"name": "B", "successes": 60, "failures": 40}
            ],
            "prior_alpha": 2.0,
            "prior_beta": 3.0,
            "seed": 99
        }

        response = client.post("/infer/beta-binomial", json=payload)
        data = response.json()
        run_log = data["run_log"]

        assert run_log["model_type"] == "beta_binomial"
        assert run_log["prior_alpha"] == 2.0
        assert run_log["prior_beta"] == 3.0
        assert run_log["seed"] == 99
        assert "variants" in run_log


class TestDirichletMultinomialEndpoint:
    """Test Dirichlet-Multinomial inference endpoint."""

    def test_valid_request(self, client):
        """Test valid Dirichlet-Multinomial request."""
        payload = {
            "categories": [
                {"name": "Option A", "count": 45},
                {"name": "Option B", "count": 38},
                {"name": "Option C", "count": 52}
            ],
            "confidence": 0.95,
            "n_samples": 10000,
            "seed": 42
        }

        response = client.post("/infer/dirichlet-multinomial", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "best_category" in data
        assert "concentration" in data
        assert "run_log" in data
        assert len(data["results"]) == 3

    def test_response_p_best_sums_to_one(self, client):
        """Property: P(best) values sum to 1."""
        payload = {
            "categories": [
                {"name": "A", "count": 30},
                {"name": "B", "count": 40},
                {"name": "C", "count": 35}
            ]
        }

        response = client.post("/infer/dirichlet-multinomial", json=payload)
        data = response.json()

        p_best_sum = sum(r["p_best"] for r in data["results"])
        assert abs(p_best_sum - 1.0) < 1e-6

    def test_response_regret_non_negative(self, client):
        """Property: Expected regret is non-negative."""
        payload = {
            "categories": [
                {"name": "A", "count": 30},
                {"name": "B", "count": 40}
            ]
        }

        response = client.post("/infer/dirichlet-multinomial", json=payload)
        data = response.json()

        for result in data["results"]:
            assert result["expected_regret"] >= 0

    def test_with_custom_priors(self, client):
        """Test request with custom prior alphas."""
        payload = {
            "categories": [
                {"name": "A", "count": 30},
                {"name": "B", "count": 40},
                {"name": "C", "count": 35}
            ],
            "prior_alphas": [2.0, 1.0, 1.0]
        }

        response = client.post("/infer/dirichlet-multinomial", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["run_log"]["prior_alphas"] == [2.0, 1.0, 1.0]

    def test_mismatched_prior_alphas(self, client):
        """Test mismatched prior alphas length fails."""
        payload = {
            "categories": [
                {"name": "A", "count": 30},
                {"name": "B", "count": 40},
                {"name": "C", "count": 35}
            ],
            "prior_alphas": [1.0, 1.0]  # Only 2 alphas for 3 categories
        }

        response = client.post("/infer/dirichlet-multinomial", json=payload)
        assert response.status_code == 400

    def test_negative_prior_alphas(self, client):
        """Test negative prior alphas are rejected."""
        payload = {
            "categories": [
                {"name": "A", "count": 30},
                {"name": "B", "count": 40}
            ],
            "prior_alphas": [1.0, -1.0]
        }

        response = client.post("/infer/dirichlet-multinomial", json=payload)
        assert response.status_code == 422

    def test_duplicate_category_names(self, client):
        """Test duplicate category names are rejected."""
        payload = {
            "categories": [
                {"name": "A", "count": 30},
                {"name": "A", "count": 40}
            ]
        }

        response = client.post("/infer/dirichlet-multinomial", json=payload)
        assert response.status_code == 422

    def test_negative_counts(self, client):
        """Test negative counts are rejected."""
        payload = {
            "categories": [
                {"name": "A", "count": -10},
                {"name": "B", "count": 40}
            ]
        }

        response = client.post("/infer/dirichlet-multinomial", json=payload)
        assert response.status_code == 422

    def test_reproducibility_with_seed(self, client):
        """Property: Same seed produces identical results."""
        payload = {
            "categories": [
                {"name": "A", "count": 30},
                {"name": "B", "count": 40},
                {"name": "C", "count": 35}
            ],
            "seed": 456
        }

        response1 = client.post("/infer/dirichlet-multinomial", json=payload)
        response2 = client.post("/infer/dirichlet-multinomial", json=payload)

        data1 = response1.json()
        data2 = response2.json()

        # P(best) should be identical with same seed
        for r1, r2 in zip(data1["results"], data2["results"]):
            assert abs(r1["p_best"] - r2["p_best"]) < 1e-10


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/infer/beta-binomial" in schema["paths"]
        assert "/infer/dirichlet-multinomial" in schema["paths"]

    def test_docs_page(self, client):
        """Test Swagger docs page is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_page(self, client):
        """Test ReDoc page is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
