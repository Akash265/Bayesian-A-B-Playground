"""
Pydantic schemas for API request/response validation.

Ensures strict type checking and validation for all API inputs and outputs.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class VariantData(BaseModel):
    """Data for a single variant in A/B test."""

    name: str = Field(..., description="Variant identifier (e.g., 'Control', 'Treatment A')")
    successes: int = Field(..., ge=0, description="Number of successful conversions")
    failures: int = Field(..., ge=0, description="Number of failed conversions")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "Control",
            "successes": 120,
            "failures": 80
        }
    })


class BetaBinomialRequest(BaseModel):
    """Request schema for Beta-Binomial inference."""

    variants: List[VariantData] = Field(
        ...,
        min_length=2,
        description="List of variants to compare (minimum 2)"
    )
    prior_alpha: float = Field(
        1.0,
        gt=0,
        description="Beta prior alpha parameter (default 1.0 for uniform prior)"
    )
    prior_beta: float = Field(
        1.0,
        gt=0,
        description="Beta prior beta parameter (default 1.0 for uniform prior)"
    )
    confidence: float = Field(
        0.95,
        gt=0,
        lt=1,
        description="Credible interval confidence level (default 0.95)"
    )
    n_samples: int = Field(
        10000,
        ge=1000,
        le=1000000,
        description="Number of Monte Carlo samples for P(best) estimation"
    )
    seed: int = Field(
        42,
        description="Random seed for reproducibility"
    )

    @field_validator("variants")
    @classmethod
    def check_unique_names(cls, v: List[VariantData]) -> List[VariantData]:
        """Ensure all variant names are unique."""
        names = [variant.name for variant in v]
        if len(names) != len(set(names)):
            raise ValueError("All variant names must be unique")
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "variants": [
                {"name": "Control", "successes": 120, "failures": 80},
                {"name": "Treatment A", "successes": 135, "failures": 65},
                {"name": "Treatment B", "successes": 125, "failures": 75}
            ],
            "prior_alpha": 1.0,
            "prior_beta": 1.0,
            "confidence": 0.95,
            "n_samples": 10000,
            "seed": 42
        }
    })


class CategoryData(BaseModel):
    """Data for a single category in multinomial experiment."""

    name: str = Field(..., description="Category identifier")
    count: int = Field(..., ge=0, description="Observed count for this category")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "Option A",
            "count": 45
        }
    })


class DirichletMultinomialRequest(BaseModel):
    """Request schema for Dirichlet-Multinomial inference."""

    categories: List[CategoryData] = Field(
        ...,
        min_length=2,
        description="List of categories with observed counts (minimum 2)"
    )
    prior_alphas: List[float] | None = Field(
        None,
        description="Prior concentration parameters (default: uniform with alpha=1 for each)"
    )
    confidence: float = Field(
        0.95,
        gt=0,
        lt=1,
        description="Credible interval confidence level (default 0.95)"
    )
    n_samples: int = Field(
        10000,
        ge=1000,
        le=1000000,
        description="Number of Monte Carlo samples for P(best) estimation"
    )
    seed: int = Field(
        42,
        description="Random seed for reproducibility"
    )

    @field_validator("categories")
    @classmethod
    def check_unique_names(cls, v: List[CategoryData]) -> List[CategoryData]:
        """Ensure all category names are unique."""
        names = [cat.name for cat in v]
        if len(names) != len(set(names)):
            raise ValueError("All category names must be unique")
        return v

    @field_validator("prior_alphas")
    @classmethod
    def check_prior_alphas(cls, v: List[float] | None, info) -> List[float] | None:
        """Validate prior alphas if provided."""
        if v is not None:
            if any(alpha <= 0 for alpha in v):
                raise ValueError("All prior alphas must be positive")
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "categories": [
                {"name": "Option A", "count": 45},
                {"name": "Option B", "count": 38},
                {"name": "Option C", "count": 52}
            ],
            "prior_alphas": None,
            "confidence": 0.95,
            "n_samples": 10000,
            "seed": 42
        }
    })


class VariantResult(BaseModel):
    """Inference results for a single variant."""

    name: str = Field(..., description="Variant identifier")
    posterior_mean: float = Field(..., description="Posterior mean (expected conversion rate)")
    posterior_variance: float = Field(..., description="Posterior variance")
    credible_interval_lower: float = Field(..., description="Lower bound of credible interval")
    credible_interval_upper: float = Field(..., description="Upper bound of credible interval")
    p_best: float = Field(..., description="Probability this variant is the best")
    expected_regret: float = Field(
        ...,
        description="Expected regret from choosing this variant"
    )
    posterior_alpha: float = Field(..., description="Posterior alpha parameter")
    posterior_beta: float = Field(..., description="Posterior beta parameter")


class BetaBinomialResponse(BaseModel):
    """Response schema for Beta-Binomial inference."""

    results: List[VariantResult] = Field(..., description="Inference results per variant")
    best_variant: str = Field(..., description="Variant with highest P(best)")
    total_observations: int = Field(..., description="Total number of observations across variants")
    run_log: Dict[str, Any] = Field(
        ...,
        description="Complete run parameters for reproducibility"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
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
            "best_variant": "Treatment A",
            "total_observations": 600,
            "run_log": {
                "model_type": "beta_binomial",
                "prior_alpha": 1.0,
                "prior_beta": 1.0,
                "confidence": 0.95,
                "n_samples": 10000,
                "seed": 42
            }
        }
    })


class CategoryResult(BaseModel):
    """Inference results for a single category."""

    name: str = Field(..., description="Category identifier")
    posterior_mean: float = Field(..., description="Posterior mean probability")
    posterior_variance: float = Field(..., description="Posterior variance")
    credible_interval_lower: float = Field(..., description="Lower bound of credible interval")
    credible_interval_upper: float = Field(..., description="Upper bound of credible interval")
    p_best: float = Field(..., description="Probability this category is the best")
    expected_regret: float = Field(
        ...,
        description="Expected regret from choosing this category"
    )
    posterior_alpha: float = Field(..., description="Posterior concentration parameter")


class DirichletMultinomialResponse(BaseModel):
    """Response schema for Dirichlet-Multinomial inference."""

    results: List[CategoryResult] = Field(..., description="Inference results per category")
    best_category: str = Field(..., description="Category with highest P(best)")
    total_observations: int = Field(..., description="Total number of observations")
    concentration: float = Field(..., description="Total concentration parameter (sum of alphas)")
    run_log: Dict[str, Any] = Field(
        ...,
        description="Complete run parameters for reproducibility"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "results": [
                {
                    "name": "Option A",
                    "posterior_mean": 0.333,
                    "posterior_variance": 0.0015,
                    "credible_interval_lower": 0.256,
                    "credible_interval_upper": 0.413,
                    "p_best": 0.215,
                    "expected_regret": 0.052,
                    "posterior_alpha": 46.0
                }
            ],
            "best_category": "Option C",
            "total_observations": 135,
            "concentration": 138.0,
            "run_log": {
                "model_type": "dirichlet_multinomial",
                "prior_alphas": [1.0, 1.0, 1.0],
                "confidence": 0.95,
                "n_samples": 10000,
                "seed": 42
            }
        }
    })


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy"] = "healthy"
    version: str = "0.1.0"
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "version": "0.1.0"
        }
    })
