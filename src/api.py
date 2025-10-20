"""
FastAPI application for Bayesian A/B Testing inference.

Provides RESTful endpoints for Beta-Binomial and Dirichlet-Multinomial inference
with strict input validation and structured JSON responses.
"""

from typing import Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    BetaBinomialRequest,
    BetaBinomialResponse,
    DirichletMultinomialRequest,
    DirichletMultinomialResponse,
    VariantResult,
    CategoryResult,
    HealthResponse,
)
from .beta_binomial import (
    BetaBinomialModel,
    compute_p_best,
    compute_expected_regret,
)
from .dirichlet_multinomial import (
    DirichletMultinomialModel,
    compute_p_best_dirichlet,
    compute_expected_regret_dirichlet,
    compute_concentration,
)

# Initialize FastAPI app
app = FastAPI(
    title="Bayesian A/B Testing Playground",
    description="Probabilistic decision engine using Beta-Binomial and Dirichlet-Multinomial inference",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Status and version information
    """
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/infer/beta-binomial", response_model=BetaBinomialResponse)
async def beta_binomial_inference(request: BetaBinomialRequest) -> BetaBinomialResponse:
    """
    Perform Bayesian inference for A/B testing with binary outcomes.

    Uses Beta-Binomial conjugate model to compute:
    - Posterior distributions for each variant
    - Credible intervals
    - P(best) for each variant
    - Expected regret

    Args:
        request: BetaBinomialRequest with variant data and parameters

    Returns:
        BetaBinomialResponse with inference results and run log

    Raises:
        HTTPException: If inference fails
    """
    try:
        # Create posterior models for each variant
        models = []
        for variant in request.variants:
            model = BetaBinomialModel(
                alpha=request.prior_alpha,
                beta=request.prior_beta,
                n_samples=request.n_samples,
                seed=request.seed,
            )
            posterior = model.update(
                successes=variant.successes,
                failures=variant.failures
            )
            models.append(posterior)

        # Compute decision metrics
        p_best = compute_p_best(models)
        expected_regret = compute_expected_regret(models)

        # Build results for each variant
        results = []
        for i, (variant, model) in enumerate(zip(request.variants, models)):
            ci_lower, ci_upper = model.credible_interval(confidence=request.confidence)

            result = VariantResult(
                name=variant.name,
                posterior_mean=model.posterior_mean(),
                posterior_variance=model.posterior_variance(),
                credible_interval_lower=ci_lower,
                credible_interval_upper=ci_upper,
                p_best=float(p_best[i]),
                expected_regret=float(expected_regret[i]),
                posterior_alpha=model.alpha,
                posterior_beta=model.beta,
            )
            results.append(result)

        # Identify best variant (highest P(best))
        best_idx = int(np.argmax(p_best))
        best_variant_name = request.variants[best_idx].name

        # Calculate total observations
        total_obs = sum(v.successes + v.failures for v in request.variants)

        # Create run log for reproducibility
        run_log: Dict[str, Any] = {
            "model_type": "beta_binomial",
            "prior_alpha": request.prior_alpha,
            "prior_beta": request.prior_beta,
            "confidence": request.confidence,
            "n_samples": request.n_samples,
            "seed": request.seed,
            "variants": [
                {
                    "name": v.name,
                    "successes": v.successes,
                    "failures": v.failures,
                }
                for v in request.variants
            ],
        }

        return BetaBinomialResponse(
            results=results,
            best_variant=best_variant_name,
            total_observations=total_obs,
            run_log=run_log,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {str(e)}")


@app.post("/infer/dirichlet-multinomial", response_model=DirichletMultinomialResponse)
async def dirichlet_multinomial_inference(
    request: DirichletMultinomialRequest,
) -> DirichletMultinomialResponse:
    """
    Perform Bayesian inference for categorical outcomes.

    Uses Dirichlet-Multinomial conjugate model to compute:
    - Posterior distributions for each category
    - Credible intervals
    - P(best) for each category
    - Expected regret

    Args:
        request: DirichletMultinomialRequest with category data and parameters

    Returns:
        DirichletMultinomialResponse with inference results and run log

    Raises:
        HTTPException: If inference fails
    """
    try:
        # Set up prior alphas (uniform by default)
        n_categories = len(request.categories)
        if request.prior_alphas is None:
            prior_alphas = np.ones(n_categories)
        else:
            if len(request.prior_alphas) != n_categories:
                raise ValueError(
                    f"Number of prior_alphas ({len(request.prior_alphas)}) "
                    f"must match number of categories ({n_categories})"
                )
            prior_alphas = np.array(request.prior_alphas)

        # Create posterior model
        prior_model = DirichletMultinomialModel(
            alphas=prior_alphas,
            n_samples=request.n_samples,
            seed=request.seed,
        )

        counts = np.array([cat.count for cat in request.categories])
        posterior_model = prior_model.update(counts)

        # Compute decision metrics
        p_best = compute_p_best_dirichlet(posterior_model)
        expected_regret = compute_expected_regret_dirichlet(posterior_model)
        concentration = compute_concentration(posterior_model)

        # Build results for each category
        results = []
        posterior_means = posterior_model.posterior_mean()
        posterior_vars = posterior_model.posterior_variance()

        for i, category in enumerate(request.categories):
            ci_lower, ci_upper = posterior_model.credible_interval(
                category=i,
                confidence=request.confidence
            )

            result = CategoryResult(
                name=category.name,
                posterior_mean=float(posterior_means[i]),
                posterior_variance=float(posterior_vars[i]),
                credible_interval_lower=ci_lower,
                credible_interval_upper=ci_upper,
                p_best=float(p_best[i]),
                expected_regret=float(expected_regret[i]),
                posterior_alpha=float(posterior_model.alphas[i]),
            )
            results.append(result)

        # Identify best category (highest P(best))
        best_idx = int(np.argmax(p_best))
        best_category_name = request.categories[best_idx].name

        # Calculate total observations
        total_obs = sum(cat.count for cat in request.categories)

        # Create run log for reproducibility
        run_log: Dict[str, Any] = {
            "model_type": "dirichlet_multinomial",
            "prior_alphas": prior_alphas.tolist(),
            "confidence": request.confidence,
            "n_samples": request.n_samples,
            "seed": request.seed,
            "categories": [
                {"name": cat.name, "count": cat.count}
                for cat in request.categories
            ],
        }

        return DirichletMultinomialResponse(
            results=results,
            best_category=best_category_name,
            total_observations=total_obs,
            concentration=concentration,
            run_log=run_log,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
