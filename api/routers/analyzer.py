from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


class AnalyzerFactorWeights(BaseModel):
    quality: float = Field(default=0.30, ge=0)
    price_momentum: float = Field(default=0.40, ge=0)
    fundamental_momentum: float = Field(default=0.30, ge=0)
    valuation: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def require_nonzero(self):
        if self.quality + self.price_momentum + self.fundamental_momentum + self.valuation <= 0:
            raise ValueError("factor_weights must include at least one positive weight.")
        return self


class AnalyzerFundamentalMomentumWeights(BaseModel):
    revenue: float = Field(default=2.0, ge=0)
    eps: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def require_nonzero(self):
        if self.revenue + self.eps <= 0:
            raise ValueError("fundamental_momentum_weights must include at least one positive weight.")
        return self


class AnalyzerValuationWeights(BaseModel):
    price_sales: float = Field(default=1.0, ge=0)
    price_operating_income: float = Field(default=1.0, ge=0)
    price_fcf: float = Field(default=1.0, ge=0)
    price_earnings: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def require_nonzero(self):
        total = self.price_sales + self.price_operating_income + self.price_fcf + self.price_earnings
        if total <= 0:
            raise ValueError("valuation_weights must include at least one positive weight.")
        return self


class AnalyzerMetricScores(BaseModel):
    quality: float = Field(default=0.0, ge=0, le=100)
    price_momentum: float = Field(default=0.0, ge=0, le=100)
    revenue: float = Field(default=0.0, ge=0, le=100)
    eps: float = Field(default=0.0, ge=0, le=100)
    price_sales: float = Field(default=0.0, ge=0, le=100)
    price_operating_income: float = Field(default=0.0, ge=0, le=100)
    price_fcf: float = Field(default=0.0, ge=0, le=100)
    price_earnings: float = Field(default=0.0, ge=0, le=100)

    @model_validator(mode="after")
    def require_nonzero(self):
        total = (
            self.quality
            + self.price_momentum
            + self.revenue
            + self.eps
            + self.price_sales
            + self.price_operating_income
            + self.price_fcf
            + self.price_earnings
        )
        if total <= 0:
            raise ValueError("metric_scores must include at least one positive score.")
        return self


class AnalyzerScenarioBrakes(BaseModel):
    drawdown_sensitivity: float = Field(default=0.0, ge=0, le=100)
    contrarian_penalty: float = Field(default=0.0, ge=0, le=100)
    short_squeeze_brake: float = Field(default=0.0, ge=0, le=100)


class AnalyzerScenario(BaseModel):
    preset: str = "balanced"
    metric_scores: AnalyzerMetricScores | None = None
    factor_weights: AnalyzerFactorWeights = Field(default_factory=AnalyzerFactorWeights)
    fundamental_momentum_weights: AnalyzerFundamentalMomentumWeights = Field(
        default_factory=AnalyzerFundamentalMomentumWeights
    )
    valuation_weights: AnalyzerValuationWeights = Field(default_factory=AnalyzerValuationWeights)
    brakes: AnalyzerScenarioBrakes = Field(default_factory=AnalyzerScenarioBrakes)


class AnalyzerRequest(BaseModel):
    # Legacy optimizer fields are accepted for backward compatibility and ignored by analyzer logic.
    book: float | None = None
    target_leverage: float | None = None
    beta_neutral: bool | None = None
    scenario: AnalyzerScenario | None = None


class AnalyzerBriefRequest(BaseModel):
    action: dict[str, Any]


def _normalize_group(values: dict[str, Any]) -> dict[str, float]:
    numeric = {k: max(0.0, float(v or 0.0)) for k, v in values.items()}
    total = sum(numeric.values())
    if total <= 0:
        return numeric
    return {k: round(v / total, 8) for k, v in numeric.items()}


def _unit_brake_value(value: Any) -> float:
    numeric = max(0.0, float(value or 0.0))
    if numeric > 1.0:
        numeric = numeric / 100.0
    return round(max(0.0, min(1.0, numeric)), 8)


def _scenario_from_metric_scores(metric_scores: dict[str, Any]) -> dict[str, dict[str, float]]:
    scores = {key: max(0.0, float(value or 0.0)) for key, value in metric_scores.items()}
    total = sum(scores.values())
    if total <= 0:
        raise ValueError("metric_scores must include at least one positive score.")

    fundamental_total = scores.get("revenue", 0.0) + scores.get("eps", 0.0)
    valuation_total = (
        scores.get("price_sales", 0.0)
        + scores.get("price_operating_income", 0.0)
        + scores.get("price_fcf", 0.0)
        + scores.get("price_earnings", 0.0)
    )

    factor_weights = _normalize_group(
        {
            "quality": scores.get("quality", 0.0),
            "price_momentum": scores.get("price_momentum", 0.0),
            "fundamental_momentum": fundamental_total,
            "valuation": valuation_total,
        }
    )
    fundamental_momentum_weights = (
        _normalize_group({"revenue": scores.get("revenue", 0.0), "eps": scores.get("eps", 0.0)})
        if fundamental_total > 0
        else _normalize_group(AnalyzerFundamentalMomentumWeights().model_dump())
    )
    valuation_weights = (
        _normalize_group(
            {
                "price_sales": scores.get("price_sales", 0.0),
                "price_operating_income": scores.get("price_operating_income", 0.0),
                "price_fcf": scores.get("price_fcf", 0.0),
                "price_earnings": scores.get("price_earnings", 0.0),
            }
        )
        if valuation_total > 0
        else _normalize_group(AnalyzerValuationWeights().model_dump())
    )

    return {
        "factor_weights": factor_weights,
        "fundamental_momentum_weights": fundamental_momentum_weights,
        "valuation_weights": valuation_weights,
    }


def _canonical_scenario(req: AnalyzerRequest) -> dict[str, Any]:
    scenario = req.scenario or AnalyzerScenario()
    raw = scenario.model_dump()
    weights = (
        _scenario_from_metric_scores(raw["metric_scores"])
        if raw.get("metric_scores") is not None
        else {
            "factor_weights": _normalize_group(raw["factor_weights"]),
            "fundamental_momentum_weights": _normalize_group(raw["fundamental_momentum_weights"]),
            "valuation_weights": _normalize_group(raw["valuation_weights"]),
        }
    )
    return {
        "preset": raw.get("preset") or "balanced",
        **weights,
        "brakes": {k: _unit_brake_value(v) for k, v in raw["brakes"].items()},
    }


def _cache_key(req: AnalyzerRequest) -> str:
    strategy_version = "v3_course_of_action"
    scenario = json.dumps(_canonical_scenario(req), sort_keys=True, separators=(",", ":"))
    return f"portfolio_analyzer:{strategy_version}:scenario={scenario}"


def _compute_analyzer_result(req: AnalyzerRequest) -> dict[str, Any]:
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import get_data

        data = get_data(
            book=req.book,
            target_leverage=req.target_leverage,
            beta_neutral=True if req.beta_neutral is None else req.beta_neutral,
            scenario=_canonical_scenario(req),
        )
    except Exception as e:
        raise RuntimeError(str(e)) from e

    if "error" in data and data["error"]:
        raise RuntimeError(str(data["error"]))

    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return result


@router.post("/portfolio-analyzer")
@router.post("/portfolio-optimizer")
def run_analyzer(req: AnalyzerRequest = Body(default_factory=AnalyzerRequest)):  # noqa: B008
    row, _disposition = enqueue_registered_job("analyzer", req.model_dump(), cache_key=_cache_key(req))
    return enqueue_response(row, "/api/v1/portfolio-analyzer/async/{job_id}")


@router.post("/portfolio-analyzer/async")
@router.post("/portfolio-optimizer/async")
def start_analyzer(req: AnalyzerRequest = Body(default_factory=AnalyzerRequest)):  # noqa: B008
    """
    Start an analyzer job and return a job_id quickly.
    """
    row, _disposition = enqueue_registered_job("analyzer", req.model_dump(), cache_key=_cache_key(req))
    return enqueue_response(row, "/api/v1/portfolio-analyzer/async/{job_id}")


@router.get("/portfolio-analyzer/async/{job_id}")
@router.get("/portfolio-optimizer/async/{job_id}")
def get_analyzer_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904


@router.post("/portfolio-analyzer/course-of-action/brief")
def generate_course_of_action_brief(req: AnalyzerBriefRequest):
    try:
        from llm_utils import MODEL_LOW, call_llm_text, has_llm_api_key

        if not has_llm_api_key():
            raise HTTPException(status_code=424, detail="No configured LLM API key for analyzer briefs.")

        evidence_json = json.dumps(req.action, sort_keys=True, default=str)
        prompt = (
            "Write a concise portfolio analyzer brief from the structured evidence below. "
            "Do not rescore the recommendation, do not invent missing evidence, and do not give executable order instructions. "
            "Use 3 short bullets: action, evidence, watch-outs.\n\n"
            f"Evidence:\n{evidence_json}"
        )
        text, _citations, _response = call_llm_text(
            prompt=prompt,
            model=MODEL_LOW,
            max_tokens=600,
            system="You summarize deterministic portfolio-analysis evidence. You never add facts not present in the input.",
            max_web_search_uses=0,
        )
        return {"brief": text.strip()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate analyzer brief: {e}") from e
