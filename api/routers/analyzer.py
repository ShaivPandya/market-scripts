from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.serializers import serialize_dataframe, serialize_value
from portfolio.portfolio_optimizer.analyzer_scenarios import (
    SCENARIO_BRAKE_DEFAULTS,
    SCENARIO_FACTOR_DEFAULTS,
    SCENARIO_FUNDAMENTAL_DEFAULTS,
    SCENARIO_METRIC_SCORE_DEFAULTS,
    SCENARIO_QUALITATIVE_DEFAULTS,
    SCENARIO_VALUATION_DEFAULTS,
    normalize_analyzer_scenario,
)

router = APIRouter()


class AnalyzerFactorWeights(BaseModel):
    quality: float = Field(default=SCENARIO_FACTOR_DEFAULTS["quality"], ge=0)
    price_momentum: float = Field(default=SCENARIO_FACTOR_DEFAULTS["price_momentum"], ge=0)
    fundamental_momentum: float = Field(default=SCENARIO_FACTOR_DEFAULTS["fundamental_momentum"], ge=0)
    valuation: float = Field(default=SCENARIO_FACTOR_DEFAULTS["valuation"], ge=0)
    qualitative: float = Field(default=SCENARIO_FACTOR_DEFAULTS["qualitative"], ge=0)

    @model_validator(mode="after")
    def require_nonzero(self):
        legacy_total = self.quality + self.price_momentum + self.fundamental_momentum + self.valuation
        if "qualitative" not in self.model_fields_set and legacy_total <= 0:
            raise ValueError("factor_weights must include at least one positive weight.")
        if legacy_total + self.qualitative <= 0:
            raise ValueError("factor_weights must include at least one positive weight.")
        return self


class AnalyzerFundamentalMomentumWeights(BaseModel):
    revenue: float = Field(default=SCENARIO_FUNDAMENTAL_DEFAULTS["revenue"], ge=0)
    eps: float = Field(default=SCENARIO_FUNDAMENTAL_DEFAULTS["eps"], ge=0)

    @model_validator(mode="after")
    def require_nonzero(self):
        if self.revenue + self.eps <= 0:
            raise ValueError("fundamental_momentum_weights must include at least one positive weight.")
        return self


class AnalyzerValuationWeights(BaseModel):
    price_sales: float = Field(default=SCENARIO_VALUATION_DEFAULTS["price_sales"], ge=0)
    price_operating_income: float = Field(default=SCENARIO_VALUATION_DEFAULTS["price_operating_income"], ge=0)
    price_fcf: float = Field(default=SCENARIO_VALUATION_DEFAULTS["price_fcf"], ge=0)
    price_earnings: float = Field(default=SCENARIO_VALUATION_DEFAULTS["price_earnings"], ge=0)
    price_book: float = Field(default=SCENARIO_VALUATION_DEFAULTS["price_book"], ge=0)

    @model_validator(mode="after")
    def require_nonzero(self):
        total = self.price_sales + self.price_operating_income + self.price_fcf + self.price_earnings + self.price_book
        if total <= 0:
            raise ValueError("valuation_weights must include at least one positive weight.")
        return self


class AnalyzerQualitativeWeights(BaseModel):
    business_quality_qualitative: float = Field(
        default=SCENARIO_QUALITATIVE_DEFAULTS["business_quality_qualitative"], ge=0
    )
    industry_quality: float = Field(default=SCENARIO_QUALITATIVE_DEFAULTS["industry_quality"], ge=0)
    management_quality: float = Field(default=SCENARIO_QUALITATIVE_DEFAULTS["management_quality"], ge=0)

    @model_validator(mode="after")
    def require_nonzero(self):
        total = self.business_quality_qualitative + self.industry_quality + self.management_quality
        if total <= 0:
            raise ValueError("qualitative_weights must include at least one positive weight.")
        return self


class AnalyzerMetricScores(BaseModel):
    quality: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["quality"], ge=0, le=100)
    price_momentum: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["price_momentum"], ge=0, le=100)
    revenue: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["revenue"], ge=0, le=100)
    eps: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["eps"], ge=0, le=100)
    price_sales: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["price_sales"], ge=0, le=100)
    price_operating_income: float = Field(
        default=SCENARIO_METRIC_SCORE_DEFAULTS["price_operating_income"], ge=0, le=100
    )
    price_fcf: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["price_fcf"], ge=0, le=100)
    price_earnings: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["price_earnings"], ge=0, le=100)
    price_book: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["price_book"], ge=0, le=100)
    business_quality_qualitative: float = Field(
        default=SCENARIO_METRIC_SCORE_DEFAULTS["business_quality_qualitative"], ge=0, le=100
    )
    industry_quality: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["industry_quality"], ge=0, le=100)
    management_quality: float = Field(default=SCENARIO_METRIC_SCORE_DEFAULTS["management_quality"], ge=0, le=100)

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
            + self.price_book
            + self.business_quality_qualitative
            + self.industry_quality
            + self.management_quality
        )
        if total <= 0:
            raise ValueError("metric_scores must include at least one positive score.")
        return self


class AnalyzerScenarioBrakes(BaseModel):
    drawdown_sensitivity: float = Field(default=SCENARIO_BRAKE_DEFAULTS["drawdown_sensitivity"], ge=0, le=100)
    contrarian_penalty: float = Field(default=SCENARIO_BRAKE_DEFAULTS["contrarian_penalty"], ge=0, le=100)
    short_squeeze_brake: float = Field(default=SCENARIO_BRAKE_DEFAULTS["short_squeeze_brake"], ge=0, le=100)


class AnalyzerScenario(BaseModel):
    preset: str = "balanced"
    metric_scores: AnalyzerMetricScores | None = None
    factor_weights: AnalyzerFactorWeights = Field(default_factory=AnalyzerFactorWeights)
    fundamental_momentum_weights: AnalyzerFundamentalMomentumWeights = Field(
        default_factory=AnalyzerFundamentalMomentumWeights
    )
    valuation_weights: AnalyzerValuationWeights = Field(default_factory=AnalyzerValuationWeights)
    qualitative_weights: AnalyzerQualitativeWeights = Field(default_factory=AnalyzerQualitativeWeights)
    brakes: AnalyzerScenarioBrakes = Field(default_factory=AnalyzerScenarioBrakes)


class AnalyzerRequest(BaseModel):
    # Legacy optimizer fields are accepted for backward compatibility and ignored by analyzer logic.
    book: float | None = None
    target_leverage: float | None = None
    beta_neutral: bool | None = None
    scenario: AnalyzerScenario | None = None


class AnalyzerBriefRequest(BaseModel):
    action: dict[str, Any]


def _canonical_scenario(req: AnalyzerRequest) -> dict[str, Any]:
    if req.scenario is None:
        return normalize_analyzer_scenario()
    return normalize_analyzer_scenario(req.scenario.model_dump(exclude_unset=True))


def _cache_key(req: AnalyzerRequest) -> str:
    strategy_version = "v4_qualitative_course_of_action"
    scenario = json.dumps(_canonical_scenario(req), sort_keys=True, separators=(",", ":"))
    source_token = {}
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import analyzer_source_cache_token

        source_token = analyzer_source_cache_token()
    except Exception:
        source_token = {"status": "unavailable"}
    source = json.dumps(source_token, sort_keys=True, separators=(",", ":"), default=str)
    return f"portfolio_analyzer:{strategy_version}:scenario={scenario}:source={source}"


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
        from llm_utils import MODEL_MID, call_llm_text, has_llm_api_key

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
            model=MODEL_MID,
            max_tokens=600,
            system="You summarize deterministic portfolio-analysis evidence. You never add facts not present in the input.",
            max_web_search_uses=0,
        )
        return {"brief": text.strip()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate analyzer brief: {e}") from e
