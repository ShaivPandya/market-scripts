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


class AnalyzerScenarioBrakes(BaseModel):
    drawdown_sensitivity: float = Field(default=0.0, ge=0, le=1)
    contrarian_penalty: float = Field(default=0.0, ge=0, le=1)
    short_squeeze_brake: float = Field(default=0.0, ge=0, le=1)


class AnalyzerScenario(BaseModel):
    preset: str = "balanced"
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


def _canonical_scenario(req: AnalyzerRequest) -> dict[str, Any]:
    scenario = req.scenario or AnalyzerScenario()
    raw = scenario.model_dump()
    return {
        "preset": raw.get("preset") or "balanced",
        "factor_weights": _normalize_group(raw["factor_weights"]),
        "fundamental_momentum_weights": _normalize_group(raw["fundamental_momentum_weights"]),
        "valuation_weights": _normalize_group(raw["valuation_weights"]),
        "brakes": {k: round(max(0.0, min(1.0, float(v or 0.0))), 8) for k, v in raw["brakes"].items()},
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
            scenario=req.scenario.model_dump() if req.scenario is not None else None,
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
