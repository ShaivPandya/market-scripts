from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, cast

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.cache import get_or_set_cached, short_cache
from api.exceptions import DataFetchError, SnapshotUnavailableError
from api.job_queue import cancel_job, get_job
from api.job_registry import get_job_spec
from api.serializers import serialize_dataframe, serialize_value
from api.snapshot_keys import SNAPSHOT_SIGNAL_AGGREGATOR
from api.snapshot_store import snapshots_required
from portfolio.portfolio_optimizer.analyzer_scenarios import (
    AI_RECOMMENDED_PRESET,
    REMOVED_SCENARIO_PRESETS,
    SCENARIO_BRAKE_DEFAULTS,
    SCENARIO_FACTOR_DEFAULTS,
    SCENARIO_FUNDAMENTAL_DEFAULTS,
    SCENARIO_METRIC_SCORE_DEFAULTS,
    SCENARIO_QUALITATIVE_DEFAULTS,
    SCENARIO_VALUATION_DEFAULTS,
    build_ai_recommended_scenario,
    normalize_analyzer_scenario,
)

router = APIRouter()

_ANALYZER_STRATEGY_VERSION = "v5_qualitative_idea_universe"


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

    @model_validator(mode="after")
    def reject_removed_presets(self):
        if self.preset in REMOVED_SCENARIO_PRESETS:
            raise ValueError(f"Unsupported analyzer preset: {self.preset}")
        return self


class AnalyzerRequest(BaseModel):
    # Legacy optimizer fields are accepted for backward compatibility and ignored by analyzer logic.
    book: float | None = None
    target_leverage: float | None = None
    beta_neutral: bool | None = None
    scenario: AnalyzerScenario | None = None
    universe_mode: Literal["portfolio", "portfolio_plus_ideas"] = "portfolio"


class AnalyzerBriefRequest(BaseModel):
    action: dict[str, Any]


class AnalyzerRecommendedScenarioBriefRequest(BaseModel):
    recommendation: dict[str, Any]


def _canonical_scenario(req: AnalyzerRequest) -> dict[str, Any]:
    if req.scenario is None:
        return normalize_analyzer_scenario()
    return normalize_analyzer_scenario(req.scenario.model_dump(exclude_unset=True))


def _cache_key(req: AnalyzerRequest, *, freshness_bucket: int | None = None) -> str:
    del freshness_bucket
    scenario = json.dumps(_canonical_scenario(req), sort_keys=True, separators=(",", ":"))
    source_token = {}
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import analyzer_source_cache_token

        try:
            source_token = analyzer_source_cache_token(universe_mode=req.universe_mode)
        except TypeError:
            source_token = analyzer_source_cache_token()
    except Exception:
        source_token = {"status": "unavailable"}
    source = json.dumps(source_token, sort_keys=True, separators=(",", ":"), default=str)
    return (
        f"portfolio_analyzer:{_ANALYZER_STRATEGY_VERSION}:"
        f"universe={req.universe_mode}:scenario={scenario}:source={source}"
    )


def _job_cancelled(job_id: str | None) -> bool:
    if not job_id:
        return False
    row = get_job(job_id)
    return bool(row and str(row.get("status") or "") == "cancelled")


def _compute_analyzer_result_uncached(req: AnalyzerRequest, *, job_id: str | None = None) -> dict[str, Any]:
    try:
        from portfolio.portfolio_optimizer.portfolio_analyzer import get_data

        data = get_data(
            book=req.book,
            target_leverage=req.target_leverage,
            beta_neutral=True if req.beta_neutral is None else req.beta_neutral,
            scenario=_canonical_scenario(req),
            universe_mode=req.universe_mode,
            is_cancelled=lambda: _job_cancelled(job_id),
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


def _compute_analyzer_result_cached(req: AnalyzerRequest, *, job_id: str | None = None) -> dict[str, Any]:
    # This is independent from the async-job DB cache. In-memory short_cache is
    # per process/container; cross-process reuse depends on api.cache disk/GCS
    # fallback when those backends are shared or enabled.
    if _job_cancelled(job_id):
        raise RuntimeError("Portfolio analyzer job cancelled")
    return cast(
        dict[str, Any],
        get_or_set_cached(
            short_cache,
            _cache_key(req),
            lambda: _compute_analyzer_result_uncached(req, job_id=job_id),
        ),
    )


def _compute_analyzer_result(req: AnalyzerRequest, *, job_id: str | None = None) -> dict[str, Any]:
    result = _compute_analyzer_result_cached(req, job_id=job_id)
    if _job_cancelled(job_id):
        raise RuntimeError("Portfolio analyzer job cancelled")
    return result


def _snapshot_meta(payload: dict[str, Any]) -> dict[str, Any] | None:
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        return None
    snapshot = meta.get("snapshot")
    return cast(dict[str, Any], snapshot) if isinstance(snapshot, dict) else None


def _build_recommended_scenario_response(signal_payload: dict[str, Any]) -> dict[str, Any]:
    recommendation = build_ai_recommended_scenario(signal_payload)
    raw_regime = signal_payload.get("regime")
    regime = cast(Mapping[str, Any], raw_regime) if isinstance(raw_regime, dict) else {}
    snapshot = _snapshot_meta(signal_payload)
    response: dict[str, Any] = {
        "status": "ok",
        "preset": AI_RECOMMENDED_PRESET,
        **recommendation,
        "regime": {
            "label": regime.get("label"),
            "score": regime.get("score"),
            "confidence": regime.get("confidence"),
            "history_percentile": regime.get("history_percentile"),
        },
        "source": {
            "tool": "signal_aggregator",
            "as_of": signal_payload.get("as_of"),
            "status": signal_payload.get("status"),
            "failed_modules": signal_payload.get("failed_modules") or [],
        },
    }
    if snapshot is not None:
        response["_meta"] = {"snapshot": snapshot}
    return cast(dict[str, Any], serialize_value(response))


@router.get("/portfolio-analyzer/recommended-scenario")
def get_recommended_analyzer_scenario(force_refresh: bool = Query(False)):
    key = f"portfolio_analyzer:{_ANALYZER_STRATEGY_VERSION}:recommended_scenario:v1"

    def loader():
        try:
            if force_refresh:
                from api.signal_aggregator import build_signal_aggregator

                signal_payload = build_signal_aggregator(lookback_weeks=156, include_history=False)
            else:
                from api.signal_snapshot import get_signal_aggregator_snapshot_or_module_response

                signal_payload = get_signal_aggregator_snapshot_or_module_response(
                    lookback_weeks=156,
                    include_raw_modules=False,
                )
                if signal_payload is None:
                    if snapshots_required():
                        raise SnapshotUnavailableError(SNAPSHOT_SIGNAL_AGGREGATOR)
                    from api.signal_aggregator import build_signal_aggregator

                    signal_payload = build_signal_aggregator(lookback_weeks=156, include_history=False)
        except SnapshotUnavailableError:
            raise
        except Exception as exc:
            raise DataFetchError(source="portfolio_analyzer_recommended_scenario", detail=str(exc)) from exc

        if not isinstance(signal_payload, dict):
            raise DataFetchError(source="portfolio_analyzer_recommended_scenario", detail="Invalid signal payload")
        return _build_recommended_scenario_response(signal_payload)

    return get_or_set_cached(short_cache, key, loader, force_refresh=force_refresh)


@router.post("/portfolio-analyzer/recommended-scenario/brief")
def generate_recommended_scenario_brief(req: AnalyzerRecommendedScenarioBriefRequest):
    try:
        from llm_utils import MODEL_MID, call_llm_text, has_llm_api_key

        if not has_llm_api_key():
            raise HTTPException(status_code=424, detail="No configured LLM API key for analyzer briefs.")

        evidence_json = json.dumps(req.recommendation, sort_keys=True, default=str)
        prompt = (
            "Summarize the deterministic AI Recommended Portfolio Analyzer preset below. "
            "The slider values are already fixed by code; do not change, reinterpret, or propose alternative values. "
            "Use 3 short bullets: regime input, score/brake changes, and watch-outs.\n\n"
            f"Evidence:\n{evidence_json}"
        )
        text, _citations, _response = call_llm_text(
            prompt=prompt,
            model=MODEL_MID,
            max_tokens=600,
            system=(
                "You explain deterministic portfolio-analysis settings. You never choose slider values, ticker scores, "
                "or trade actions."
            ),
            max_web_search_uses=0,
        )
        return {"brief": text.strip()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommended scenario brief: {e}") from e


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


@router.post("/portfolio-analyzer/async/{job_id}/cancel")
@router.post("/portfolio-optimizer/async/{job_id}/cancel")
def cancel_analyzer_job(job_id: str):
    row = get_job(job_id)
    if not row or str(row.get("job_type") or "") != "analyzer":
        raise HTTPException(status_code=404, detail="Unknown job_id")
    cancel_job(
        job_id,
        "Portfolio analyzer job cancelled by user",
        result_ttl_seconds=get_job_spec("analyzer").failed_ttl_s,
    )
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
