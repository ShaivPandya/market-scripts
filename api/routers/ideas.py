"""Investment idea watchlist and evaluator endpoints."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, field_validator

from api.action_execution import stage_api_action
from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.exceptions import NotFoundError, ValidationError

router = APIRouter()

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "auto_report" / "prompts"
IDEA_EVALUATION_VERSION = "v1"
IDEA_ACTIONS = {"buy", "watch", "avoid", "do_nothing"}
CRITICAL_MISSING_SEVERITIES = {"critical", "block"}
RECOMMENDATION_STATUSES = {"clear", "review_required", "blocked", "error"}
SOURCE_QUALITY_VALUES = {"ok", "degraded", "stale", "failed"}
OPTIONAL_JSON_BODY = Body(default=None)


class IdeaCreateRequest(BaseModel):
    ticker: str
    company_name: str | None = None
    user_notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: Literal["watching", "researching", "ready_for_review", "accepted", "rejected", "archived"] = "watching"

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        return ticker


class IdeaUpdateRequest(BaseModel):
    ticker: str | None = None
    company_name: str | None = None
    user_notes: str | None = None
    tags: list[str] | None = None
    status: Literal["watching", "researching", "ready_for_review", "accepted", "rejected", "archived"] | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_optional_ticker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker cannot be empty.")
        return ticker


class IdeaEvaluationRequest(BaseModel):
    idea_id: int
    force_refresh: bool = False


class IdeaAcceptRequest(BaseModel):
    note: str | None = None


class IdeaRejectRequest(BaseModel):
    note: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _safe_text(value: Any, *, max_len: int = 80_000) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "\n\n[truncated]"
    return text


def _normalize_missing_rows(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, str) and row.strip():
            out.append({"field": row.strip(), "severity": "medium", "reason": row.strip()})
        elif isinstance(row, dict):
            field = str(row.get("field") or row.get("name") or "unspecified").strip()
            if not field:
                field = "unspecified"
            severity = str(row.get("severity") or "medium").strip().lower()
            out.append(
                {
                    "field": field,
                    "severity": severity,
                    "reason": str(row.get("reason") or row.get("message") or "").strip(),
                }
            )
    return out


def _has_critical_missing(rows: list[dict[str, Any]]) -> bool:
    return any(str(row.get("severity") or "").lower() in CRITICAL_MISSING_SEVERITIES for row in rows)


def _source_quality_from_missing(rows: list[dict[str, Any]], tool_errors: list[str]) -> dict[str, Any]:
    critical = _has_critical_missing(rows)
    degraded = critical or bool(tool_errors) or bool(rows)
    return {
        "critical_data_quality": "degraded" if degraded else "ok",
        "source_quality": "degraded" if degraded else "ok",
        "quality": "degraded" if degraded else "ok",
        "tool_errors": tool_errors,
        "missing_count": len(rows),
        "critical_missing_count": sum(
            1 for row in rows if str(row.get("severity") or "").lower() in CRITICAL_MISSING_SEVERITIES
        ),
    }


def _recommendation_status(value: Any, *, fallback: str = "clear") -> str:
    normalized = str(value or fallback).strip().lower()
    return normalized if normalized in RECOMMENDATION_STATUSES else fallback


def _quality_value(value: Any, *, fallback: str = "ok") -> str:
    normalized = str(value or fallback).strip().lower()
    return normalized if normalized in SOURCE_QUALITY_VALUES else fallback


def _numeric_or_none(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None:
        numeric = max(minimum, numeric)
    if maximum is not None:
        numeric = min(maximum, numeric)
    return numeric


def _read_state_text(folder: str, ticker: str) -> tuple[str | None, str | None]:
    try:
        from api.state_storage import exists_text, read_text
        from paths import PROJECT_ROOT

        local_path = PROJECT_ROOT / folder / f"{ticker}.md"
        gcs_key = f"live/{'overviews' if folder == 'investment_overviews' else 'theses'}/{ticker}.md"
        if not exists_text(local_path, gcs_key):
            return None, None
        return read_text(local_path, gcs_key, encoding="utf-8"), str(local_path)
    except Exception as exc:
        return None, f"{folder}: {exc}"


def _safe_tool(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        from api.agent_tools import execute_tool
        from ontology.policy import admin_actor

        raw = execute_tool(name, args or {}, actor=admin_actor(source="idea_evaluator"))
        try:
            data = json.loads(raw)
        except Exception:
            data = {"raw": raw}
        return {"ok": True, "data": data}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _build_context(idea: dict[str, Any]) -> dict[str, Any]:
    ticker = str(idea["ticker"]).upper()
    overview, overview_error = _read_state_text("investment_overviews", ticker)
    thesis, thesis_error = _read_state_text("investment_theses", ticker)

    portfolio = _safe_tool("get_portfolio", {"include_hedges": True})
    signal_aggregator = _safe_tool("get_signal_aggregator", {"include_history": False, "lookback_weeks": 156})
    industry_monitor = _safe_tool("get_industry_monitor", {"refresh": False})
    dossier = _safe_tool("get_dossier", {"ticker": ticker})

    tool_errors = [
        f"{label}: {payload.get('error')}"
        for label, payload in {
            "portfolio": portfolio,
            "signal_aggregator": signal_aggregator,
            "industry_monitor": industry_monitor,
            "dossier": dossier,
        }.items()
        if not payload.get("ok")
    ]
    if overview_error:
        tool_errors.append(overview_error)
    if thesis_error:
        tool_errors.append(thesis_error)

    return {
        "idea": idea,
        "ticker": ticker,
        "overview_content": _safe_text(overview),
        "thesis_content": _safe_text(thesis),
        "portfolio": portfolio,
        "signal_aggregator": signal_aggregator,
        "industry_monitor": industry_monitor,
        "dossier": dossier,
        "tool_errors": tool_errors,
        "evaluated_at": _now(),
    }


def _factor(score: float, status: str, rationale: str, missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "score": max(0, min(100, round(float(score), 1))),
        "status": status,
        "rationale": rationale,
        "missing": missing or [],
    }


def _deterministic_evaluation(context: dict[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    idea = context["idea"]
    ticker = context["ticker"]
    notes = str(idea.get("user_notes") or "").strip()
    overview = str(context.get("overview_content") or "").strip()
    thesis = str(context.get("thesis_content") or "").strip()
    tool_errors = list(context.get("tool_errors") or [])

    missing: list[dict[str, Any]] = []
    if not overview:
        missing.append(
            {
                "field": "overview",
                "severity": "critical",
                "reason": "No uploaded business overview is available for the idea.",
            }
        )
    if not notes:
        missing.append(
            {
                "field": "user_notes",
                "severity": "medium",
                "reason": "No user rationale or reason-for-consideration notes are stored.",
            }
        )
    if not thesis:
        missing.append(
            {
                "field": "thesis",
                "severity": "medium",
                "reason": "No full thesis file is available for this non-position idea.",
            }
        )
    if context.get("industry_monitor", {}).get("ok") is not True:
        missing.append(
            {
                "field": "industry_management_commentary",
                "severity": "medium",
                "reason": "Industry monitor context was unavailable or failed.",
            }
        )

    signal_payload = context.get("signal_aggregator", {}).get("data")
    regime = ""
    if isinstance(signal_payload, dict):
        regime = str(signal_payload.get("regime") or signal_payload.get("regime_label") or "").lower()
    macro_score = 60
    macro_status = "mixed"
    if "risk-on" in regime or "risk_on" in regime:
        macro_score = 70
        macro_status = "supportive"
    elif "risk-off" in regime or "risk_off" in regime:
        macro_score = 35
        macro_status = "challenged"

    info_score = 50 + (12 if overview else 0) + (8 if notes else 0) + (6 if thesis else 0)
    factor_scores = {
        "macro_support": _factor(
            macro_score, macro_status, "Derived from the internal signal aggregator when available."
        ),
        "industry_attractiveness": _factor(
            52 if context.get("industry_monitor", {}).get("ok") else 45,
            "mixed",
            "Industry evidence requires manual review of monitor context and uploaded materials.",
        ),
        "business_quality": _factor(
            min(info_score, 72),
            "incomplete" if not overview else "reviewable",
            "Business quality is not fully scoreable without a complete overview and thesis.",
        ),
        "management_quality": _factor(
            45 if not overview else 55,
            "incomplete",
            "Management quality requires explicit track record, capital allocation, and transcript evidence.",
        ),
        "valuation_asymmetry": _factor(
            45,
            "incomplete",
            "No dedicated valuation/asymmetry evidence was computed in deterministic fallback mode.",
            ["valuation", "expected upside/downside"],
        ),
        "portfolio_fit": _factor(
            55,
            "reviewable",
            "Portfolio fit uses current holdings context when available; concentration still needs human review.",
        ),
    }
    score = round(sum(float(row["score"]) for row in factor_scores.values()) / len(factor_scores), 1)
    action = "watch"
    if _has_critical_missing(missing):
        action = "watch"
    elif score >= 72:
        action = "buy"
    elif score <= 40:
        action = "avoid"

    rationale_parts = [
        f"{ticker} is not investment-ready until the missing evidence is filled."
        if action == "watch"
        else f"{ticker} has enough stored evidence for review.",
    ]
    if reason:
        rationale_parts.append(f"Evaluator fallback reason: {reason}")
    if tool_errors:
        rationale_parts.append(
            "Some live/internal evidence sources failed, so the result is deliberately conservative."
        )

    data_quality = _source_quality_from_missing(missing, tool_errors)
    result = {
        "idea_id": idea["id"],
        "ticker": ticker,
        "evaluated_at": context["evaluated_at"],
        "action": action,
        "recommendation_status": "review_required" if data_quality["critical_data_quality"] != "ok" else "clear",
        "score": score,
        "confidence": 0.35 if missing else 0.55,
        "thesis_statement": f"{ticker} may be worth monitoring, but the evidence set is incomplete.",
        "rationale": " ".join(rationale_parts),
        "factor_scores": factor_scores,
        "missing_information": missing,
        "data_quality": data_quality,
        "evidence": [
            {"source": "user_notes", "summary": notes[:500]}
            if notes
            else {"source": "watchlist", "summary": "Idea exists."},
        ],
        "disconfirming_evidence": [
            {
                "source": "data_gap",
                "summary": "Critical overview, valuation, or management evidence may be missing.",
            }
        ],
        "catalyst": "Define a reason-now catalyst before treating this as actionable.",
        "invalidation": "Do not act if the missing overview, valuation, or management evidence cannot support the thesis.",
        "portfolio_fit": {"status": "needs_review", "note": "No position change is staged by evaluation."},
    }
    result["recommendation_record"] = _recommendation_record_from_result(idea, result)
    return result


def _normalize_llm_result(context: dict[str, Any], parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return _deterministic_evaluation(context, reason="model did not return JSON")

    idea = context["idea"]
    ticker = context["ticker"]
    action = str(parsed.get("action") or "watch").strip().lower()
    if action not in IDEA_ACTIONS:
        action = "watch"
    missing = _normalize_missing_rows(parsed.get("missing_information"))
    tool_errors = list(context.get("tool_errors") or [])
    data_quality_raw = parsed.get("data_quality")
    data_quality: dict[str, Any] = cast(dict[str, Any], data_quality_raw) if isinstance(data_quality_raw, dict) else {}
    data_quality = {**_source_quality_from_missing(missing, tool_errors), **data_quality}
    data_quality["critical_data_quality"] = _quality_value(
        data_quality.get("critical_data_quality"), fallback="degraded" if missing else "ok"
    )
    data_quality["source_quality"] = _quality_value(
        data_quality.get("source_quality") or data_quality.get("quality"), fallback="degraded" if missing else "ok"
    )
    data_quality["quality"] = data_quality["source_quality"]
    if _has_critical_missing(missing) and action == "buy":
        action = "watch"

    factor_scores_raw = parsed.get("factor_scores")
    factor_scores: dict[str, Any] = (
        cast(dict[str, Any], factor_scores_raw) if isinstance(factor_scores_raw, dict) else {}
    )
    score = _numeric_or_none(parsed.get("score"), minimum=0, maximum=100)
    confidence = _numeric_or_none(parsed.get("confidence"), minimum=0, maximum=1)
    result: dict[str, Any] = {
        "idea_id": idea["id"],
        "ticker": ticker,
        "evaluated_at": str(parsed.get("evaluated_at") or context["evaluated_at"]),
        "action": action,
        "recommendation_status": _recommendation_status(
            parsed.get("recommendation_status"),
            fallback="review_required" if _has_critical_missing(missing) else "clear",
        ),
        "score": score,
        "confidence": confidence,
        "thesis_statement": str(parsed.get("thesis_statement") or f"{ticker} idea evaluation"),
        "rationale": str(parsed.get("rationale") or ""),
        "factor_scores": factor_scores,
        "missing_information": missing,
        "data_quality": data_quality,
        "evidence": parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else [],
        "disconfirming_evidence": (
            parsed.get("disconfirming_evidence") if isinstance(parsed.get("disconfirming_evidence"), list) else []
        ),
        "catalyst": parsed.get("catalyst"),
        "invalidation": parsed.get("invalidation"),
        "portfolio_fit": parsed.get("portfolio_fit") if isinstance(parsed.get("portfolio_fit"), dict) else {},
    }
    if result["score"] is None and factor_scores:
        scores = []
        for row in factor_scores.values():
            if not isinstance(row, dict):
                continue
            row_score = row.get("score")
            if isinstance(row_score, int | float):
                scores.append(float(row_score))
        result["score"] = round(sum(scores) / len(scores), 1) if scores else None
    if not result["rationale"]:
        result["rationale"] = "No rationale returned; review the factor scores and missing information before acting."
    if not result["disconfirming_evidence"]:
        result["disconfirming_evidence"] = [
            {"source": "required_review", "summary": "No explicit disconfirming evidence was returned."}
        ]
    if not result["invalidation"]:
        result["invalidation"] = "Reject the idea if the missing evidence cannot support the simple thesis."
    result["recommendation_record"] = _recommendation_record_from_result(idea, result)
    return result


def _call_llm_evaluator(context: dict[str, Any]) -> dict[str, Any]:
    from llm_utils import MODEL_HIGH, call_llm_text, has_llm_api_key, parse_json_text

    if not has_llm_api_key():
        return _deterministic_evaluation(context, reason="no configured LLM API key")

    system = "\n\n---\n\n".join(
        [
            _read_prompt("system.md"),
            _read_prompt("agent_system.md"),
            _read_prompt("recommendations_system.md"),
            (
                "You are evaluating independent watchlist ideas. Return only valid JSON. "
                "Do not invent missing evidence. If critical evidence is missing, action must be watch, avoid, or do_nothing."
            ),
        ]
    )
    prompt = (
        "Evaluate the investment idea below against the investment philosophy and recommendation contract. "
        "Use current web/news search only to fill high-level current context; cite sources inside evidence items when used. "
        "Return JSON with keys: thesis_statement, action, recommendation_status, score, confidence, rationale, "
        "factor_scores, missing_information, data_quality, evidence, disconfirming_evidence, catalyst, invalidation, portfolio_fit. "
        "factor_scores must include macro_support, industry_attractiveness, business_quality, management_quality, "
        "valuation_asymmetry, and portfolio_fit. action must be one of buy, watch, avoid, do_nothing.\n\n"
        f"Context JSON:\n{json.dumps(context, default=str, sort_keys=True)}"
    )
    try:
        text, citations, _response = call_llm_text(
            prompt=prompt,
            model=MODEL_HIGH,
            max_tokens=3000,
            system=system,
            max_web_search_uses=4,
        )
        parsed = parse_json_text(text)
        result = _normalize_llm_result(context, parsed)
        if citations:
            evidence = result.setdefault("evidence", [])
            if isinstance(evidence, list):
                evidence.extend(
                    {"source": title, "url": url, "summary": "Live web source used by evaluator."}
                    for title, url in citations[:8]
                )
        return result
    except Exception as exc:
        return _deterministic_evaluation(context, reason=str(exc))


def _recommendation_record_from_result(idea: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    action = str(result.get("action") or "watch").lower()
    now = str(result.get("evaluated_at") or _now())
    ticker = str(idea.get("ticker") or result.get("ticker") or "").upper()
    data_quality_raw = result.get("data_quality")
    data_quality: dict[str, Any] = cast(dict[str, Any], data_quality_raw) if isinstance(data_quality_raw, dict) else {}
    missing_raw = result.get("missing_information")
    missing: list[Any] = missing_raw if isinstance(missing_raw, list) else []
    blocked_reasons = [
        str(row.get("reason") or row.get("field") or "Missing evidence")
        for row in missing
        if isinstance(row, dict) and str(row.get("severity") or "").lower() in CRITICAL_MISSING_SEVERITIES
    ]
    record = {
        "report_type": "daily",
        "as_of": now,
        "created_at": now,
        "source_report_path": "idea_watchlist",
        "source_json_path": f"idea:{idea.get('id')}",
        "stance": action,
        "recommendation_status": _recommendation_status(
            result.get("recommendation_status"),
            fallback="review_required" if blocked_reasons else "clear",
        ),
        "critical_data_quality": _quality_value(data_quality.get("critical_data_quality")),
        "blocked_reasons": blocked_reasons,
        "what_changed": ["Idea evaluator run accepted by user."],
        "do_nothing_rationale": result.get("rationale") if action == "do_nothing" else "",
        "action": action,
        "ticker": ticker,
        "instrument": ticker,
        "horizon": "18-24 months",
        "target_change": "Initial one-third entry review" if action == "buy" else None,
        "rationale": str(result.get("rationale") or ""),
        "confidence": result.get("confidence"),
        "source_quality": _quality_value(data_quality.get("source_quality") or data_quality.get("quality")),
        "evidence": result.get("evidence") if isinstance(result.get("evidence"), list) else [],
        "disconfirming_evidence": (
            result.get("disconfirming_evidence") if isinstance(result.get("disconfirming_evidence"), list) else []
        ),
        "catalyst": result.get("catalyst"),
        "invalidation": result.get("invalidation"),
        "expected_onset_window": "18-24 months",
        "alternatives": [],
        "opportunity_cost": [],
        "source_quality_summary": data_quality,
        "validation_status": "idea_evaluator_v1",
        "idempotency_key": f"idea:{idea.get('id')}:evaluation:{result.get('evaluated_at')}:action:{action}",
        "source_type": "idea_evaluator",
        "report_id": f"idea-evaluation-{idea.get('id')}-{_stable_hash(result)}",
    }
    try:
        from portfolio.policy_gate import attach_policy_gate_to_recommendation

        record, _gate = attach_policy_gate_to_recommendation(
            record,
            source_quality=data_quality,
            context={"source_type": "idea_evaluator", "source_id": str(idea.get("id"))},
        )
    except Exception:
        pass
    return record


def _cache_key(req: IdeaEvaluationRequest) -> str:
    from portfolio import core_db

    idea = core_db.get_investment_idea(req.idea_id)
    if not idea:
        return f"idea_evaluation:{IDEA_EVALUATION_VERSION}:missing:{req.idea_id}"
    token = {
        "id": idea.get("id"),
        "ticker": idea.get("ticker"),
        "company_name": idea.get("company_name"),
        "user_notes": idea.get("user_notes"),
        "tags": idea.get("tags"),
        "metadata": idea.get("metadata"),
        "version": IDEA_EVALUATION_VERSION,
    }
    return f"idea_evaluation:{IDEA_EVALUATION_VERSION}:{_stable_hash(token)}"


def _compute_idea_evaluation_result(
    req: IdeaEvaluationRequest,
    *,
    job_id: str | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    from portfolio import core_db

    idea = core_db.get_investment_idea(req.idea_id)
    if not idea:
        raise RuntimeError(f"No investment idea with id {req.idea_id}")
    if callable(progress_callback):
        progress_callback("context", 1, 4)
    context = _build_context(idea)
    if callable(progress_callback):
        progress_callback("evaluation", 2, 4)
    result = _call_llm_evaluator(context)
    result["job_id"] = job_id
    if callable(progress_callback):
        progress_callback("persisting", 3, 4)
    evaluation = core_db.create_idea_evaluation(req.idea_id, result, job_id=job_id)
    idea = core_db.get_investment_idea(req.idea_id)
    return {"idea": idea, "evaluation": evaluation, "result": result, "final_count": 4}


def _idea_detail(idea_id: int) -> dict[str, Any]:
    from portfolio import core_db

    idea = core_db.get_investment_idea(idea_id)
    if not idea:
        raise NotFoundError("Investment idea", str(idea_id))
    overview, overview_error = _read_state_text("investment_overviews", str(idea["ticker"]).upper())
    thesis, thesis_error = _read_state_text("investment_theses", str(idea["ticker"]).upper())
    return {
        "idea": idea,
        "evaluations": core_db.get_idea_evaluations(idea_id, limit=20),
        "documents": {
            "overview_present": bool(overview),
            "overview_content": _safe_text(overview, max_len=120_000),
            "overview_error": overview_error,
            "thesis_present": bool(thesis),
            "thesis_content": _safe_text(thesis, max_len=120_000),
            "thesis_error": thesis_error,
        },
    }


@router.get("/ideas")
def list_ideas(status: str | None = None, include_archived: bool = False, limit: int = 200):
    from portfolio import core_db

    ideas = core_db.list_investment_ideas(status=status, include_archived=include_archived, limit=limit)
    for idea in ideas:
        evaluation_id = idea.get("latest_evaluation_id")
        idea["latest_evaluation"] = core_db.get_idea_evaluation(int(evaluation_id)) if evaluation_id else None
    return {"ideas": ideas, "count": len(ideas)}


@router.post("/ideas")
def create_idea(body: IdeaCreateRequest):
    from portfolio import core_db

    try:
        idea = core_db.create_investment_idea(**body.model_dump(), source_type="user", source_id="ideas.create")
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return _idea_detail(int(idea["id"]))


@router.get("/ideas/{idea_id}")
def get_idea(idea_id: int):
    return _idea_detail(idea_id)


@router.put("/ideas/{idea_id}")
def update_idea(idea_id: int, body: IdeaUpdateRequest):
    from portfolio import core_db

    updates = body.model_dump(exclude_unset=True)
    try:
        core_db.update_investment_idea(idea_id, **updates)
    except ValueError as exc:
        if str(exc).startswith("No investment idea"):
            raise NotFoundError("Investment idea", str(idea_id)) from exc
        raise ValidationError(str(exc)) from exc
    return _idea_detail(idea_id)


@router.delete("/ideas/{idea_id}")
def delete_idea(idea_id: int):
    from portfolio import core_db

    try:
        idea = core_db.archive_investment_idea(idea_id)
    except ValueError as exc:
        raise NotFoundError("Investment idea", str(idea_id)) from exc
    return {"status": "archived", "idea": idea}


@router.post("/ideas/{idea_id}/evaluate/async")
def start_idea_evaluation(idea_id: int, body: dict[str, Any] | None = OPTIONAL_JSON_BODY):
    from portfolio import core_db

    if not core_db.get_investment_idea(idea_id):
        raise NotFoundError("Investment idea", str(idea_id))
    force_refresh = bool((body or {}).get("force_refresh", False))
    req = IdeaEvaluationRequest(idea_id=idea_id, force_refresh=force_refresh)
    row, _disposition = enqueue_registered_job(
        "idea_evaluation",
        req.model_dump(),
        cache_key=None if force_refresh else _cache_key(req),
        reuse_completed=not force_refresh,
    )
    status = str(row.get("status") or "")
    if status in {"queued", "running"}:
        core_db.update_investment_idea(idea_id, status="researching", latest_job_id=str(row.get("job_id") or ""))
    else:
        core_db.update_investment_idea(idea_id, latest_job_id=str(row.get("job_id") or ""))
    return enqueue_response(row, "/api/v1/ideas/evaluate/async/{job_id}")


@router.get("/ideas/evaluate/async/{job_id}")
def get_idea_evaluation_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id") from None


@router.post("/ideas/{idea_id}/evaluations/{evaluation_id}/accept")
def accept_idea_evaluation(idea_id: int, evaluation_id: int, body: IdeaAcceptRequest | None = None):
    from portfolio import core_db

    idea = core_db.get_investment_idea(idea_id)
    evaluation = core_db.get_idea_evaluation(evaluation_id)
    if not idea:
        raise NotFoundError("Investment idea", str(idea_id))
    if not evaluation or int(evaluation.get("idea_id") or 0) != int(idea_id):
        raise NotFoundError("Idea evaluation", str(evaluation_id))

    record = (
        evaluation.get("recommendation_record") if isinstance(evaluation.get("recommendation_record"), dict) else {}
    )
    if not record:
        record = _recommendation_record_from_result(idea, evaluation)
    recommendation = core_db.upsert_recommendation(record)

    action_proposal = None
    action_error = None
    action = str(evaluation.get("action") or "").lower()
    if action in {"buy", "watch"}:
        action_type = "enter" if action == "buy" else "research"
        description = (
            f"{'Evaluate initial entry' if action == 'buy' else 'Research remaining evidence'} for "
            f"{idea['ticker']} from idea evaluator recommendation #{recommendation['id']}."
        )
        missing = (
            evaluation.get("missing_information") if isinstance(evaluation.get("missing_information"), list) else []
        )
        if missing:
            description += " Missing evidence: " + "; ".join(
                str(row.get("field") or row) for row in missing[:4] if isinstance(row, dict) or row
            )
        try:
            action_proposal = stage_api_action(
                "create_action_item",
                {
                    "recommendation_id": recommendation["id"],
                    "ticker": idea.get("ticker"),
                    "description": description,
                    "action_type": action_type,
                    "urgency": "normal" if action == "buy" else "low",
                },
                source_id=f"ideas.accept:{idea_id}:{evaluation_id}",
                reason=(body.note if body else None)
                or f"Accept idea evaluator recommendation for {idea.get('ticker')}",
            )
            core_db.update_recommendation_approval(
                int(recommendation["id"]),
                int(action_proposal["approval_id"]),
                "pending",
            )
            recommendation = core_db.get_recommendation(int(recommendation["id"])) or recommendation
        except Exception as exc:
            action_error = str(exc)

    accepted = core_db.mark_idea_evaluation_accepted(
        evaluation_id,
        recommendation_id=int(recommendation["id"]),
        action_approval_id=int(action_proposal["approval_id"]) if action_proposal else None,
        accepted_by="user",
    )
    return {
        "status": "accepted",
        "idea": core_db.get_investment_idea(idea_id),
        "evaluation": accepted,
        "recommendation": recommendation,
        "action_proposal": action_proposal,
        "action_error": action_error,
    }


@router.post("/ideas/{idea_id}/reject")
def reject_idea(idea_id: int, body: IdeaRejectRequest | None = None):
    from portfolio import core_db

    try:
        idea = core_db.update_investment_idea(
            idea_id,
            status="rejected",
            metadata={"rejection_note": body.note if body else None, "rejected_at": _now()},
        )
    except ValueError as exc:
        raise NotFoundError("Investment idea", str(idea_id)) from exc
    return {"status": "rejected", "idea": idea}
