"""Workspace aggregate API endpoint -- landing page data."""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.decision_state import (
    normalize_action_item,
    normalize_approval,
    normalize_course_of_action,
    normalize_recommendation,
)
from api.llm_settings import get_setting, set_setting
from api.source_health import build_approval_source_health_review, build_workspace_source_health
from ontology.change_summary import ChangeSummaryInputError, build_workspace_change_summary
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()
logger = logging.getLogger("api.workspace")
THESIS_PRESSURE_DISMISSALS_KEY = "workspace.thesis_pressure.dismissals.v1"
MAX_THESIS_PRESSURE_DISMISSALS = 1000


class DismissThesisPressureRequest(BaseModel):
    ticker: str
    pressure_key: str
    note: str | None = None


def _safe_call(fn, *args, **kwargs) -> Any:
    """Call a function, returning None on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _portfolio_tickers(portfolio_data: Any) -> set[str]:
    if not isinstance(portfolio_data, dict):
        return set()
    positions = portfolio_data.get("positions")
    if not isinstance(positions, list):
        return set()
    return {
        str(position.get("ticker") or "").strip().upper()
        for position in positions
        if isinstance(position, dict) and str(position.get("ticker") or "").strip()
    }


def _pressure_field(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _pressure_key(row: dict[str, Any]) -> str:
    payload = {
        "ticker": _pressure_field(row.get("ticker")).upper(),
        "action": _pressure_field(row.get("action")).lower(),
        "risk_flag": _pressure_field(row.get("risk_flag")),
        "confidence": _pressure_field(row.get("confidence")),
        "evaluated_at": _pressure_field(row.get("evaluated_at")),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"{payload['ticker']}:{digest[:24]}"


def _load_pressure_dismissals() -> dict[str, dict[str, Any]]:
    try:
        row = get_setting(THESIS_PRESSURE_DISMISSALS_KEY)
    except Exception:
        return {}
    if not row:
        return {}
    try:
        raw = json.loads(str(row.get("value") or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(value, dict)}


def _save_pressure_dismissals(dismissals: dict[str, dict[str, Any]]) -> None:
    trimmed = dict(
        sorted(
            dismissals.items(),
            key=lambda item: str(item[1].get("dismissed_at") or ""),
            reverse=True,
        )[:MAX_THESIS_PRESSURE_DISMISSALS]
    )
    set_setting(THESIS_PRESSURE_DISMISSALS_KEY, json.dumps(trimmed, separators=(",", ":"), sort_keys=True))


@router.post("/workspace/thesis-pressure/dismiss")
def dismiss_thesis_pressure(body: DismissThesisPressureRequest):
    ticker = str(body.ticker or "").strip().upper()
    pressure_key = str(body.pressure_key or "").strip()
    if not ticker:
        raise HTTPException(status_code=422, detail="ticker is required")
    if not pressure_key:
        raise HTTPException(status_code=422, detail="pressure_key is required")

    dismissals = _load_pressure_dismissals()
    dismissals[pressure_key] = {
        "ticker": ticker,
        "pressure_key": pressure_key,
        "note": str(body.note or "").strip() or None,
        "dismissed_at": datetime.now(UTC).isoformat(),
    }
    _save_pressure_dismissals(dismissals)
    return {"status": "dismissed", "ticker": ticker, "pressure_key": pressure_key}


@router.get("/workspace")
def get_workspace(since: str | None = None):
    """Return workspace landing page data."""
    reads = OntologyRuntimeReadService()

    # Parallel fetch for expensive cached calls
    regime_data = None
    portfolio_data = None

    def _fetch_regime():
        nonlocal regime_data
        try:
            from api.signal_snapshot import get_signal_aggregator_snapshot_or_module_response
            from api.snapshot_store import snapshots_required

            regime_data = get_signal_aggregator_snapshot_or_module_response(
                lookback_weeks=156,
                include_raw_modules=False,
            )
            if regime_data is None and not snapshots_required():
                from api.signal_aggregator import build_signal_aggregator

                regime_data = build_signal_aggregator(include_history=False)
        except Exception:
            regime_data = None

    def _fetch_portfolio():
        nonlocal portfolio_data
        try:
            import json

            from api.agent_tools import execute_tool

            raw = execute_tool("get_portfolio", {})
            portfolio_data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            portfolio_data = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(_fetch_regime)
        pool.submit(_fetch_portfolio)
        pool.shutdown(wait=True)

    # Regime summary
    regime_summary = None
    if isinstance(regime_data, dict):
        regime_val = regime_data.get("regime")
        composite_score = regime_data.get("composite_score")
        signal = regime_data.get("signal")

        # The signal aggregator may return regime as a nested object
        # e.g. {"label": "risk-on", "score": 32.55, ...} instead of a string.
        if isinstance(regime_val, dict):
            if composite_score is None:
                composite_score = regime_val.get("score")
            if signal is None:
                signal = regime_val.get("label")
            regime_val = regime_val.get("label", str(regime_val))

        snapshot_meta = None
        meta = regime_data.get("_meta")
        if isinstance(meta, dict) and isinstance(meta.get("snapshot"), dict):
            snapshot_meta = meta["snapshot"]

        regime_summary = {
            "regime": regime_val,
            "composite_score": composite_score,
            "signal": signal,
            "snapshot": snapshot_meta,
        }

    # Portfolio summary
    portfolio_summary = None
    portfolio_risk = None
    try:
        from api.position_risk import get_latest_portfolio_risk

        portfolio_risk = get_latest_portfolio_risk()
    except Exception:
        portfolio_risk = None
    if isinstance(portfolio_data, dict):
        positions = portfolio_data.get("positions", [])
        portfolio_summary = {
            "position_count": len(positions) if isinstance(positions, list) else 0,
            "total_pnl": portfolio_data.get("total_pnl"),
            "total_pnl_pct": portfolio_data.get("total_pnl_pct"),
            "risk": {
                "result_id": portfolio_risk.get("result_id"),
                "as_of": portfolio_risk.get("as_of"),
                "computed_at": portfolio_risk.get("computed_at"),
                "quality": portfolio_risk.get("quality"),
                "confidence": portfolio_risk.get("confidence"),
                "average_risk_score": portfolio_risk.get("average_risk_score"),
                "max_risk_score": portfolio_risk.get("max_risk_score"),
                "risk_level": portfolio_risk.get("risk_level"),
                "risk_buckets": portfolio_risk.get("risk_buckets"),
                "top_contributors": portfolio_risk.get("top_contributors", [])[:5]
                if isinstance(portfolio_risk.get("top_contributors"), list)
                else [],
            }
            if isinstance(portfolio_risk, dict)
            else None,
        }
    source_health = _safe_call(
        build_workspace_source_health,
        portfolio_risk=portfolio_risk,
        portfolio_data=portfolio_data if isinstance(portfolio_data, dict) else None,
        regime_data=regime_data if isinstance(regime_data, dict) else None,
    )

    # Positions under thesis pressure
    owned_tickers = _portfolio_tickers(portfolio_data)
    ontology_bundle = reads.workspace_bundle()
    thesis_pressure = []
    try:
        dismissed_pressure_keys = set(_load_pressure_dismissals())
        latest_evals = {
            str(e.get("ticker") or "").strip().upper(): e for e in ontology_bundle.get("latest_evaluations", [])
        }
        for meta in ontology_bundle.get("theses", []):
            ticker = str(meta["ticker"]).strip().upper()
            if ticker not in owned_tickers:
                continue
            ev = latest_evals.get(ticker)
            if not ev:
                continue
            action = (ev.get("action") or "").lower()
            risk_flag = ev.get("risk_flag")
            if action not in ("hold", "") or risk_flag:
                row = {
                    "ticker": ticker,
                    "status": meta.get("status"),
                    "action": action,
                    "confidence": ev.get("confidence"),
                    "risk_flag": risk_flag,
                    "evaluated_at": ev.get("evaluated_at"),
                }
                row["pressure_key"] = _pressure_key(row)
                if row["pressure_key"] not in dismissed_pressure_keys:
                    thesis_pressure.append(row)
    except Exception:
        pass

    # Pending approvals
    pending_approvals = [
        normalize_approval(
            a,
            source_health_review=build_approval_source_health_review(a, source_health),
        )
        for a in ontology_bundle.get("pending_approvals", [])
    ]
    recommendation_approvals = [
        a
        for a in pending_approvals
        if isinstance(a.get("proposed_change"), dict) and a["proposed_change"].get("recommendation_id") is not None
    ]
    course_of_action_approvals = [
        a
        for a in pending_approvals
        if a.get("entity_type") == "course_of_action"
        or a.get("target_object_type") == "CourseOfAction"
        or (
            isinstance(a.get("proposed_change"), dict)
            and (
                a["proposed_change"].get("course_of_action_id") is not None
                or (
                    isinstance(a["proposed_change"].get("record"), dict)
                    and a["proposed_change"]["record"].get("course_of_action_id") is not None
                )
            )
        )
    ]

    latest_daily_recommendation = normalize_recommendation(ontology_bundle.get("latest_daily_recommendation"))
    latest_weekly_recommendation = normalize_recommendation(ontology_bundle.get("latest_weekly_recommendation"))
    pending_actionable_recommendations = ontology_bundle.get("pending_actionable_recommendations", [])
    pending_actionable_recommendations = [
        normalize_recommendation(rec) for rec in pending_actionable_recommendations if isinstance(rec, dict)
    ]
    pending_course_of_actions = [
        item
        for item in (
            normalize_course_of_action(coa)
            for coa in ontology_bundle.get("pending_course_of_actions", [])
            if isinstance(coa, dict)
        )
        if item is not None
    ]
    recent_course_of_actions = [
        item
        for item in (
            normalize_course_of_action(coa)
            for coa in ontology_bundle.get("recent_course_of_actions", [])
            if isinstance(coa, dict)
        )
        if item is not None
    ]
    open_course_of_action_comparisons = ontology_bundle.get("open_course_of_action_comparisons", [])
    blocked_recommendation_warnings = []
    for rec in (latest_daily_recommendation, latest_weekly_recommendation):
        if not isinstance(rec, dict):
            continue
        if rec.get("recommendation_status") == "blocked" or rec.get("critical_data_quality") in {"stale", "failed"}:
            blocked_recommendation_warnings.append(
                {
                    "report_type": rec.get("report_type"),
                    "as_of": rec.get("as_of"),
                    "critical_data_quality": rec.get("critical_data_quality"),
                    "blocked_reasons": rec.get("blocked_reasons_json", []),
                }
            )

    # Open action items
    open_actions = [normalize_action_item(a) for a in ontology_bundle.get("open_action_items", [])]

    # Continuous optimization alerts
    optimizer_alerts = ontology_bundle.get("optimizer_alerts", [])

    # Active watch triggers
    active_triggers = ontology_bundle.get("active_watch_triggers", [])
    monitor_hits = ontology_bundle.get("recent_monitor_hits", [])

    # Latest workflow run
    recent_runs = ontology_bundle.get("recent_workflow_runs", [])
    recent_report_runs = ontology_bundle.get("recent_report_runs", [])
    challenged_claims = ontology_bundle.get("challenged_claims", []) + ontology_bundle.get("disconfirmed_claims", [])
    try:
        what_changed = build_workspace_change_summary(ontology_bundle, since=since)
    except ChangeSummaryInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "regime": regime_summary,
        "portfolio": portfolio_summary,
        "source_health": source_health,
        "what_changed": what_changed,
        "thesis_pressure": thesis_pressure,
        "pending_approvals": {
            "count": len(pending_approvals),
            "items": pending_approvals[:5],
        },
        "recommendations": {
            "latest_daily": latest_daily_recommendation,
            "latest_weekly": latest_weekly_recommendation,
            "pending_actionable": {
                "count": len(pending_actionable_recommendations),
                "items": pending_actionable_recommendations[:5],
            },
            "blocked_warnings": blocked_recommendation_warnings,
            "pending_approval_count": len(recommendation_approvals),
        },
        "course_of_actions": {
            "pending": {
                "count": len(pending_course_of_actions),
                "items": pending_course_of_actions[:5],
            },
            "recent": {
                "count": len(recent_course_of_actions),
                "items": recent_course_of_actions[:5],
            },
            "comparisons": {
                "count": len(open_course_of_action_comparisons),
                "items": open_course_of_action_comparisons[:5],
            },
            "pending_approval_count": len(course_of_action_approvals),
        },
        "open_actions": {
            "count": len(open_actions),
            "items": open_actions,
        },
        "continuous_optimization": {
            "open_alert_count": len(optimizer_alerts),
            "open_alerts": optimizer_alerts[:5],
        },
        "active_triggers": {
            "count": len(active_triggers),
            "items": active_triggers,
        },
        "monitor_hits": {
            "count": len(monitor_hits),
            "items": monitor_hits,
        },
        "recent_workflow_runs": recent_runs,
        "recent_report_runs": recent_report_runs,
        "thesis_claims": {
            "challenged_count": len(challenged_claims),
            "items": challenged_claims[:5],
        },
    }
