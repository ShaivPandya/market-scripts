"""Workspace aggregate API endpoint -- landing page data."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger("api.workspace")


def _safe_call(fn, *args, **kwargs) -> Any:
    """Call a function, returning None on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


@router.get("/workspace")
def get_workspace():
    """Return workspace landing page data."""
    from portfolio.core_db import (
        get_action_items,
        get_latest_recommendation,
        get_pending_approvals,
        get_recommendations,
        get_watch_triggers,
        get_workflow_runs,
    )

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
    if isinstance(portfolio_data, dict):
        positions = portfolio_data.get("positions", [])
        portfolio_summary = {
            "position_count": len(positions) if isinstance(positions, list) else 0,
            "total_pnl": portfolio_data.get("total_pnl"),
            "total_pnl_pct": portfolio_data.get("total_pnl_pct"),
        }

    # Positions under thesis pressure
    thesis_pressure = []
    try:
        from portfolio.thesis_db import get_all_thesis_meta, get_latest_evaluations

        latest_evals = {e["ticker"]: e for e in get_latest_evaluations()}
        for meta in get_all_thesis_meta():
            ticker = meta["ticker"]
            ev = latest_evals.get(ticker)
            if not ev:
                continue
            action = (ev.get("action") or "").lower()
            risk_flag = ev.get("risk_flag")
            if action not in ("hold", "") or risk_flag:
                thesis_pressure.append(
                    {
                        "ticker": ticker,
                        "status": meta.get("status"),
                        "action": action,
                        "confidence": ev.get("confidence"),
                        "risk_flag": risk_flag,
                        "evaluated_at": ev.get("evaluated_at"),
                    }
                )
    except Exception:
        pass

    # Pending approvals
    pending_approvals = get_pending_approvals(status="pending")
    recommendation_approvals = [
        a
        for a in pending_approvals
        if isinstance(a.get("proposed_change"), dict) and a["proposed_change"].get("recommendation_id") is not None
    ]

    latest_daily_recommendation = _safe_call(get_latest_recommendation, "daily")
    latest_weekly_recommendation = _safe_call(get_latest_recommendation, "weekly")
    pending_actionable_recommendations = (
        _safe_call(
            get_recommendations,
            approval_status="pending",
            limit=5,
        )
        or []
    )
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
    open_actions = get_action_items(status="open")

    # Active watch triggers
    active_triggers = get_watch_triggers(status="active")

    # Latest workflow run
    recent_runs = get_workflow_runs(limit=3)

    return {
        "regime": regime_summary,
        "portfolio": portfolio_summary,
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
        "open_actions": {
            "count": len(open_actions),
            "items": open_actions[:5],
        },
        "active_triggers": {
            "count": len(active_triggers),
            "items": active_triggers[:5],
        },
        "recent_workflow_runs": recent_runs,
    }
