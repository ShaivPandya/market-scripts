"""Shared recommendation contract, quality gates, and persistence helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from decision_quality import (
    ACTIONABLE_ACTIONS as DECISION_ACTIONABLE_ACTIONS,
)
from decision_quality import (
    CANONICAL_ACTIONS,
    apply_decision_quality_gates,
    parse_decision_quality,
)

log = logging.getLogger("auto_report.recommendations")

RECOMMENDATIONS_SEPARATOR = "<!-- RECOMMENDATIONS_JSON -->"

STANCE_OPTIONS = (
    "Aggressively Offensive",
    "Offensive",
    "Neutral / Watchful",
    "Defensive",
    "Aggressively Defensive",
)

ACTION_OPTIONS = CANONICAL_ACTIONS

ACTIONABLE_ACTIONS = set(DECISION_ACTIONABLE_ACTIONS)
NON_APPROVAL_ACTIONS = {"hold", "watch", "research", "avoid", "do_nothing"}
QUALITY_OPTIONS = ("ok", "degraded", "stale", "failed")
RECOMMENDATION_STATUSES = ("clear", "review_required", "blocked", "error")

MAX_RECOMMENDATIONS_EVIDENCE_CHARS = 180_000
MAX_RECOMMENDATIONS_COMMENTARY_CHARS = 24_000
MAX_RECOMMENDATIONS_EXTRA_CONTEXT_CHARS = 32_000
_COMPACT_MARKER_KEY = "_prompt_compaction"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

CRITICAL_SOURCES = {
    "daily": {
        "portfolio_positions",
        "risk_data",
        "sizer_summary",
        "indices",
        "breadth",
        "vix",
        "liquidity",
        "yield_curve",
    },
    "weekly": {
        "indices",
        "breadth",
        "vix",
        "liquidity",
        "yield_curve",
        "economic_growth",
        "labor_market",
        "housing",
        "portfolio_context",
    },
}

MAX_SOURCE_AGE_DAYS = {
    "indices": 7,
    "breadth": 7,
    "vix": 7,
    "liquidity": 14,
    "yield_curve": 7,
    "economic_growth": 14,
    "labor_market": 45,
    "housing": 75,
    "portfolio_positions": 30,
    "portfolio_context": 30,
    "risk_data": 7,
    "sizer_summary": 7,
}


class RecommendationValidationError(ValueError):
    """Raised when an LLM recommendation payload violates the contract."""


def _read_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _quality_rank(value: Any) -> int:
    state = str(value or "ok").strip().lower()
    if state == "ok":
        return 0
    if state == "degraded":
        return 1
    if state == "stale":
        return 2
    if state in {"failed", "missing"}:
        return 3
    return 3


def _quality_from_rank(rank: int) -> str:
    if rank <= 0:
        return "ok"
    if rank == 1:
        return "degraded"
    if rank == 2:
        return "stale"
    return "failed"


def _worst_quality(*values: Any) -> str:
    return _quality_from_rank(max((_quality_rank(value) for value in values), default=0))


def _risk_score(value: dict[str, Any] | None, *, portfolio: bool = False) -> float | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("average_risk_score") if portfolio else value.get("risk_score")
    if raw is None and portfolio:
        raw = value.get("risk_score")
    try:
        return None if raw is None else float(raw)
    except (TypeError, ValueError):
        return None


def _compact_risk_snapshot(snapshot: dict[str, Any] | None, *, portfolio: bool = False) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    top_contributors_raw = snapshot.get("top_contributors")
    top_contributors = top_contributors_raw[:5] if isinstance(top_contributors_raw, list) else []
    results_raw = snapshot.get("results")
    results = results_raw[:50] if isinstance(results_raw, list) else []
    out = {
        "result_id": snapshot.get("result_id"),
        "as_of": snapshot.get("as_of"),
        "computed_at": snapshot.get("computed_at"),
        "quality": snapshot.get("quality"),
        "confidence": snapshot.get("confidence"),
        "risk_score": _risk_score(snapshot, portfolio=portfolio),
        "risk_level": snapshot.get("risk_level"),
    }
    if portfolio:
        out.update(
            {
                "position_count": snapshot.get("position_count"),
                "average_risk_score": snapshot.get("average_risk_score"),
                "max_risk_score": snapshot.get("max_risk_score"),
                "risk_buckets": snapshot.get("risk_buckets"),
                "top_contributors": top_contributors,
                "positions": [
                    {
                        "ticker": row.get("ticker"),
                        "risk_score": row.get("risk_score"),
                        "risk_level": row.get("risk_level"),
                        "quality": row.get("quality"),
                        "confidence": row.get("confidence"),
                        "risk_snapshot_id": row.get("risk_snapshot_id"),
                    }
                    for row in results
                    if isinstance(row, dict)
                ],
            }
        )
    else:
        out.update(
            {
                "ticker": snapshot.get("ticker"),
                "sector": snapshot.get("sector"),
                "component_scores": snapshot.get("component_scores"),
                "degraded_modules": snapshot.get("degraded_modules", [])[:5]
                if isinstance(snapshot.get("degraded_modules"), list)
                else [],
            }
        )
    return out


def _latest_first_class_risk_context(ticker: str | None = None) -> dict[str, Any]:
    try:
        from api.position_risk import get_latest_portfolio_risk, get_latest_position_risk
    except Exception:
        return {}

    ticker_norm = str(ticker or "").strip().upper()
    position = get_latest_position_risk(ticker_norm) if ticker_norm else None
    portfolio = get_latest_portfolio_risk()
    return {
        "position": position,
        "portfolio": portfolio,
        "position_compact": _compact_risk_snapshot(position),
        "portfolio_compact": _compact_risk_snapshot(portfolio, portfolio=True),
    }


def _first_class_risk_context_for_prompt() -> dict[str, Any]:
    context = _latest_first_class_risk_context()
    portfolio = context.get("portfolio_compact")
    if not portfolio:
        return {}
    return {"portfolio": portfolio}


def _actionable(action: Any) -> bool:
    return str(action or "").strip().lower() in ACTIONABLE_ACTIONS


def _risk_gate_enabled() -> bool:
    try:
        from api.position_risk import risk_recommendation_gate_enabled
    except Exception:
        return False
    return risk_recommendation_gate_enabled()


def _compat_projections_enabled() -> bool:
    try:
        from api.position_risk import risk_compat_projections_enabled
    except Exception:
        return True
    return risk_compat_projections_enabled()


def _attach_first_class_risk(record: dict[str, Any]) -> dict[str, Any]:
    context = _latest_first_class_risk_context(record.get("ticker"))
    position = context.get("position") if isinstance(context.get("position"), dict) else None
    portfolio = context.get("portfolio") if isinstance(context.get("portfolio"), dict) else None
    if not position and not portfolio:
        if _risk_gate_enabled() and _actionable(record.get("action")):
            record["risk_quality"] = "missing"
            record["critical_data_quality"] = _worst_quality(record.get("critical_data_quality"), "failed")
        return record

    pos_score = _risk_score(position)
    portfolio_score = _risk_score(portfolio, portfolio=True)
    risk_quality = _worst_quality(
        position.get("quality") if position else "ok",
        portfolio.get("quality") if portfolio else "ok",
    )
    confidences = [
        float(value)
        for value in (
            position.get("confidence") if position else None,
            portfolio.get("confidence") if portfolio else None,
        )
        if isinstance(value, (int, float))
    ]
    risk_bindings = {
        "position": context.get("position_compact"),
        "portfolio": context.get("portfolio_compact"),
        "risk_score": pos_score if pos_score is not None else portfolio_score,
        "portfolio_risk_score": portfolio_score,
        "position_risk_score": pos_score,
    }
    record.update(
        {
            "risk_snapshot_id": position.get("result_id") if position else None,
            "portfolio_risk_snapshot_id": portfolio.get("result_id") if portfolio else None,
            "risk_quality": risk_quality,
            "risk_confidence": round(min(confidences), 2) if confidences else None,
            "risk_score": pos_score if pos_score is not None else portfolio_score,
            "risk_level": position.get("risk_level")
            if position
            else portfolio.get("risk_level")
            if portfolio
            else None,
            "risk_source_status": {
                "position": position.get("source_status") if position else None,
                "portfolio": portfolio.get("source_status") if portfolio else None,
            },
            "risk_bindings": risk_bindings,
        }
    )
    if _compat_projections_enabled():
        record["critical_data_quality"] = _worst_quality(record.get("critical_data_quality"), risk_quality)
        record["source_quality"] = _worst_quality(record.get("source_quality"), risk_quality)
    return record


def _strip_json_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, out))


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 120:
        return value[:max_chars]
    marker = f"\n\n... [truncated {len(value) - max_chars} chars for recommendations prompt] ...\n\n"
    head_len = max((max_chars - len(marker)) // 2, 1)
    tail_len = max(max_chars - len(marker) - head_len, 1)
    return (value[:head_len].rstrip() + marker + value[-tail_len:].lstrip())[:max_chars]


def _is_probably_time_series(rows: list[Any]) -> bool:
    if len(rows) < 8:
        return False
    sample = [row for row in rows[:8] if isinstance(row, dict)]
    if len(sample) < 4:
        return False
    date_keys = {"date", "as_of", "timestamp", "time", "datetime", "published_at", "fetched_at"}
    return any(date_keys & {str(key).lower() for key in row} for row in sample)


def _compact_prompt_value(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 6,
    list_limit: int = 50,
    dict_limit: int = 100,
    string_limit: int = 6_000,
) -> Any:
    """Return a JSON-safe, prompt-sized copy of recommendation evidence."""
    if isinstance(value, str):
        return _truncate_text(value, string_limit)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= max_depth:
        if isinstance(value, dict):
            return {"_type": "dict", "keys": sorted(str(k) for k in value)[:dict_limit], "truncated_at_depth": depth}
        if isinstance(value, list):
            return {"_type": "list", "length": len(value), "truncated_at_depth": depth}
        return str(value)
    if isinstance(value, list):
        if len(value) <= list_limit:
            return [
                _compact_prompt_value(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    list_limit=list_limit,
                    dict_limit=dict_limit,
                    string_limit=string_limit,
                )
                for item in value
            ]
        if _is_probably_time_series(value):
            kept = value[-list_limit:]
            mode = "latest"
        else:
            head = list_limit // 2
            tail = list_limit - head
            kept = [*value[:head], *value[-tail:]]
            mode = "head_tail"
        return {
            "_type": "list",
            "original_length": len(value),
            "kept": mode,
            "items": [
                _compact_prompt_value(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    list_limit=list_limit,
                    dict_limit=dict_limit,
                    string_limit=string_limit,
                )
                for item in kept
            ],
        }
    if isinstance(value, dict):
        items = list(value.items())
        if len(items) > dict_limit:
            priority = []
            priority_keys = set()
            for key, item in items:
                key_l = str(key).lower()
                if key_l in {
                    "ticker",
                    "instrument",
                    "summary",
                    "latest",
                    "data_quality",
                    "risk_summary",
                    "stance",
                    "portfolio_positions",
                    "risk_data",
                    "sizer_summary",
                    "weekly_summary",
                    "market_data",
                    "sources",
                    "error",
                }:
                    priority.append((key, item))
                    priority_keys.add(key)
            remaining = [(key, item) for key, item in items if key not in priority_keys]
            items = [*priority[:dict_limit], *remaining[: max(0, dict_limit - len(priority))]]
        out = {
            str(key): _compact_prompt_value(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                list_limit=list_limit,
                dict_limit=dict_limit,
                string_limit=string_limit,
            )
            for key, item in items
        }
        if len(value) > dict_limit:
            out["_truncated_keys"] = len(value) - len(items)
        return out
    return str(value)


def _json_for_prompt(value: Any, max_chars: int) -> str:
    tiers = (
        {"max_depth": 6, "list_limit": 50, "dict_limit": 100, "string_limit": 6_000},
        {"max_depth": 5, "list_limit": 30, "dict_limit": 70, "string_limit": 3_000},
        {"max_depth": 4, "list_limit": 15, "dict_limit": 40, "string_limit": 1_200},
    )
    for tier in tiers:
        compacted = _compact_prompt_value(value, **tier)
        text = json.dumps(compacted, indent=2, default=str)
        if len(text) <= max_chars:
            return text
    summary = _compact_prompt_value(value, max_depth=3, list_limit=8, dict_limit=24, string_limit=600)
    text = json.dumps(
        {
            _COMPACT_MARKER_KEY: {
                "reason": "evidence exceeded recommendations prompt budget",
                "max_chars": max_chars,
            },
            "summary": summary,
        },
        indent=2,
        default=str,
    )
    if len(text) <= max_chars:
        return text
    if isinstance(value, dict):
        top_level_keys = sorted(str(key) for key in value)[:80]
    else:
        top_level_keys = []
    return json.dumps(
        {
            _COMPACT_MARKER_KEY: {
                "reason": "evidence exceeded recommendations prompt budget after aggressive compaction",
                "max_chars": max_chars,
            },
            "top_level_keys": top_level_keys,
        },
        indent=2,
        default=str,
    )


def _compact_commentary_context(commentary_md: str) -> str:
    text = commentary_md.strip()
    if "\n## Sources" in text:
        text = text.split("\n## Sources", 1)[0].rstrip()
    return _truncate_text(text, MAX_RECOMMENDATIONS_COMMENTARY_CHARS)


def _compact_extra_context(extra_context_md: str) -> str:
    return _truncate_text(extra_context_md.strip(), MAX_RECOMMENDATIONS_EXTRA_CONTEXT_CHARS)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def _collect_dates(value: Any, *, depth: int = 0) -> list[date]:
    if depth > 5:
        return []
    dates: list[date] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if any(token in key_l for token in ("date", "as_of", "timestamp", "fetched_at", "latest")):
                parsed = _parse_date(item)
                if parsed is not None:
                    dates.append(parsed)
            if isinstance(item, (dict, list)):
                dates.extend(_collect_dates(item, depth=depth + 1))
    elif isinstance(value, list):
        for item in value[:25]:
            dates.extend(_collect_dates(item, depth=depth + 1))
    return dates


def _has_nested_error(value: Any, *, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(value, dict):
        err = value.get("error")
        if isinstance(err, str) and err.strip():
            return True
        return any(_has_nested_error(v, depth=depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(_has_nested_error(v, depth=depth + 1) for v in value[:25])
    return False


def _extract_source_map(raw: dict) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return sources
    for key, value in raw.items():
        if key == "market_data" and isinstance(value, dict):
            sources.update(value)
        else:
            sources[key] = value
    return sources


def assess_report_data_quality(raw: dict, report_type: str) -> dict:
    """Return source-level status plus a critical gate for recommendations."""
    sources = _extract_source_map(raw)
    critical = CRITICAL_SOURCES.get(report_type, set())
    today = datetime.now(UTC).date()
    entries: list[dict[str, Any]] = []
    blocked_reasons: list[str] = []

    for name in sorted(set(sources) | critical):
        value = sources.get(name)
        is_critical = name in critical
        status = "ok"
        error = None
        latest = None
        freshness_days = None

        if value is None or value == [] or value == {}:
            status = "failed" if is_critical else "degraded"
            error = "missing or empty source"
        elif isinstance(value, dict) and isinstance(value.get("error"), str):
            status = "failed"
            error = value.get("error")
        elif _has_nested_error(value):
            status = "degraded"
            error = "nested source error"

        dates = _collect_dates(value)
        if dates:
            latest = max(dates).isoformat()
            freshness_days = (today - max(dates)).days
            max_age = MAX_SOURCE_AGE_DAYS.get(name)
            if max_age is not None and freshness_days > max_age and status != "failed":
                status = "stale"
                error = f"latest observation {freshness_days} days old"

        if is_critical and status in {"failed", "stale"}:
            blocked_reasons.append(f"{name}: {error or status}")

        entries.append(
            {
                "module": name,
                "status": status,
                "as_of": latest,
                "freshness_days": freshness_days,
                "critical": is_critical,
                "error": error,
            }
        )

    critical_entries = [e for e in entries if e["critical"]]
    if any(e["status"] == "failed" for e in critical_entries):
        critical_status = "failed"
    elif any(e["status"] == "stale" for e in critical_entries):
        critical_status = "stale"
    elif any(e["status"] == "degraded" for e in critical_entries):
        critical_status = "degraded"
    else:
        critical_status = "ok"

    if critical_status in {"failed", "stale"}:
        overall = critical_status
    elif any(e["status"] in {"failed", "stale", "degraded"} for e in entries):
        overall = "degraded"
    else:
        overall = "ok"

    return {
        "overall_status": overall,
        "critical_data_quality": critical_status,
        "recommendations_blocked": bool(blocked_reasons),
        "blocked_reasons": blocked_reasons,
        "sources": entries,
    }


def fallback_recommendations_payload(
    report_type: str,
    as_of: str,
    stance: str,
    data_quality: dict,
    *,
    status: str | None = None,
    reason: str | None = None,
) -> dict:
    recommendation_status = status or ("blocked" if data_quality.get("recommendations_blocked") else "error")
    quality = data_quality.get("critical_data_quality", "failed")
    blocked_reasons = list(data_quality.get("blocked_reasons") or [])
    if reason:
        blocked_reasons.append(reason)
    rationale = reason or "Critical inputs are unavailable or the recommendation payload could not be validated."
    return {
        "report_type": report_type,
        "as_of": as_of,
        "stance": stance if stance in STANCE_OPTIONS else "Neutral / Watchful",
        "recommendation_status": recommendation_status,
        "critical_data_quality": quality if quality in QUALITY_OPTIONS else "failed",
        "blocked_reasons": blocked_reasons,
        "do_nothing_rationale": rationale,
        "what_changed": [],
        "recommended_actions": [
            {
                "action": "do_nothing",
                "ticker": None,
                "instrument": "portfolio",
                "horizon": "1 trading day" if report_type == "daily" else "1 week",
                "target_change": "none",
                "rationale": rationale,
                "evidence": blocked_reasons,
                "disconfirming_evidence": [],
                "catalyst": "",
                "invalidation": "Data quality restored and a fresh recommendation pass is valid.",
                "expected_onset_window": "",
                "confidence": 1.0 if recommendation_status == "blocked" else 0.0,
                "source_quality": quality if quality in QUALITY_OPTIONS else "failed",
                "approval_required": False,
            }
        ],
        "alternatives": [],
        "opportunity_cost": [],
    }


def validate_recommendations_payload(
    payload: dict,
    *,
    report_type: str,
    as_of: str,
    stance: str,
    data_quality: dict,
) -> dict:
    if not isinstance(payload, dict):
        raise RecommendationValidationError("Recommendation payload must be a JSON object.")

    required_top_level = {
        "report_type",
        "as_of",
        "stance",
        "recommendation_status",
        "critical_data_quality",
        "blocked_reasons",
        "do_nothing_rationale",
        "what_changed",
        "recommended_actions",
        "alternatives",
        "opportunity_cost",
    }
    errors: list[str] = []
    missing_top_level = sorted(required_top_level - set(payload))
    if missing_top_level:
        errors.append(f"missing required top-level fields: {', '.join(missing_top_level)}")

    normalized = dict(payload)
    normalized["report_type"] = normalized.get("report_type") or report_type
    normalized["as_of"] = normalized.get("as_of") or as_of
    normalized["stance"] = normalized.get("stance") or stance
    normalized["critical_data_quality"] = normalized.get("critical_data_quality") or data_quality.get(
        "critical_data_quality", "ok"
    )
    normalized["recommendation_status"] = normalized.get("recommendation_status") or "clear"

    if normalized["report_type"] not in {"daily", "weekly"}:
        errors.append("report_type must be daily or weekly")
    elif normalized["report_type"] != report_type:
        errors.append(f"report_type must match pipeline report_type {report_type!r}")
    if str(normalized["as_of"]) != str(as_of):
        errors.append(f"as_of must match pipeline as_of {as_of!r}")
    if normalized["stance"] not in STANCE_OPTIONS:
        errors.append(f"stance must be one of {', '.join(STANCE_OPTIONS)}")
    elif normalized["stance"] != stance:
        errors.append(f"stance must match pipeline stance {stance!r}")
    if normalized["recommendation_status"] not in RECOMMENDATION_STATUSES:
        errors.append("recommendation_status must be clear, review_required, blocked, or error")
    if normalized["critical_data_quality"] not in QUALITY_OPTIONS:
        errors.append("critical_data_quality must be ok, degraded, stale, or failed")

    blocked_reasons = _as_list(normalized.get("blocked_reasons"))
    if data_quality.get("recommendations_blocked"):
        normalized["recommendation_status"] = "blocked"
        normalized["critical_data_quality"] = data_quality.get("critical_data_quality", "failed")
        for reason in data_quality.get("blocked_reasons", []):
            if reason not in blocked_reasons:
                blocked_reasons.append(reason)
    normalized["blocked_reasons"] = [str(x) for x in blocked_reasons if str(x).strip()]
    normalized["what_changed"] = [str(x) for x in _as_list(normalized.get("what_changed")) if str(x).strip()]
    normalized["alternatives"] = _as_list(normalized.get("alternatives"))
    normalized["opportunity_cost"] = _as_list(normalized.get("opportunity_cost"))
    normalized["do_nothing_rationale"] = str(normalized.get("do_nothing_rationale") or "")

    actions = normalized.get("recommended_actions")
    if not isinstance(actions, list) or not actions:
        errors.append("recommended_actions must be a non-empty list")
        actions = []

    normalized_actions: list[dict[str, Any]] = []
    required_action_fields = {
        "action",
        "ticker",
        "instrument",
        "horizon",
        "target_change",
        "rationale",
        "evidence",
        "disconfirming_evidence",
        "catalyst",
        "invalidation",
        "expected_onset_window",
        "confidence",
        "source_quality",
        "approval_required",
    }
    for idx, raw_action in enumerate(actions):
        if not isinstance(raw_action, dict):
            errors.append(f"recommended_actions[{idx}] must be an object")
            continue
        missing_action_fields = sorted(required_action_fields - set(raw_action))
        if missing_action_fields:
            errors.append(f"recommended_actions[{idx}] missing required fields: {', '.join(missing_action_fields)}")
        action = str(raw_action.get("action") or "").lower()
        if action == "review":
            action = "watch"
        if action not in ACTION_OPTIONS:
            errors.append(f"recommended_actions[{idx}].action is invalid: {action!r}")
            continue
        decision_quality, dq_errors = parse_decision_quality(raw_action.get("decision_quality"))
        dq_gate = apply_decision_quality_gates(
            decision_quality,
            current_action=action,
            recommendation_status=str(normalized["recommendation_status"]),
            data_quality={
                **data_quality,
                "critical_data_quality": normalized.get("critical_data_quality"),
                "source_quality": raw_action.get("source_quality"),
            },
            parse_errors=dq_errors,
        )
        action = dq_gate.final_action
        if dq_gate.final_recommendation_status == "review_required" and normalized["recommendation_status"] == "clear":
            normalized["recommendation_status"] = "review_required"
        elif dq_gate.final_recommendation_status in {"blocked", "error"}:
            normalized["recommendation_status"] = dq_gate.final_recommendation_status
        if normalized["recommendation_status"] in {"blocked", "error"} and action not in {"watch", "do_nothing"}:
            action = "watch"
        ticker = raw_action.get("ticker")
        if isinstance(ticker, str):
            ticker = ticker.strip().upper() or None
        else:
            ticker = None
        confidence = _as_float(raw_action.get("confidence"), 0.0)
        if dq_gate.confidence_cap is not None:
            confidence = min(confidence, dq_gate.confidence_cap)
        approval_required = bool(raw_action.get("approval_required"))
        if normalized["recommendation_status"] in {"clear", "review_required"} and action in ACTIONABLE_ACTIONS:
            approval_required = True
        else:
            approval_required = False
        normalized_actions.append(
            {
                "action": action,
                "ticker": ticker,
                "instrument": str(raw_action.get("instrument") or ticker or "portfolio"),
                "horizon": str(raw_action.get("horizon") or ("1 trading day" if report_type == "daily" else "1 week")),
                "target_change": str(raw_action.get("target_change") or ""),
                "rationale": str(raw_action.get("rationale") or ""),
                "evidence": [str(x) for x in _as_list(raw_action.get("evidence")) if str(x).strip()],
                "disconfirming_evidence": [
                    str(x) for x in _as_list(raw_action.get("disconfirming_evidence")) if str(x).strip()
                ],
                "catalyst": str(raw_action.get("catalyst") or ""),
                "invalidation": str(raw_action.get("invalidation") or ""),
                "expected_onset_window": str(raw_action.get("expected_onset_window") or ""),
                "confidence": confidence,
                "source_quality": raw_action.get("source_quality")
                if raw_action.get("source_quality") in QUALITY_OPTIONS
                else normalized["critical_data_quality"],
                "approval_required": approval_required,
                "decision_quality": decision_quality.model_dump(mode="json") if decision_quality else None,
                "decision_quality_gate": dq_gate.model_dump(mode="json"),
            }
        )

    if not normalized_actions and not errors:
        errors.append("recommended_actions contains no valid actions")

    if errors:
        raise RecommendationValidationError("; ".join(errors))

    normalized["recommended_actions"] = normalized_actions
    return normalized


def parse_recommendations_response(
    text: str,
    *,
    report_type: str,
    as_of: str,
    stance: str,
    data_quality: dict,
) -> tuple[str, dict]:
    if RECOMMENDATIONS_SEPARATOR not in text:
        raise RecommendationValidationError("No recommendations JSON separator found.")
    md, raw_json = text.split(RECOMMENDATIONS_SEPARATOR, 1)
    try:
        payload = json.loads(_strip_json_fence(raw_json))
    except json.JSONDecodeError as exc:
        raise RecommendationValidationError(f"Invalid recommendations JSON: {exc}") from exc
    return md.strip(), validate_recommendations_payload(
        payload,
        report_type=report_type,
        as_of=as_of,
        stance=stance,
        data_quality=data_quality,
    )


def build_recommendations_user_message(
    *,
    report_type: str,
    as_of: str,
    stance: str,
    data_quality: dict,
    evidence_bundle: dict,
    commentary_md: str,
    extra_context_md: str = "",
) -> str:
    horizon = "today / next 1-5 trading days" if report_type == "daily" else "next week / next 1-3 months"
    enriched_bundle = dict(evidence_bundle)
    first_class_risk = _first_class_risk_context_for_prompt()
    if first_class_risk and "first_class_risk" not in enriched_bundle:
        enriched_bundle["first_class_risk"] = first_class_risk
    bundle_json = _json_for_prompt(enriched_bundle, MAX_RECOMMENDATIONS_EVIDENCE_CHARS)
    quality_json = json.dumps(data_quality, indent=2, default=str)
    commentary_context = _compact_commentary_context(commentary_md)
    extra_context = _compact_extra_context(extra_context_md)
    stance_options = " | ".join(STANCE_OPTIONS)
    action_options = " | ".join(ACTION_OPTIONS)
    decision_quality_contract = _read_prompt("decision_quality.md")
    log.info(
        "Recommendation prompt context prepared (report_type=%s evidence_chars=%d commentary_chars=%d extra_chars=%d)",
        report_type,
        len(bundle_json),
        len(commentary_context),
        len(extra_context),
    )
    return f"""Produce the {report_type} recommendations report for {as_of}.

Current stance: {stance}
Decision horizon: {horizon}

## Data Quality

```json
{quality_json}
```

## Evidence Bundle

```json
{bundle_json}
```

## Commentary Context

{commentary_context}

{extra_context}

Write a short recommendations memo first. The memo must be decision-oriented and may not repeat the commentary.

## Shared Decision Quality Contract

{decision_quality_contract}

Hard rules:
- If critical data quality is stale or failed, set recommendation_status to blocked and use only watch or do_nothing.
- Do not imply trade execution. The deterministic financial policy gate runs after JSON validation and can downgrade clear actions to review_required or blocked.
- Commentary is context only; it cannot by itself justify an action.
- do_nothing is an active recommendation when no fat pitch exists.
- New entries normally start at one-third intended size.
- Adds require validation from price action, news, and/or fundamentals.
- If the expected onset window failed, recommend reduce, exit, or watch.
- Default hedge is position reduction. Hedge overlays require explicit justification.
- Use the shared stance enum exactly: {stance_options}.
- Use only these actions: {action_options}.
- `review` is not an action. Use recommendation_status `review_required` when human review is needed, and use action `watch` for non-directional review/monitoring items.

After the memo, output the separator `{RECOMMENDATIONS_SEPARATOR}` on its own line, then a JSON block matching this contract:
```json
{{
  "report_type": "{report_type}",
  "as_of": "{as_of}",
  "stance": "<{stance_options}>",
  "recommendation_status": "<clear|review_required|blocked|error>",
  "critical_data_quality": "<ok|degraded|stale|failed>",
  "blocked_reasons": [],
  "do_nothing_rationale": "",
  "what_changed": [],
  "recommended_actions": [
    {{
      "action": "<{action_options}>",
      "ticker": null,
      "instrument": "portfolio",
      "horizon": "{"1 trading day" if report_type == "daily" else "1 week"}",
      "target_change": "",
      "rationale": "",
      "evidence": [],
      "disconfirming_evidence": [],
      "catalyst": "",
      "invalidation": "",
      "expected_onset_window": "",
      "confidence": 0.0,
      "source_quality": "<ok|degraded|stale|failed>",
      "approval_required": false,
      "decision_quality": {{}}
    }}
  ],
  "alternatives": [],
  "opportunity_cost": []
}}
```

End immediately after the JSON. No assistant meta text."""


def repair_recommendations_response(
    raw_text: str,
    validation_error: str,
    *,
    report_type: str,
    as_of: str,
    stance: str,
    data_quality: dict,
) -> tuple[str, dict]:
    from auto_report.shared import call_report_llm

    action_options = " | ".join(ACTION_OPTIONS)
    prompt = f"""Repair the recommendations output so it strictly matches the JSON contract.

Validation error:
{validation_error}

Allowed recommended_actions[].action values:
{action_options}

If any recommended_actions[].action is `review`, convert that action to `watch` and preserve its rationale, evidence, and monitoring intent. `review_required` is only a recommendation_status value, not an action.

Original output:
```
{raw_text}
```

Return a concise memo, then `{RECOMMENDATIONS_SEPARATOR}`, then valid JSON only. Do not add meta text."""
    repaired, _ = call_report_llm(
        system_msg="You repair malformed investment recommendation JSON. Do not change the intent unless required by schema or blocked data-quality rules.",
        user_msg=prompt,
        web_search=False,
        max_tokens=8192,
    )
    return parse_recommendations_response(
        repaired,
        report_type=report_type,
        as_of=as_of,
        stance=stance,
        data_quality=data_quality,
    )


def format_recommendations_markdown(payload: dict) -> str:
    lines = [
        "## Recommendation Status",
        f"- Status: **{payload.get('recommendation_status', 'unknown')}**",
        f"- Critical data quality: **{payload.get('critical_data_quality', 'unknown')}**",
        f"- Stance: **{payload.get('stance', 'unknown')}**",
    ]
    blocked = payload.get("blocked_reasons") or []
    if blocked:
        lines.append("- Blocked reasons:")
        lines.extend(f"  - {reason}" for reason in blocked)
    gate = payload.get("policy_gate_result")
    if isinstance(gate, dict):
        lines.append(f"- Policy gate: **{gate.get('decision', 'unknown')}**")
        warnings = gate.get("warnings") or []
        failures = gate.get("failure_reasons") or []
        if failures:
            lines.append(
                "- Policy failures: "
                + "; ".join(str(item.get("message") if isinstance(item, dict) else item) for item in failures)
            )
        if warnings:
            lines.append(
                "- Policy warnings: "
                + "; ".join(str(item.get("message") if isinstance(item, dict) else item) for item in warnings)
            )
    if payload.get("do_nothing_rationale"):
        lines.extend(["", "## Do Nothing Rationale", str(payload["do_nothing_rationale"])])
    if payload.get("what_changed"):
        lines.append("")
        lines.append("## What Changed")
        lines.extend(f"- {item}" for item in payload["what_changed"])
    lines.append("")
    lines.append("## Decision-Support Recommendations")
    for action in payload.get("recommended_actions", []):
        label = action.get("instrument") or action.get("ticker") or "portfolio"
        lines.append("")
        lines.append(f"### {str(action.get('action', '')).replace('_', ' ').title()} - {label}")
        lines.append(f"- Horizon: {action.get('horizon', '')}")
        lines.append(f"- Candidate internal adjustment: {action.get('target_change') or 'none'}")
        lines.append(f"- Confidence: {action.get('confidence', 0):.2f}")
        lines.append(
            f"- Internal approval required before state change: {'yes' if action.get('approval_required') else 'no'}"
        )
        lines.append(f"- Rationale: {action.get('rationale', '')}")
        dq_gate = action.get("decision_quality_gate")
        if isinstance(dq_gate, dict) and dq_gate.get("reasons"):
            reasons = dq_gate.get("reasons") if isinstance(dq_gate.get("reasons"), list) else []
            lines.append(
                "- Decision quality gate: "
                + "; ".join(str(item.get("code") if isinstance(item, dict) else item) for item in reasons if item)
            )
        if action.get("invalidation"):
            lines.append(f"- Invalidation: {action['invalidation']}")
        if action.get("evidence"):
            lines.append("- Evidence: " + "; ".join(action["evidence"]))
        if action.get("disconfirming_evidence"):
            lines.append("- Disconfirming evidence: " + "; ".join(action["disconfirming_evidence"]))
    return "\n".join(lines).strip()


def _approval_action_type(action: str) -> str:
    if action in {"buy", "add", "short", "sell"}:
        return "enter"
    if action == "exit":
        return "exit"
    if action == "hedge":
        return "hedge"
    if action in {"trim", "reduce", "rebalance"}:
        return "resize"
    return "review"


def persist_recommendations(
    payload: dict,
    *,
    source_report_path: str,
    source_json_path: str,
    prompt_metadata: dict | None = None,
) -> list[dict]:
    from ontology.command_service import OntologyCommandContext, OntologyCommandService
    from ontology.object_service import OntologyObjectService
    from ontology.policy import actor_to_dict, system_actor
    from ontology.schemas.identity import policy_gate_result_id
    from portfolio.policy_gate import attach_policy_gate_to_recommendation

    prompt_metadata = prompt_metadata or {}
    report_id = prompt_metadata.get("report_id")
    persisted: list[dict] = []
    actor = system_actor("recommendations")
    context = OntologyCommandContext(
        actor=actor,
        source_type="workflow",
        source_id=report_id or f"{payload['report_type']}:{payload['as_of']}",
    )
    command_service = OntologyCommandService()
    object_service = OntologyObjectService()
    for action in payload.get("recommended_actions", []):
        action_hash = stable_hash(
            {
                "action": action.get("action"),
                "ticker": action.get("ticker"),
                "instrument": action.get("instrument"),
                "horizon": action.get("horizon"),
                "target_change": action.get("target_change"),
                "rationale": action.get("rationale"),
                "evidence": action.get("evidence", []),
                "invalidation": action.get("invalidation"),
            }
        )
        idempotency_key = f"{payload['report_type']}:{payload['as_of']}:{action_hash}" if report_id else None
        record = {
            **action,
            "report_type": payload["report_type"],
            "as_of": payload["as_of"],
            "source_report_path": source_report_path,
            "source_json_path": source_json_path,
            "stance": payload["stance"],
            "recommendation_status": payload["recommendation_status"],
            "critical_data_quality": payload["critical_data_quality"],
            "blocked_reasons": payload.get("blocked_reasons", []),
            "what_changed": payload.get("what_changed", []),
            "do_nothing_rationale": payload.get("do_nothing_rationale", ""),
            "alternatives": payload.get("alternatives", []),
            "opportunity_cost": payload.get("opportunity_cost", []),
            "status": "blocked"
            if payload["recommendation_status"] == "blocked"
            else "error"
            if payload["recommendation_status"] == "error"
            else "open",
            "approval_status": "none",
            "outcome_status": "pending",
            "report_id": report_id,
            "idempotency_key": idempotency_key,
            **prompt_metadata,
        }
        record = _attach_first_class_risk(record)
        record, gate = attach_policy_gate_to_recommendation(
            record,
            source_quality={
                "critical_data_quality": record.get("critical_data_quality") or payload.get("critical_data_quality"),
                "overall_status": record.get("critical_data_quality") or payload.get("critical_data_quality"),
            },
            context={"report_type": payload["report_type"], "as_of": payload["as_of"], "report_id": report_id},
        )
        if gate:
            gate = {**gate, "evaluated_at": payload["as_of"]}
            record["policy_gate_result"] = gate
        if gate and gate.get("decision") == "review_required" and record.get("recommendation_status") == "clear":
            record["recommendation_status"] = "review_required"
            record["status"] = "open"
        elif gate and gate.get("decision") in {"blocked", "error"}:
            record["recommendation_status"] = "blocked"
            record["status"] = "blocked"
            policy_reason = f"policy_gate:{gate.get('decision')}"
            if policy_reason not in record.get("blocked_reasons", []):
                record["blocked_reasons"] = [*record.get("blocked_reasons", []), policy_reason]
        if gate and not record.get("policy_gate_result_id"):
            gate_target_id = idempotency_key or action_hash
            gate_key = f"create_recommendation:{gate_target_id}"
            gate_uid = policy_gate_result_id(gate_key)
            object_service.write_object(
                "PolicyGateResult",
                gate_uid,
                {
                    "gate_result_id": gate_key,
                    "decision": gate.get("decision") or "review_required",
                    "review_required": bool(gate.get("review_required")),
                    "failure_reasons": gate.get("failure_reasons", []),
                    "warnings": gate.get("warnings", []),
                    "evaluated_at": payload["as_of"],
                    "ontology_run_id": "operational",
                },
                payload["as_of"],
                actor=actor_to_dict(actor),
                provenance=f"pv:recommendation_policy_gate:{gate_target_id}",
                input_hash=stable_hash({"gate": gate, "record": record}),
            )
            gate["policy_gate_result_id"] = gate_uid
            record["policy_gate_result_id"] = gate_uid
            record["policy_gate_result"] = gate
            record["policy_gate_status"] = gate.get("decision")
            record["policy_gate_decision"] = gate.get("decision")
            record["policy_gate_review_required"] = bool(gate.get("review_required"))
            record["policy_gate_failures"] = gate.get("failure_reasons", [])
            record["policy_gate_warnings"] = gate.get("warnings", [])
            record["policy_gate_disclosures"] = gate.get("disclosures", [])
        reason = (
            f"{payload['report_type'].title()} recommendation for "
            f"{action.get('instrument') or action.get('ticker') or 'portfolio'}"
        )
        approval = command_service.propose_action(
            "create_recommendation",
            {"record": record},
            context,
            reason=reason,
        )
        persisted.append({"status": "pending_approval_created", "approval_id": approval["id"], "record": record})
    return persisted


def _horizon_days(horizon: str | None) -> int:
    text = (horizon or "").lower()
    if "trading day" in text or "1 day" in text:
        return 1
    if "week" in text:
        return 7
    if "month" in text:
        return 30
    return 7


def _expected_direction(action: str) -> str | None:
    if action in {"buy", "add"}:
        return "up"
    if action in {"short", "sell", "trim", "reduce", "exit", "avoid"}:
        return "down"
    return None


def _download_close_series(ticker: str, start: date, end: date):
    import yfinance as yf

    hist = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if hist is None or hist.empty or "Close" not in hist:
        raise RuntimeError(f"no close price history for {ticker}")
    close = hist["Close"]
    if hasattr(close, "iloc") and getattr(close, "ndim", 1) > 1:
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise RuntimeError(f"empty close series for {ticker}")
    return close


def _series_return_pct(close) -> float:
    start = float(close.iloc[0])
    end = float(close.iloc[-1])
    if start == 0:
        return 0.0
    return (end / start - 1.0) * 100.0


def _excursions_pct(close, expected_direction: str) -> tuple[float, float]:
    start = float(close.iloc[0])
    running = (close / start - 1.0) * 100.0
    if expected_direction == "up":
        return float(running.min()), float(running.max())
    return float(-running.max()), float(-running.min())


def _timing_label(as_of: date, today: date, horizon: str | None, expected_onset_window: str | None) -> str:
    elapsed_days = (today - as_of).days
    horizon_days = _horizon_days(horizon)
    onset_days = _horizon_days(expected_onset_window)
    if elapsed_days < max(1, min(onset_days, horizon_days)):
        return "too_early"
    if elapsed_days <= max(onset_days, horizon_days) * 2:
        return "on_time"
    return "late"


def _process_label(process_quality: str, outcome_quality: str) -> str:
    if process_quality == "inconclusive" or outcome_quality == "inconclusive":
        return "inconclusive"
    return f"{process_quality}_process_{outcome_quality}_outcome"


def _thesis_and_kill_context(ticker: str | None) -> dict[str, Any]:
    if not ticker:
        return {"thesis_validation": None, "kill_condition_status": None}
    context: dict[str, Any] = {"thesis_validation": None, "kill_condition_status": None}
    try:
        from ontology.runtime_read_service import OntologyRuntimeReadService

        latest = OntologyRuntimeReadService().evaluations(ticker, limit=1)
        if latest:
            ev = latest[0]
            context["thesis_validation"] = {
                "evaluated_at": ev.get("evaluated_at"),
                "thesis_status": ev.get("thesis_status"),
                "action": ev.get("action"),
                "risk_flag": ev.get("risk_flag"),
            }
    except Exception:
        context["thesis_validation"] = {"status": "unavailable"}
    try:
        from ontology.runtime_read_service import OntologyRuntimeReadService

        conditions = OntologyRuntimeReadService().kill_conditions(ticker)
        context["kill_condition_status"] = {
            "active": sum(1 for row in conditions if row.get("status") == "active"),
            "triggered": sum(1 for row in conditions if row.get("status") == "triggered"),
            "retired": sum(1 for row in conditions if row.get("status") == "retired"),
        }
    except Exception:
        context["kill_condition_status"] = {"status": "unavailable"}
    return context


def evaluate_due_recommendations(limit: int = 50) -> dict:
    from ontology.object_service import OntologyObjectService
    from ontology.policy import actor_to_dict, system_actor
    from ontology.runtime_read_service import OntologyRuntimeReadService

    today = datetime.now(UTC).date()
    reads = OntologyRuntimeReadService()
    objects = OntologyObjectService()
    actor = system_actor("recommendation_evaluator")

    def update_recommendation_outcome(rec: dict[str, Any], status: str, outcome: dict[str, Any]) -> None:
        rec_uid = str(rec.get("object_uid") or rec.get("id") or rec.get("recommendation_id") or "")
        payload = dict(rec.get("payload") or {})
        payload["outcome"] = outcome
        props = {
            "recommendation_id": rec.get("recommendation_id") or rec_uid,
            "idempotency_key": rec.get("idempotency_key"),
            "source_kind": rec.get("source_kind") or "report",
            "report_type": rec.get("report_type"),
            "as_of": rec.get("as_of"),
            "action": rec.get("action") or "watch",
            "ticker": rec.get("ticker"),
            "instrument": rec.get("instrument") or rec.get("ticker") or "portfolio",
            "decision_state": rec.get("decision_state") or "generated",
            "status": rec.get("status"),
            "approval_id": str(rec.get("approval_id")) if rec.get("approval_id") is not None else None,
            "approval_required": bool(rec.get("approval_required")),
            "approval_status": rec.get("approval_status"),
            "outcome_status": status,
            "supersedes_recommendation_id": rec.get("supersedes_recommendation_id"),
            "account_id": rec.get("account_id"),
            "portfolio_id": rec.get("portfolio_id"),
            "policy_id": rec.get("policy_id"),
            "policy_gate_result_id": rec.get("policy_gate_result_id"),
            "policy_gate_decision": rec.get("policy_gate_decision") or rec.get("policy_gate_status"),
            "policy_gate_review_required": bool(rec.get("policy_gate_review_required")),
            "confidence": _as_float(rec.get("confidence"), 0.0),
            "horizon": rec.get("horizon"),
            "rationale_summary": str(rec.get("rationale") or "")[:500] or None,
            "rationale_hash": stable_hash(str(rec.get("rationale") or "")) if rec.get("rationale") else None,
            "source_quality": rec.get("source_quality"),
            "payload": payload,
            "ontology_run_id": "operational",
        }
        objects.write_object(
            "Recommendation",
            rec_uid,
            props,
            datetime.now(UTC).isoformat(),
            actor=actor_to_dict(actor),
            provenance=f"pv:recommendation_outcome:{rec_uid}",
        )

    checked = 0
    updated = 0
    unavailable = 0
    for rec in reads.recommendations(outcome_status="pending", limit=limit):
        checked += 1
        as_of = _parse_date(rec.get("as_of"))
        if as_of is None:
            update_recommendation_outcome(rec, "unavailable", {"reason": "missing as_of date"})
            unavailable += 1
            continue
        if today < as_of + timedelta(days=_horizon_days(rec.get("horizon"))):
            continue
        action = rec.get("action")
        if action == "do_nothing":
            update_recommendation_outcome(
                rec,
                "evaluated",
                {
                    "evaluation_authority": "ai_draft_user_final",
                    "final_label_status": "draft",
                    "process_label": "inconclusive",
                    "timing_vs_expected_onset": _timing_label(
                        as_of,
                        today,
                        rec.get("horizon"),
                        rec.get("expected_onset_window"),
                    ),
                    "opportunity_cost": rec.get("opportunity_cost_json", []),
                    "draft_postmortem": "No-action recommendation reached its review horizon. User should confirm whether inaction preserved optionality or missed an actionable opportunity.",
                    "objective_score_available": False,
                },
            )
            updated += 1
            continue
        ticker = rec.get("ticker")
        direction = _expected_direction(str(action))
        if not ticker or direction is None:
            update_recommendation_outcome(
                rec,
                "unavailable",
                {
                    "reason": "broad or non-directional recommendation; manual review required",
                    "process_label": "inconclusive",
                    "opportunity_cost": rec.get("opportunity_cost_json", []),
                },
            )
            unavailable += 1
            continue
        try:
            close = _download_close_series(ticker, as_of, today)
            benchmark_close = _download_close_series("SPY", as_of, today)
            start = float(close.iloc[0])
            end = float(close.iloc[-1])
            forward_return = _series_return_pct(close)
            benchmark_return = _series_return_pct(benchmark_close)
            relative_return = forward_return - benchmark_return
            max_adverse, max_favorable = _excursions_pct(close, direction)
            directionally_right = forward_return > 0 if direction == "up" else forward_return < 0
            relative_right = relative_return > 0 if direction == "up" else relative_return < 0
            source_quality = str(rec.get("source_quality") or "")
            confidence = _as_float(rec.get("confidence"), 0.0)
            process_quality = (
                "good"
                if source_quality in {"ok", "degraded"}
                and confidence >= 0.5
                and rec.get("recommendation_status") == "clear"
                else "bad"
            )
            outcome_quality = "good" if directionally_right and relative_right else "bad"
            thesis_context = _thesis_and_kill_context(ticker)
            update_recommendation_outcome(
                rec,
                "evaluated",
                {
                    "evaluation_authority": "ai_draft_user_final",
                    "final_label_status": "draft",
                    "start_price": start,
                    "end_price": end,
                    "forward_return_pct": round(forward_return, 2),
                    "benchmark": "SPY",
                    "benchmark_return_pct": round(benchmark_return, 2),
                    "benchmark_relative_return_pct": round(relative_return, 2),
                    "max_adverse_move_pct": round(max_adverse, 2),
                    "max_favorable_move_pct": round(max_favorable, 2),
                    "expected_direction": direction,
                    "directionally_right": directionally_right,
                    "relative_directionally_right": relative_right,
                    "alternative_trade_performance": {
                        "cash_return_pct": 0.0,
                        "benchmark_return_pct": round(benchmark_return, 2),
                    },
                    "sizing_quality": {
                        "target_change": rec.get("target_change"),
                        "approval_status": rec.get("approval_status"),
                        "label": "unverified_execution"
                        if rec.get("approval_status") != "approved"
                        else "requires_trade_fill_review",
                    },
                    "timing_vs_expected_onset": _timing_label(
                        as_of,
                        today,
                        rec.get("horizon"),
                        rec.get("expected_onset_window"),
                    ),
                    "catalyst_result": {
                        "catalyst": rec.get("catalyst"),
                        "label": "requires_review" if rec.get("catalyst") else "none_specified",
                    },
                    "opportunity_cost": rec.get("opportunity_cost_json", []),
                    "process_label": _process_label(process_quality, outcome_quality),
                    **thesis_context,
                    "draft_postmortem": "Objective price and process-attribution fields computed. User should confirm execution, catalyst, and thesis labels.",
                },
            )
            updated += 1
        except Exception as exc:
            update_recommendation_outcome(rec, "unavailable", {"reason": str(exc)})
            unavailable += 1
    return {"checked": checked, "updated": updated, "unavailable": unavailable}
