"""Normalize agent tool result quality for chat decision-quality gates."""

from __future__ import annotations

from typing import Any

from ontology.sources.reliability import gate_action_for_tier

PRICE_CONFIRMATION_TOOLS = frozenset({"run_chart", "get_price_volume_signals"})
BLOCKING_TOOL_STATUSES = frozenset({"blocked", "error", "timeout", "failed", "failed_closed", "denied"})
WARNING_TOOL_STATUSES = frozenset({"partial", "cancelled", "retrying"})
BLOCKING_SOURCE_STATUSES = frozenset({"stale", "failed", "missing"})
WARNING_SOURCE_STATUSES = frozenset({"degraded", "partial"})

TOOL_RELIABILITY_TIERS: dict[str, str] = {
    "run_chart": "critical",
    "get_price_volume_signals": "critical",
    "get_dossier": "standard",
    "get_thesis": "standard",
    "get_thesis_evaluations": "standard",
    "get_position_valuation": "standard",
    "get_portfolio": "standard",
    "search_knowledge_base": "supplemental",
    "search_web": "supplemental",
    "query_ontology": "standard",
}


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _missing_fields(result: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    explicit = meta.get("missing_fields") or result.get("missing_fields")
    if isinstance(explicit, list):
        return [str(item) for item in explicit if isinstance(item, str) and item.strip()]
    data_needed = result.get("data_needed")
    if isinstance(data_needed, list):
        return [str(item) for item in data_needed if isinstance(item, str) and item.strip()]
    technical_read = result.get("technical_read")
    if isinstance(technical_read, dict):
        needed = technical_read.get("data_needed")
        if isinstance(needed, list):
            return [str(item) for item in needed if isinstance(item, str) and item.strip()]
    return []


def _source_status(result: dict[str, Any], meta: dict[str, Any], tool_status: str) -> str:
    explicit = _first_str(
        meta.get("source_status"),
        result.get("source_status"),
        meta.get("quality_state"),
        result.get("quality_state"),
    )
    if explicit:
        return explicit

    quality = _first_str(meta.get("quality"), result.get("quality"), result.get("data_quality"))
    if quality in {"ok", "degraded", "missing", "schema_drift"}:
        mapping = {"ok": "ok", "degraded": "degraded", "missing": "missing", "schema_drift": "degraded"}
        return mapping[quality]

    if meta.get("stale") is True or result.get("stale") is True:
        return "stale"
    freshness = _first_str(meta.get("freshness"), result.get("freshness"))
    if freshness == "stale":
        return "stale"

    nested_status = result.get("status")
    if isinstance(nested_status, str) and nested_status.strip().lower() in {
        "ok",
        "stale",
        "degraded",
        "missing",
        "failed",
        "partial",
        "blocked",
    }:
        return nested_status.strip().lower()

    if tool_status in BLOCKING_TOOL_STATUSES:
        if tool_status in {"blocked", "denied", "failed_closed"}:
            return "failed"
        if tool_status == "timeout":
            return "failed"
        return "missing"
    return "ok"


def _price_confirmation_state(name: str, tool_status: str, result: dict[str, Any], meta: dict[str, Any]) -> str | None:
    if name not in PRICE_CONFIRMATION_TOOLS:
        return None
    if tool_status in BLOCKING_TOOL_STATUSES:
        return "blocked" if tool_status in {"blocked", "denied", "failed_closed"} else "missing"
    source_status = _source_status(result, meta, tool_status)
    if source_status in {"stale"}:
        return "stale"
    if source_status in {"failed", "missing"}:
        return "missing"

    status = _first_str(result.get("status"), meta.get("price_confirmation_status"))
    if status in {"confirmed", "missing", "stale", "blocked", "inconclusive"}:
        return status

    if result.get("error"):
        return "missing"
    if _missing_fields(result, meta):
        return "missing"

    technical_read = result.get("technical_read")
    if isinstance(technical_read, str) and not technical_read.strip():
        return "missing"
    if isinstance(technical_read, dict) and not any(
        _nonempty(technical_read.get(key)) for key in ("observed_behavior", "interpretation", "summary", "signal")
    ):
        return "missing"

    signals = result.get("signals")
    if isinstance(signals, list) and not signals:
        return "missing"

    return "confirmed"


def _nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _confidence_limit(meta: dict[str, Any], result: dict[str, Any], gate_action: str) -> float | None:
    raw = meta.get("confidence_limit", result.get("confidence_limit"))
    if isinstance(raw, (int, float)):
        return float(raw)
    if gate_action == "block":
        return 0.45
    if gate_action == "warn":
        return 0.6
    return None


def normalize_tool_quality(tool_result: dict[str, Any]) -> dict[str, Any]:
    """Convert one chat tool result into a stable data-quality envelope."""
    name = str(tool_result.get("name") or "unknown")
    tool_status = str(tool_result.get("status") or "ok").strip().lower()
    result = _as_dict(tool_result.get("result"))
    meta = _as_dict(result.get("_meta"))
    reliability_tier = _first_str(
        meta.get("reliability_tier"), result.get("reliability_tier")
    ) or TOOL_RELIABILITY_TIERS.get(name, "standard")
    source_status = _source_status(result, meta, tool_status)
    gate_action = gate_action_for_tier(reliability_tier, source_status)
    if tool_status in BLOCKING_TOOL_STATUSES and reliability_tier == "critical":
        gate_action = "block"
    elif tool_status in BLOCKING_TOOL_STATUSES and gate_action == "ok":
        gate_action = "warn"

    missing_fields = _missing_fields(result, meta)
    price_confirmation = _price_confirmation_state(name, tool_status, result, meta)
    blocks_actionable = gate_action == "block" or (
        name in PRICE_CONFIRMATION_TOOLS and price_confirmation in {"missing", "stale", "blocked"}
    )

    reason_parts: list[str] = []
    if tool_status not in {"ok"}:
        reason_parts.append(f"tool_status={tool_status}")
    if source_status not in {"ok"}:
        reason_parts.append(f"source_status={source_status}")
    if missing_fields:
        reason_parts.append(f"missing_fields={', '.join(missing_fields[:3])}")
    if price_confirmation and price_confirmation != "confirmed":
        reason_parts.append(f"price_confirmation={price_confirmation}")

    return {
        "name": name,
        "tool_status": tool_status,
        "source_status": source_status,
        "freshness": "stale" if source_status == "stale" else "fresh" if source_status == "ok" else "unknown",
        "reliability_tier": reliability_tier,
        "missing_fields": missing_fields,
        "confidence_limit": _confidence_limit(meta, result, gate_action),
        "gate_action": gate_action,
        "blocks_actionable": blocks_actionable,
        "price_confirmation": price_confirmation,
        "reason": "; ".join(reason_parts) if reason_parts else None,
    }


def aggregate_tool_data_quality(tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-tool quality into the chat decision-quality gate payload."""
    summaries = [normalize_tool_quality(item) for item in tool_results if isinstance(item, dict)]
    blockers = [item for item in summaries if item.get("blocks_actionable")]
    warnings = [
        item
        for item in summaries
        if not item.get("blocks_actionable") and item.get("gate_action") in {"warn", "inform"}
    ]
    tool_errors = [
        f"{item['name']}: {item.get('reason') or item.get('tool_status')}"
        for item in summaries
        if item.get("tool_status") in BLOCKING_TOOL_STATUSES or item.get("source_status") in BLOCKING_SOURCE_STATUSES
    ]

    price_states = [
        str(item.get("price_confirmation")) for item in summaries if item.get("price_confirmation") is not None
    ]
    if any(state in {"blocked", "missing"} for state in price_states):
        price_confirmation_status = "blocked" if "blocked" in price_states else "missing"
    elif any(state == "stale" for state in price_states):
        price_confirmation_status = "stale"
    elif price_states and all(state == "confirmed" for state in price_states):
        price_confirmation_status = "confirmed"
    elif price_states:
        price_confirmation_status = "missing"
    else:
        price_confirmation_status = "missing"

    if blockers:
        if any(
            item.get("source_status") in {"failed", "missing"} or item.get("tool_status") in {"blocked", "error"}
            for item in blockers
        ):
            critical_data_quality = "failed"
        else:
            critical_data_quality = "stale"
    elif any(item.get("source_status") in WARNING_SOURCE_STATUSES for item in summaries):
        critical_data_quality = "degraded"
    else:
        critical_data_quality = "ok"

    source_health_status = "blocked" if blockers else "warning" if warnings else "ok"
    blocking_reason_codes: list[str] = []
    if blockers:
        blocking_reason_codes.append("CRITICAL_DATA_QUALITY")
    if price_confirmation_status in {"missing", "stale", "blocked"}:
        blocking_reason_codes.append("MISSING_PRICE_CONFIRMATION")

    overall_status = critical_data_quality if critical_data_quality in {"stale", "failed"} else source_health_status
    source_quality = critical_data_quality if critical_data_quality != "ok" else ("degraded" if warnings else "ok")

    return {
        "critical_data_quality": critical_data_quality,
        "source_quality": source_quality,
        "quality": source_quality,
        "overall_status": overall_status,
        "tool_errors": tool_errors,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "blocking_reason_codes": blocking_reason_codes,
        "price_confirmation_status": price_confirmation_status,
        "source_health_status": source_health_status,
        "tool_summaries": summaries,
    }
