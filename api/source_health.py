"""Workspace source freshness and quality read model."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from api.snapshot_keys import (
    DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
    SNAPSHOT_ECONOMIC_GROWTH,
    SNAPSHOT_LABOR_MARKET,
    SNAPSHOT_LIQUIDITY,
    SNAPSHOT_MARKET_BREADTH,
    SNAPSHOT_POSITIONING_SUMMARY,
    SNAPSHOT_SECTOR_METRICS,
    SNAPSHOT_SENTIMENT,
    SNAPSHOT_SIGNAL_AGGREGATOR,
    SNAPSHOT_TOP50_BREADTH,
    SNAPSHOT_VIX_TERM_STRUCTURE,
)
from api.snapshot_store import SnapshotRecord, list_snapshot_records

_SNAPSHOT_SOURCE_NAMES = {
    SNAPSHOT_MARKET_BREADTH: "market_breadth",
    SNAPSHOT_TOP50_BREADTH: "top50_breadth",
    SNAPSHOT_VIX_TERM_STRUCTURE: "vix_term_structure",
    SNAPSHOT_SECTOR_METRICS: "sector_metrics",
    SNAPSHOT_LIQUIDITY: "liquidity",
    SNAPSHOT_SENTIMENT: "sentiment",
    SNAPSHOT_POSITIONING_SUMMARY: "positioning_summary",
    SNAPSHOT_ECONOMIC_GROWTH: "economic_growth",
    SNAPSHOT_LABOR_MARKET: "labor_market",
    SNAPSHOT_SIGNAL_AGGREGATOR: "market_regime",
}

_SNAPSHOT_DOMAINS = {
    SNAPSHOT_MARKET_BREADTH: "market",
    SNAPSHOT_TOP50_BREADTH: "market",
    SNAPSHOT_VIX_TERM_STRUCTURE: "market",
    SNAPSHOT_SECTOR_METRICS: "market",
    SNAPSHOT_SIGNAL_AGGREGATOR: "market",
    SNAPSHOT_LIQUIDITY: "macro",
    SNAPSHOT_ECONOMIC_GROWTH: "macro",
    SNAPSHOT_LABOR_MARKET: "macro",
    SNAPSHOT_SENTIMENT: "retrieval",
    SNAPSHOT_POSITIONING_SUMMARY: "risk",
}

_RISK_SNAPSHOT_KEYS = {
    "market_breadth": SNAPSHOT_MARKET_BREADTH,
    "top50_breadth": SNAPSHOT_TOP50_BREADTH,
    "vix_term_structure": SNAPSHOT_VIX_TERM_STRUCTURE,
    "sector_metrics": SNAPSHOT_SECTOR_METRICS,
    "liquidity": SNAPSHOT_LIQUIDITY,
    "economic_growth": SNAPSHOT_ECONOMIC_GROWTH,
    "sentiment": SNAPSHOT_SENTIMENT,
    "positioning_summary": SNAPSHOT_POSITIONING_SUMMARY,
    "labor_market": SNAPSHOT_LABOR_MARKET,
}


def build_workspace_source_health(
    *,
    portfolio_risk: dict[str, Any] | None = None,
    now: datetime | None = None,
    snapshot_records: list[SnapshotRecord] | None = None,
) -> dict[str, Any]:
    """Aggregate current snapshot and risk-source quality into a Workspace read model."""
    generated_at = (now or datetime.now()).isoformat()
    records = snapshot_records if snapshot_records is not None else list_snapshot_records()
    sources_by_key: dict[str, dict[str, Any]] = {}

    required_sources = _required_sources(portfolio_risk)
    for record in records:
        source = _source_from_snapshot(record, required=record.snapshot_key in required_sources, now=now)
        sources_by_key[source["id"]] = source

    for module, state in _risk_source_status(portfolio_risk).items():
        source = _source_from_risk_status(module, state, required=module in required_sources)
        sources_by_key[source["id"]] = _merge_source(sources_by_key.get(source["id"]), source)

    for source_id, required in required_sources.items():
        if source_id in sources_by_key:
            sources_by_key[source_id]["required"] = required or bool(sources_by_key[source_id].get("required"))
            continue
        sources_by_key[source_id] = _missing_required_source(source_id)

    sources = sorted(
        sources_by_key.values(),
        key=lambda row: (str(row["domain"]), not bool(row["required"]), str(row["source_name"])),
    )
    counts = _counts(sources)
    domains = []
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        by_domain[str(source["domain"])].append(source)
    for domain_name in sorted(by_domain):
        domain_sources = by_domain[domain_name]
        domains.append(
            {
                "domain": domain_name,
                "label": _domain_label(domain_name),
                "overall_quality": _overall_quality(domain_sources),
                "counts": _counts(domain_sources),
                "sources": domain_sources,
            }
        )

    return {
        "generated_at": generated_at,
        "overall_quality": _overall_quality(sources),
        "counts": counts,
        "domains": domains,
    }


def _required_sources(portfolio_risk: dict[str, Any] | None) -> dict[str, bool]:
    from api.position_risk import REQUIRED_MODULES

    required: dict[str, bool] = {SNAPSHOT_SIGNAL_AGGREGATOR: True, "portfolio": True}
    for module in REQUIRED_MODULES:
        snapshot_key = _RISK_SNAPSHOT_KEYS.get(module)
        if snapshot_key:
            required[snapshot_key] = True
        else:
            required[module] = True
    for module, state in _risk_source_status(portfolio_risk).items():
        if bool(state.get("required")):
            snapshot_key = str(state.get("snapshot_key") or "")
            if snapshot_key:
                required[snapshot_key] = True
            else:
                required[module] = True
    return required


def _risk_source_status(portfolio_risk: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(portfolio_risk, dict):
        return {}
    raw = portfolio_risk.get("source_status")
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(value, dict)}


def _source_from_snapshot(record: SnapshotRecord, *, required: bool, now: datetime | None) -> dict[str, Any]:
    stale = _snapshot_is_stale(record, now=now)
    status = _normalize_status(record.status, stale=stale, quality=record.quality, required=required)
    source_name = _SNAPSHOT_SOURCE_NAMES.get(record.snapshot_key) or _source_name_from_snapshot_key(record.snapshot_key)
    return {
        "id": record.snapshot_key,
        "domain": _domain_for_snapshot(record.snapshot_key),
        "source_name": source_name,
        "snapshot_key": record.snapshot_key,
        "status": status,
        "quality_state": _quality_state(status, required=required),
        "required": required,
        "as_of": record.as_of_date,
        "fetched_at": record.fetched_at,
        "freshness_timestamp": record.as_of_date or record.fetched_at,
        "stale": stale,
        "error": record.error,
        "detail": record.error or ("snapshot is stale" if stale else None),
        "payload_hash": record.payload_hash,
    }


def _source_from_risk_status(module: str, state: dict[str, Any], *, required: bool) -> dict[str, Any]:
    raw_freshness = state.get("freshness")
    freshness = raw_freshness if isinstance(raw_freshness, dict) else {}
    stale = str(state.get("status") or "").lower() == "stale" or freshness.get("fresh") is False
    status = _normalize_status(state.get("status"), stale=stale, quality=state.get("quality"), required=required)
    snapshot_key = str(state.get("snapshot_key") or _RISK_SNAPSHOT_KEYS.get(module) or module)
    source_id = snapshot_key if snapshot_key in _RISK_SNAPSHOT_KEYS.values() else module
    return {
        "id": source_id,
        "domain": "portfolio" if module == "portfolio" else "risk",
        "source_name": module,
        "snapshot_key": snapshot_key if snapshot_key != module else None,
        "status": status,
        "quality_state": _quality_state(status, required=required),
        "required": required or bool(state.get("required")),
        "as_of": state.get("as_of"),
        "fetched_at": state.get("fetched_at") or state.get("checked_at"),
        "freshness_timestamp": freshness.get("observed_as_of_date") or state.get("as_of") or state.get("fetched_at"),
        "stale": stale,
        "error": state.get("error"),
        "detail": state.get("detail") or state.get("error") or freshness.get("reason"),
        "freshness": freshness or None,
    }


def _merge_source(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return incoming
    out = dict(existing)
    out["required"] = bool(existing.get("required")) or bool(incoming.get("required"))
    status = _worse_status(str(existing.get("status") or "missing"), str(incoming.get("status") or "missing"))
    out["status"] = status
    out["quality_state"] = _quality_state(status, required=bool(out["required"]))
    out["stale"] = bool(existing.get("stale")) or bool(incoming.get("stale"))
    for key in ("status", "quality_state", "stale", "error", "detail", "freshness", "freshness_timestamp"):
        if key in {"status", "quality_state", "stale"}:
            continue
        if incoming.get(key) not in (None, "", False):
            out[key] = incoming[key]
    out["domain"] = existing.get("domain") or incoming.get("domain")
    out["source_name"] = incoming.get("source_name") or existing.get("source_name")
    return out


def _worse_status(left: str, right: str) -> str:
    severity = {"ok": 0, "degraded": 1, "stale": 2, "missing": 3, "failed": 4}
    return left if severity.get(left, 1) >= severity.get(right, 1) else right


def _missing_required_source(source_id: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "domain": "portfolio" if source_id == "portfolio" else _domain_for_snapshot(source_id),
        "source_name": _SNAPSHOT_SOURCE_NAMES.get(source_id) or _source_name_from_snapshot_key(source_id),
        "snapshot_key": source_id if ":" in source_id else None,
        "status": "missing",
        "quality_state": "missing",
        "required": True,
        "as_of": None,
        "fetched_at": None,
        "freshness_timestamp": None,
        "stale": False,
        "error": None,
        "detail": "source has no freshness record yet",
    }


def _snapshot_is_stale(record: SnapshotRecord, *, now: datetime | None) -> bool:
    try:
        fetched = datetime.fromisoformat(str(record.fetched_at).replace("Z", "+00:00"))
        current = now or datetime.now(tz=fetched.tzinfo) if fetched.tzinfo else now or datetime.now()
        return max(0, (current - fetched).total_seconds()) > DEFAULT_SNAPSHOT_MAX_AGE_SECONDS
    except Exception:
        return False


def _normalize_status(raw_status: Any, *, stale: bool, quality: Any, required: bool) -> str:
    status = str(raw_status or "missing").strip().lower()
    quality_text = str(quality or "").strip().lower()
    if status in {"ok", "fresh", "hit", "success"}:
        if stale:
            return "stale"
        if quality_text in {"degraded", "partial"}:
            return "degraded"
        return "ok"
    if status in {"stale"}:
        return "stale"
    if status in {"degraded", "partial"}:
        return "degraded"
    if status in {"missing", "none", "unknown"}:
        return "missing"
    if status in {"error", "failed", "failure"}:
        return "failed" if required else "degraded"
    return status


def _quality_state(status: str, *, required: bool) -> str:
    if status in {"ok", "stale", "degraded", "missing"}:
        return status
    if status == "failed":
        return "failed" if required else "degraded"
    return "degraded"


def _overall_quality(sources: list[dict[str, Any]]) -> str:
    required = [source for source in sources if source.get("required")]
    if any(source.get("status") in {"failed", "missing"} for source in required):
        return "failed"
    if any(source.get("status") == "stale" for source in required):
        return "stale"
    if any(source.get("status") in {"degraded", "failed", "missing", "stale"} for source in sources):
        return "degraded"
    return "ok"


def _counts(sources: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(sources),
        "ok": 0,
        "stale": 0,
        "degraded": 0,
        "failed": 0,
        "missing": 0,
        "required_stale": 0,
        "required_failed": 0,
        "optional_degraded": 0,
    }
    for source in sources:
        status = str(source.get("status") or "missing")
        if status in counts:
            counts[status] += 1
        if source.get("required") and status == "stale":
            counts["required_stale"] += 1
        if source.get("required") and status in {"failed", "missing"}:
            counts["required_failed"] += 1
        if not source.get("required") and status in {"degraded", "failed", "missing", "stale"}:
            counts["optional_degraded"] += 1
    return counts


def _domain_for_snapshot(snapshot_key: str) -> str:
    if snapshot_key in _SNAPSHOT_DOMAINS:
        return _SNAPSHOT_DOMAINS[snapshot_key]
    prefix = snapshot_key.split(":", 1)[0]
    if prefix in {"economic_growth", "labor_market", "liquidity", "housing"}:
        return "macro"
    if prefix in {"market_breadth", "top50_breadth", "vix_term_structure", "sector_metrics", "momentum"}:
        return "market"
    if prefix in {"sentiment"}:
        return "retrieval"
    if prefix in {"positioning_summary"}:
        return "risk"
    return "snapshots"


def _domain_label(domain: str) -> str:
    return {
        "macro": "Macro",
        "market": "Market",
        "portfolio": "Portfolio",
        "retrieval": "Retrieval",
        "risk": "Risk",
        "snapshots": "Snapshots",
    }.get(domain, domain.replace("_", " ").title())


def _source_name_from_snapshot_key(snapshot_key: str) -> str:
    return str(snapshot_key or "source").split(":", 1)[0] or "source"
