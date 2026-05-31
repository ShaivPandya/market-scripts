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
    SNAPSHOT_MOMENTUM,
    SNAPSHOT_POSITIONING_SUMMARY,
    SNAPSHOT_SECTOR_METRICS,
    SNAPSHOT_SENTIMENT,
    SNAPSHOT_SIGNAL_AGGREGATOR,
    SNAPSHOT_TOP50_BREADTH,
    SNAPSHOT_VIX_TERM_STRUCTURE,
)
from api.snapshot_store import SnapshotRecord, list_snapshot_records
from ontology.sources.reliability import (
    enrich_source_reliability,
    gate_action_for_tier,
    sla_seconds_for_registry,
    tier_counts,
)
from ontology.sources.source_registry import source_registry_metadata, source_registry_metadata_for_snapshot

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
    SNAPSHOT_MOMENTUM: "momentum",
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
    SNAPSHOT_MOMENTUM: "market",
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

_REGIME_MODULE_SNAPSHOT_KEYS = {
    **_RISK_SNAPSHOT_KEYS,
    "momentum": SNAPSHOT_MOMENTUM,
}

_APPROVAL_SOURCE_DEPENDENCY_FIELDS = (
    "source_dependencies",
    "required_sources",
    "source_requirements",
    "source_ids",
    "required_source_ids",
)


def build_workspace_source_health(
    *,
    portfolio_risk: dict[str, Any] | None = None,
    portfolio_data: dict[str, Any] | None = None,
    regime_data: dict[str, Any] | None = None,
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

    for source in _sources_from_regime_data(regime_data, required_sources=required_sources, now=now):
        if source["id"] not in sources_by_key:
            sources_by_key[source["id"]] = source

    portfolio_source = _source_from_portfolio_data(portfolio_data, now=now)
    if portfolio_source is not None and portfolio_source["id"] not in sources_by_key:
        sources_by_key[portfolio_source["id"]] = portfolio_source

    for source_id, required in required_sources.items():
        if source_id in sources_by_key:
            sources_by_key[source_id]["required"] = required or bool(sources_by_key[source_id].get("required"))
            continue
        sources_by_key[source_id] = _missing_required_source(source_id)

    sources = sorted(
        sources_by_key.values(),
        key=lambda row: (str(row["domain"]), not bool(row["required"]), str(row["source_name"])),
    )
    sources = [enrich_source_reliability(source) for source in sources]
    counts = _counts(sources)
    counts["tier_counts"] = tier_counts(sources)
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
        "tier_counts": counts.get("tier_counts") or tier_counts(sources),
        "domains": domains,
    }


def build_approval_source_health_review(
    approval: dict[str, Any] | None,
    source_health: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize source-health issues that matter during approval review."""
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    explicit_dependencies = _approval_source_dependencies(approval)

    for source in _source_health_sources(source_health):
        enriched = enrich_source_reliability(source)
        status = str(enriched.get("status") or "missing").strip().lower()
        tier = str(enriched.get("reliability_tier") or "standard").strip().lower()
        required = bool(enriched.get("required"))
        explicit = _source_matches_dependency(enriched, explicit_dependencies)

        if tier == "critical" and status in {"stale", "failed", "missing"}:
            blockers.append(_approval_source_issue(enriched, reason=f"critical source is {status}"))
            continue
        if status in {"failed", "missing"} and (required or explicit):
            blockers.append(_approval_source_issue(enriched, reason="required source unavailable"))
            continue
        if status == "stale" and explicit:
            blockers.append(_approval_source_issue(enriched, reason="explicit source dependency is stale"))
            continue
        if tier == "standard" and required and status in {"stale", "degraded"}:
            warnings.append(_approval_source_issue(enriched, reason="standard source needs review"))
            continue
        if not required and status in {"degraded", "stale", "failed", "missing"}:
            warnings.append(_approval_source_issue(enriched, reason="optional source degraded"))
            continue
        if explicit and status == "degraded":
            warnings.append(_approval_source_issue(enriched, reason="explicit source dependency is degraded"))

    return {
        "status": "blocked" if blockers else "warning" if warnings else "ok",
        "blockers": blockers,
        "warnings": warnings,
        "generated_at": (source_health or {}).get("generated_at") if isinstance(source_health, dict) else None,
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
    registry = source_registry_metadata_for_snapshot(record.snapshot_key)
    required = required or bool((registry or {}).get("required"))
    sla_seconds = sla_seconds_for_registry(registry)
    stale = _snapshot_is_stale(record, now=now, sla_seconds=sla_seconds)
    status = _normalize_status(record.status, stale=stale, quality=record.quality, required=required)
    source_name = str(
        (registry or {}).get("source_id")
        or _SNAPSHOT_SOURCE_NAMES.get(record.snapshot_key)
        or _source_name_from_snapshot_key(record.snapshot_key)
    )
    return {
        "id": record.snapshot_key,
        "domain": str((registry or {}).get("dataset_domain") or _domain_for_snapshot(record.snapshot_key)),
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
        "source_registry": registry,
    }


def _source_from_risk_status(module: str, state: dict[str, Any], *, required: bool) -> dict[str, Any]:
    raw_freshness = state.get("freshness")
    freshness = raw_freshness if isinstance(raw_freshness, dict) else {}
    stale = str(state.get("status") or "").lower() == "stale" or freshness.get("fresh") is False
    snapshot_key = str(state.get("snapshot_key") or _RISK_SNAPSHOT_KEYS.get(module) or module)
    source_id = snapshot_key if snapshot_key in _SNAPSHOT_SOURCE_NAMES else module
    raw_registry = state.get("source_registry")
    registry = raw_registry if isinstance(raw_registry, dict) else None
    if registry is None:
        registry = source_registry_metadata_for_snapshot(snapshot_key) or source_registry_metadata(module)
    required = required or bool(state.get("required")) or bool((registry or {}).get("required"))
    status = _normalize_status(state.get("status"), stale=stale, quality=state.get("quality"), required=required)
    return {
        "id": source_id,
        "domain": str((registry or {}).get("dataset_domain") or ("portfolio" if module == "portfolio" else "risk")),
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
        "source_registry": registry,
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
    for key in (
        "status",
        "quality_state",
        "stale",
        "error",
        "detail",
        "freshness",
        "freshness_timestamp",
        "source_registry",
    ):
        if key in {"status", "quality_state", "stale"}:
            continue
        if incoming.get(key) not in (None, "", False):
            out[key] = incoming[key]
    out["domain"] = existing.get("domain") or incoming.get("domain")
    out["source_name"] = incoming.get("source_name") or existing.get("source_name")
    return out


def _sources_from_regime_data(
    regime_data: dict[str, Any] | None,
    *,
    required_sources: dict[str, bool],
    now: datetime | None,
) -> list[dict[str, Any]]:
    if not isinstance(regime_data, dict):
        return []

    sources = [_source_from_regime_data(regime_data, required=SNAPSHOT_SIGNAL_AGGREGATOR in required_sources, now=now)]
    module_status = regime_data.get("module_status")
    if not isinstance(module_status, dict):
        return sources

    fallback_as_of = _first_non_empty(_snapshot_meta(regime_data).get("as_of"), regime_data.get("as_of"))
    fallback_fetched_at = _first_non_empty(
        _snapshot_meta(regime_data).get("fetched_at"), (now or datetime.now()).isoformat()
    )
    raw_modules = regime_data.get("raw_modules")
    raw_modules = raw_modules if isinstance(raw_modules, dict) else {}
    for module, raw_state in module_status.items():
        if not isinstance(raw_state, dict):
            continue
        module_name = str(module)
        snapshot_key = _REGIME_MODULE_SNAPSHOT_KEYS.get(module_name)
        if snapshot_key is None:
            continue

        state = dict(raw_state)
        state["snapshot_key"] = snapshot_key
        state.setdefault("as_of", _payload_as_of(raw_modules.get(module_name)) or fallback_as_of)
        state.setdefault("fetched_at", fallback_fetched_at)
        source = _source_from_risk_status(module_name, state, required=snapshot_key in required_sources)
        sources.append(source)
    return sources


def _source_from_regime_data(
    regime_data: dict[str, Any],
    *,
    required: bool,
    now: datetime | None,
) -> dict[str, Any]:
    registry = source_registry_metadata_for_snapshot(SNAPSHOT_SIGNAL_AGGREGATOR)
    meta = _snapshot_meta(regime_data)
    refresh_status = meta.get("refresh_status")
    status_value = refresh_status if refresh_status not in (None, "") else regime_data.get("status") or "ok"
    stale = bool(meta.get("stale"))
    status = _normalize_status(status_value, stale=stale, quality=regime_data.get("quality"), required=required)
    as_of = _first_non_empty(meta.get("as_of"), regime_data.get("as_of"))
    fetched_at = _first_non_empty(meta.get("fetched_at"), (now or datetime.now()).isoformat())
    detail = meta.get("error")
    if not detail and meta.get("source") == "module_snapshots":
        detail = "computed from module snapshots"
    return {
        "id": SNAPSHOT_SIGNAL_AGGREGATOR,
        "domain": str((registry or {}).get("dataset_domain") or _domain_for_snapshot(SNAPSHOT_SIGNAL_AGGREGATOR)),
        "source_name": "market_regime",
        "snapshot_key": SNAPSHOT_SIGNAL_AGGREGATOR,
        "status": status,
        "quality_state": _quality_state(status, required=required),
        "required": required or bool((registry or {}).get("required")),
        "as_of": as_of,
        "fetched_at": fetched_at,
        "freshness_timestamp": as_of or fetched_at,
        "stale": stale,
        "error": meta.get("error"),
        "detail": detail,
        "payload_hash": None,
        "source_registry": registry,
    }


def _source_from_portfolio_data(
    portfolio_data: dict[str, Any] | None, *, now: datetime | None
) -> dict[str, Any] | None:
    if not isinstance(portfolio_data, dict):
        return None
    registry = source_registry_metadata("portfolio")
    fetched_at = (now or datetime.now()).isoformat()
    error = portfolio_data.get("error")
    positions = portfolio_data.get("positions")
    as_of = _first_non_empty(
        portfolio_data.get("as_of"),
        portfolio_data.get("computed_at"),
        _latest_position_as_of(positions),
        fetched_at,
    )
    status = "failed" if error else "ok"
    return {
        "id": "portfolio",
        "domain": "portfolio",
        "source_name": "portfolio",
        "snapshot_key": None,
        "status": status,
        "quality_state": _quality_state(status, required=True),
        "required": True,
        "as_of": as_of,
        "fetched_at": fetched_at,
        "freshness_timestamp": as_of or fetched_at,
        "stale": False,
        "error": error,
        "detail": str(error) if error else None,
        "source_registry": registry,
        "position_count": len(positions) if isinstance(positions, list) else None,
    }


def _snapshot_meta(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        return {}
    snapshot = meta.get("snapshot")
    return snapshot if isinstance(snapshot, dict) else {}


def _latest_position_as_of(positions: Any) -> str | None:
    if not isinstance(positions, list):
        return None
    dates = [
        str(row.get("as_of") or row.get("date") or row.get("updated_at"))
        for row in positions
        if isinstance(row, dict) and (row.get("as_of") or row.get("date") or row.get("updated_at"))
    ]
    return max(dates) if dates else None


def _payload_as_of(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("as_of", "as_of_date", "latest_date", "date", "timestamp"):
        value = payload.get(key)
        if value is not None:
            return str(value)[:32]
    latest = payload.get("latest_df")
    if isinstance(latest, list) and latest and isinstance(latest[0], dict):
        value = latest[0].get("Date") or latest[0].get("date")
        if value is not None:
            return str(value)[:32]
    return None


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _worse_status(left: str, right: str) -> str:
    severity = {"ok": 0, "degraded": 1, "stale": 2, "missing": 3, "failed": 4}
    return left if severity.get(left, 1) >= severity.get(right, 1) else right


def _missing_required_source(source_id: str) -> dict[str, Any]:
    registry = source_registry_metadata_for_snapshot(source_id) or source_registry_metadata(source_id)
    return {
        "id": source_id,
        "domain": str(
            (registry or {}).get("dataset_domain")
            or ("portfolio" if source_id == "portfolio" else _domain_for_snapshot(source_id))
        ),
        "source_name": str(
            (registry or {}).get("source_id")
            or _SNAPSHOT_SOURCE_NAMES.get(source_id)
            or _source_name_from_snapshot_key(source_id)
        ),
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
        "source_registry": registry,
    }


def _snapshot_is_stale(record: SnapshotRecord, *, now: datetime | None, sla_seconds: int | None = None) -> bool:
    max_age = sla_seconds if sla_seconds is not None else DEFAULT_SNAPSHOT_MAX_AGE_SECONDS
    try:
        fetched = datetime.fromisoformat(str(record.fetched_at).replace("Z", "+00:00"))
        current = now or datetime.now(tz=fetched.tzinfo) if fetched.tzinfo else now or datetime.now()
        return max(0, (current - fetched).total_seconds()) > max_age
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
        "critical_stale": 0,
        "critical_failed": 0,
        "sla_breach": 0,
    }
    for source in sources:
        status = str(source.get("status") or "missing")
        tier = str(source.get("reliability_tier") or "")
        if status in counts:
            counts[status] += 1
        if source.get("required") and status == "stale":
            counts["required_stale"] += 1
        if source.get("required") and status in {"failed", "missing"}:
            counts["required_failed"] += 1
        if not source.get("required") and status in {"degraded", "failed", "missing", "stale"}:
            counts["optional_degraded"] += 1
        if tier == "critical" and status == "stale":
            counts["critical_stale"] += 1
        if tier == "critical" and status in {"failed", "missing"}:
            counts["critical_failed"] += 1
        if source.get("sla_breach"):
            counts["sla_breach"] += 1
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


def _source_health_sources(source_health: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(source_health, dict):
        return []
    sources: list[dict[str, Any]] = []
    for domain in source_health.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        for source in domain.get("sources") or []:
            if isinstance(source, dict):
                sources.append(source)
    return sources


def _approval_source_dependencies(approval: dict[str, Any] | None) -> set[str]:
    if not isinstance(approval, dict):
        return set()
    proposed = _as_dict(approval.get("proposed_change")) or {}
    record = _as_dict(proposed.get("record")) or {}
    policy_gate = _as_dict(approval.get("policy_gate_result")) or _as_dict(proposed.get("policy_gate_result")) or {}
    dependencies: set[str] = set()
    for container in (approval, proposed, record, policy_gate):
        for field in _APPROVAL_SOURCE_DEPENDENCY_FIELDS:
            dependencies.update(_source_dependency_values(container.get(field)))
    return dependencies


def _source_dependency_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, str):
        text = value.strip().lower()
        if text:
            values.add(text)
        return values
    if isinstance(value, dict):
        has_identity = any(field in value for field in ("id", "source_id", "snapshot_key", "source_name", "name"))
        for field in ("id", "source_id", "snapshot_key", "source_name", "name"):
            raw = value.get(field)
            if raw is not None:
                values.update(_source_dependency_values(raw))
        if has_identity:
            return values
        for key, raw in value.items():
            if raw in (None, "", False):
                continue
            values.update(_source_dependency_values(key))
            if isinstance(raw, (str, dict, list, tuple, set)):
                values.update(_source_dependency_values(raw))
        return values
    if isinstance(value, (list, tuple, set)):
        for item in value:
            values.update(_source_dependency_values(item))
    return values


def _source_matches_dependency(source: dict[str, Any], dependencies: set[str]) -> bool:
    if not dependencies:
        return False
    aliases = {
        str(source.get("id") or "").strip().lower(),
        str(source.get("source_name") or "").strip().lower(),
        str(source.get("snapshot_key") or "").strip().lower(),
    }
    registry = source.get("source_registry")
    if isinstance(registry, dict):
        aliases.add(str(registry.get("source_id") or "").strip().lower())
    return any(alias and alias in dependencies for alias in aliases)


def _approval_source_issue(source: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "id": source.get("id"),
        "source_name": source.get("source_name"),
        "domain": source.get("domain"),
        "status": source.get("status"),
        "quality_state": source.get("quality_state"),
        "required": bool(source.get("required")),
        "reliability_tier": source.get("reliability_tier"),
        "sla_breach": bool(source.get("sla_breach")),
        "gate_action": source.get("gate_action")
        or gate_action_for_tier(
            str(source.get("reliability_tier") or "standard"),
            str(source.get("status") or "missing"),
        ),
        "as_of": source.get("as_of"),
        "fetched_at": source.get("fetched_at"),
        "freshness_timestamp": source.get("freshness_timestamp"),
        "detail": source.get("detail"),
        "reason": reason,
    }


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None
