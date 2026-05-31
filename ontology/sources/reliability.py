"""Data reliability tier derivation and gate semantics."""

from __future__ import annotations

from typing import Any

from api.snapshot_keys import DEFAULT_SNAPSHOT_MAX_AGE_SECONDS
from ontology.sources.source_registry import SourceRegistryEntry

RELIABILITY_TIERS = frozenset({"critical", "standard", "supplemental", "ad_hoc"})

TIER_LABELS = {
    "critical": "Critical",
    "standard": "Standard",
    "supplemental": "Supplemental",
    "ad_hoc": "Ad hoc",
}


def derive_reliability_tier(entry: SourceRegistryEntry | None) -> str:
    """Return the effective reliability tier for a registry entry."""
    if entry is None:
        return "standard"
    if entry.reliability_tier is not None:
        tier = str(entry.reliability_tier).strip().lower()
        if tier in RELIABILITY_TIERS:
            return tier
    if entry.freshness_sla_seconds is None:
        return "ad_hoc"
    if entry.required and entry.authority_rank == 1:
        return "critical"
    if entry.required:
        return "standard"
    if entry.authority_rank >= 2:
        return "supplemental"
    return "standard"


def sla_seconds_for_registry(registry: dict[str, Any] | None) -> int:
    if isinstance(registry, dict):
        if str(registry.get("freshness_policy") or "").strip().lower() == "max_age_days":
            raw_days = registry.get("freshness_max_age_days")
            if raw_days is not None:
                try:
                    return max(1, int(raw_days)) * 24 * 60 * 60
                except (TypeError, ValueError):
                    pass
        raw = registry.get("freshness_sla_seconds")
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
    return DEFAULT_SNAPSHOT_MAX_AGE_SECONDS


def effective_reliability_tier_for_source(source: dict[str, Any]) -> str:
    explicit = source.get("reliability_tier")
    if isinstance(explicit, str) and explicit.strip().lower() in RELIABILITY_TIERS:
        return explicit.strip().lower()
    registry = source.get("source_registry")
    if isinstance(registry, dict):
        tier = registry.get("reliability_tier")
        if isinstance(tier, str) and tier.strip().lower() in RELIABILITY_TIERS:
            return tier.strip().lower()
        required = bool(source.get("required")) or bool(registry.get("required"))
        if registry.get("freshness_sla_seconds") is None:
            return "ad_hoc"
        try:
            rank = int(registry.get("authority_rank") or 1)
        except (TypeError, ValueError):
            rank = 1
        if required and rank == 1:
            return "critical"
        if required:
            return "standard"
        if rank >= 2:
            return "supplemental"
    required = bool(source.get("required"))
    return "critical" if required else "supplemental"


def gate_action_for_tier(tier: str, status: str) -> str:
    normalized_status = str(status or "missing").strip().lower()
    normalized_tier = str(tier or "standard").strip().lower()
    if normalized_tier == "critical" and normalized_status in {"stale", "failed", "missing"}:
        return "block"
    if normalized_tier == "ad_hoc":
        return "inform"
    if normalized_status in {"degraded", "stale", "failed", "missing"}:
        return "warn"
    return "ok"


def enrich_source_reliability(source: dict[str, Any]) -> dict[str, Any]:
    out = dict(source)
    tier = effective_reliability_tier_for_source(out)
    registry = out.get("source_registry")
    sla = sla_seconds_for_registry(registry if isinstance(registry, dict) else None)
    stale = bool(out.get("stale"))
    status = str(out.get("status") or "missing")
    out["reliability_tier"] = tier
    out["sla_seconds"] = sla
    out["sla_breach"] = stale and tier != "ad_hoc"
    out["gate_action"] = gate_action_for_tier(tier, status)
    return out


def tier_counts(sources: list[dict[str, Any]]) -> dict[str, int]:
    counts = {tier: 0 for tier in sorted(RELIABILITY_TIERS)}
    for source in sources:
        tier = str(source.get("reliability_tier") or effective_reliability_tier_for_source(source))
        if tier in counts:
            counts[tier] += 1
    return counts
