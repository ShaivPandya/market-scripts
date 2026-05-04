"""Fast, persisted, quality-aware position risk refresh path."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from api.position_risk_store import (
    read_latest_portfolio_risk,
    read_latest_position_risk,
    write_portfolio_risk_snapshot,
    write_position_risk_snapshot,
)
from api.serializers import serialize_value
from api.snapshot_keys import (
    SNAPSHOT_ECONOMIC_GROWTH,
    SNAPSHOT_LABOR_MARKET,
    SNAPSHOT_LIQUIDITY,
    SNAPSHOT_MARKET_BREADTH,
    SNAPSHOT_POSITIONING_SUMMARY,
    SNAPSHOT_SCHEMA_VERSION,
    SNAPSHOT_SECTOR_METRICS,
    SNAPSHOT_SENTIMENT,
    SNAPSHOT_TOP50_BREADTH,
    SNAPSHOT_VIX_TERM_STRUCTURE,
)
from api.snapshot_store import SnapshotRecord, read_snapshot, write_snapshot_failure, write_snapshot_success
from ontology.risk import (
    W_BREADTH,
    W_MACRO,
    W_SECTOR,
    W_VOLATILITY,
    compute_breadth_stress,
    compute_macro_regime,
    compute_sector_stress_map,
    compute_volatility_cluster,
    risk_level,
    score_position,
)
from ontology.sector_mapper import SectorMapper
from ontology.sources.base import SourceAdapter, payload_fingerprint
from ontology.sources.liquidity import LiquidityAdapter
from ontology.sources.macro import EconomicGrowthAdapter, LaborMarketAdapter, PositioningAdapter, SentimentAdapter
from ontology.sources.market_technicals import MarketBreadthAdapter, Top50BreadthAdapter, VixTermStructureAdapter
from ontology.sources.sector_metrics import SectorMetricsAdapter

log = logging.getLogger("api.position_risk")

REQUIRED_MODULES = (
    "portfolio",
    "market_breadth",
    "top50_breadth",
    "vix_term_structure",
    "sector_metrics",
    "liquidity",
)
OPTIONAL_MODULES = (
    "sentiment",
    "positioning_summary",
    "economic_growth",
    "labor_market",
)
_EASTERN = ZoneInfo("America/New_York")
_AFTER_CLOSE_FRESHNESS_CUTOFF = time(hour=16, minute=15)


@dataclass(frozen=True, slots=True)
class ModuleConfig:
    name: str
    snapshot_key: str
    required: bool
    adapter_factory: Callable[[], SourceAdapter[Any]]


@dataclass(frozen=True, slots=True)
class RiskInputBundle:
    now: datetime
    source_status: dict[str, dict[str, Any]]
    module_data: dict[str, Any]
    input_snapshots: dict[str, Any]
    components: dict[str, Any]
    market_snapshot_as_of: str | None


_MODULES: dict[str, ModuleConfig] = {
    "market_breadth": ModuleConfig("market_breadth", SNAPSHOT_MARKET_BREADTH, True, MarketBreadthAdapter),
    "top50_breadth": ModuleConfig("top50_breadth", SNAPSHOT_TOP50_BREADTH, True, Top50BreadthAdapter),
    "vix_term_structure": ModuleConfig(
        "vix_term_structure", SNAPSHOT_VIX_TERM_STRUCTURE, True, VixTermStructureAdapter
    ),
    "sector_metrics": ModuleConfig("sector_metrics", SNAPSHOT_SECTOR_METRICS, True, SectorMetricsAdapter),
    "liquidity": ModuleConfig("liquidity", SNAPSHOT_LIQUIDITY, True, LiquidityAdapter),
    "sentiment": ModuleConfig("sentiment", SNAPSHOT_SENTIMENT, False, SentimentAdapter),
    "positioning_summary": ModuleConfig("positioning_summary", SNAPSHOT_POSITIONING_SUMMARY, False, PositioningAdapter),
    "economic_growth": ModuleConfig("economic_growth", SNAPSHOT_ECONOMIC_GROWTH, False, EconomicGrowthAdapter),
    "labor_market": ModuleConfig("labor_market", SNAPSHOT_LABOR_MARKET, False, LaborMarketAdapter),
}


def get_latest_position_risk(ticker: str) -> dict[str, Any] | None:
    return read_latest_position_risk(_ticker(ticker))


def get_latest_portfolio_risk() -> dict[str, Any] | None:
    return read_latest_portfolio_risk()


def refresh_position_risk(ticker: str) -> dict[str, Any]:
    if not risk_engine_enabled():
        latest = get_latest_position_risk(ticker)
        if latest is not None:
            return latest
    ticker_norm = _ticker(ticker)
    now = datetime.now(UTC)
    position = _load_portfolio_position(ticker_norm, now=now)
    bundle = load_global_risk_input_bundle(now=now)
    snapshot = _compute_position_snapshot(position, bundle)
    return write_position_risk_snapshot(snapshot)


def refresh_portfolio_risk() -> dict[str, Any]:
    """Refresh risk for all current positions using one shared global input bundle."""
    if not risk_engine_enabled():
        latest = get_latest_portfolio_risk()
        if latest is not None:
            return latest
    now = datetime.now(UTC)
    positions = _load_portfolio_positions(now=now)
    bundle = load_global_risk_input_bundle(now=now)

    portfolio_result_id = f"portfolio-risk:{uuid.uuid4().hex[:12]}"
    position_snapshots: list[dict[str, Any]] = []
    position_snapshot_ids: dict[str, str] = {}
    for position in positions:
        snapshot = _compute_position_snapshot(
            position,
            bundle,
            portfolio_risk_snapshot_id=portfolio_result_id,
        )
        persisted = write_position_risk_snapshot(snapshot)
        position_snapshots.append(persisted)
        position_snapshot_ids[str(persisted["ticker"])] = str(persisted["result_id"])

    scores = [_float_or_none(row.get("risk_score")) for row in position_snapshots]
    numeric_scores = [float(score) for score in scores if score is not None]
    average_score = round(sum(numeric_scores) / len(numeric_scores), 4) if numeric_scores else 0.0
    max_score = round(max(numeric_scores), 4) if numeric_scores else 0.0
    bucket_counts = _risk_bucket_counts(position_snapshots)
    source_status = {
        "portfolio": _portfolio_aggregate_source_status(positions, now),
        **{module: dict(state) for module, state in bundle.source_status.items()},
    }
    degraded_modules = _degraded_modules(source_status)
    confidence_values = [_float_or_none(row.get("confidence")) for row in position_snapshots]
    confidence = round(
        min([float(value) for value in confidence_values if value is not None] or [_confidence(source_status)]),
        2,
    )
    quality = (
        "ok" if not degraded_modules and all(row.get("quality") == "ok" for row in position_snapshots) else "degraded"
    )
    computed_at = now.isoformat()
    aggregate = {
        "exact": quality == "ok",
        "confidence": confidence,
        "position_count": len(position_snapshots),
        "average_risk_score": average_score,
        "max_risk_score": max_score,
        "risk_buckets": bucket_counts,
    }
    portfolio_snapshot = {
        "result_id": portfolio_result_id,
        "run_id": portfolio_result_id,
        "as_of": bundle.market_snapshot_as_of or computed_at,
        "computed_at": computed_at,
        "market_snapshot_as_of": bundle.market_snapshot_as_of,
        "freshness_policy": "market_day",
        "average_risk_score": average_score,
        "max_risk_score": max_score,
        "risk_score": average_score,
        "risk_level": risk_level(max_score),
        "confidence": confidence,
        "quality": quality,
        "risk_quality": quality,
        "risk_confidence": confidence,
        "position_count": len(position_snapshots),
        "risk_buckets": bucket_counts,
        "top_contributors": _top_risk_contributors(position_snapshots),
        "degraded_modules": degraded_modules,
        "missing_modules": [m["module"] for m in degraded_modules if m.get("status") in {"missing", "error"}],
        "stale_modules": [m["module"] for m in degraded_modules if m.get("status") == "stale"],
        "source_status": source_status,
        "input_snapshots": bundle.input_snapshots,
        "input_hash": payload_fingerprint(
            {
                "positions": positions,
                "input_snapshots": bundle.input_snapshots,
                "market_snapshot_as_of": bundle.market_snapshot_as_of,
            }
        ),
        "position_snapshot_ids": position_snapshot_ids,
        "position_snapshots": position_snapshots,
        "results": [
            {
                "ticker": row.get("ticker"),
                "asset": row.get("asset"),
                "direction": row.get("direction"),
                "sector": row.get("sector"),
                "risk_score": row.get("risk_score"),
                "risk_level": row.get("risk_level"),
                "risk_snapshot_id": row.get("result_id"),
                "quality": row.get("quality"),
                "confidence": row.get("confidence"),
                "evidence": row.get("evidence"),
            }
            for row in position_snapshots
        ],
        "aggregate": aggregate,
        "_meta": {
            "intent": "portfolio_risk_refresh",
            "required_modules": list(REQUIRED_MODULES),
            "optional_modules": list(OPTIONAL_MODULES),
            "position_snapshot_count": len(position_snapshots),
            "freshness_policy": {
                "name": "market_day",
                "timezone": "America/New_York",
                "after_close_cutoff": _AFTER_CLOSE_FRESHNESS_CUTOFF.strftime("%H:%M"),
            },
        },
    }
    return write_portfolio_risk_snapshot(portfolio_snapshot)


def load_global_risk_input_bundle(*, now: datetime | None = None) -> RiskInputBundle:
    """Load/heal all non-position risk inputs once for a risk refresh run."""
    run_now = now or datetime.now(UTC)
    source_status: dict[str, dict[str, Any]] = {}
    module_data: dict[str, Any] = {}
    input_snapshots: dict[str, Any] = {}

    for name in [*REQUIRED_MODULES[1:], *OPTIONAL_MODULES]:
        config = _MODULES[name]
        state, data = _load_module(config, now=run_now)
        source_status[name] = state
        if data is not None:
            module_data[name] = data
        if state.get("snapshot_key"):
            input_snapshots[name] = {
                "snapshot_key": state.get("snapshot_key"),
                "payload_hash": state.get("payload_hash"),
                "as_of": state.get("as_of"),
                "fetched_at": state.get("fetched_at"),
                "status": state.get("snapshot_status") or state.get("status"),
                "used": bool(state.get("used")),
                "accepted": bool(state.get("accepted")),
            }

    components = _compute_components(module_data)
    return RiskInputBundle(
        now=run_now,
        source_status=source_status,
        module_data=module_data,
        input_snapshots=input_snapshots,
        components=components,
        market_snapshot_as_of=_market_snapshot_as_of(source_status),
    )


def _compute_position_snapshot(
    position: dict[str, Any],
    bundle: RiskInputBundle,
    *,
    portfolio_risk_snapshot_id: str | None = None,
) -> dict[str, Any]:
    ticker_norm = _ticker(position.get("ticker"))
    source_status: dict[str, dict[str, Any]] = {
        "portfolio": _portfolio_source_status(position, bundle.now),
        **{module: dict(state) for module, state in bundle.source_status.items()},
    }
    input_snapshots = dict(bundle.input_snapshots)
    components = bundle.components
    sector_resolution = SectorMapper().resolve_sector(ticker_norm, str(position.get("asset") or "equity"))
    sector_scores = components["sector_scores"]
    sector_stress = float(sector_scores.get(sector_resolution.sector, sector_scores.get("Unknown Equity", 0.5)))

    score = score_position(
        volatility_cluster=components["volatility_cluster"],
        breadth_stress=components["breadth_stress"],
        sector_stress=sector_stress,
        macro_regime=components["macro_regime"],
    )
    level = risk_level(score)
    evidence = _risk_evidence(
        sector=sector_resolution.sector,
        volatility_cluster=components["volatility_cluster"],
        breadth_stress=components["breadth_stress"],
        sector_stress=sector_stress,
        macro_regime=components["macro_regime"],
    )

    degraded_modules = _degraded_modules(source_status)
    confidence = _confidence(source_status)
    quality = "ok" if not degraded_modules else "degraded"
    computed_at = bundle.now.isoformat()
    result_id = f"position-risk:{ticker_norm}:{uuid.uuid4().hex[:12]}"
    market_snapshot_as_of = bundle.market_snapshot_as_of

    snapshot = {
        "result_id": result_id,
        "run_id": result_id,
        "portfolio_risk_snapshot_id": portfolio_risk_snapshot_id,
        "ticker": ticker_norm,
        "as_of": market_snapshot_as_of or computed_at,
        "computed_at": computed_at,
        "market_snapshot_as_of": market_snapshot_as_of,
        "freshness_policy": "market_day",
        "risk_score": round(score, 4),
        "risk_level": level,
        "confidence": confidence,
        "quality": quality,
        "risk_quality": quality,
        "risk_confidence": confidence,
        "position": position,
        "asset": position.get("asset"),
        "direction": position.get("direction"),
        "sector": sector_resolution.sector,
        "sector_source": sector_resolution.source,
        "component_scores": {
            "volatility_cluster": round(components["volatility_cluster"], 4),
            "breadth_stress": round(components["breadth_stress"], 4),
            "sector_stress": round(sector_stress, 4),
            "macro_regime": round(components["macro_regime"], 4),
        },
        "evidence": evidence,
        "drivers": evidence,
        "degraded_modules": degraded_modules,
        "missing_modules": [m["module"] for m in degraded_modules if m.get("status") in {"missing", "error"}],
        "stale_modules": [m["module"] for m in degraded_modules if m.get("status") == "stale"],
        "source_status": source_status,
        "input_snapshots": input_snapshots,
        "input_hash": payload_fingerprint(
            {
                "position": position,
                "input_snapshots": input_snapshots,
                "component_scores": {
                    "volatility_cluster": round(components["volatility_cluster"], 4),
                    "breadth_stress": round(components["breadth_stress"], 4),
                    "sector_stress": round(sector_stress, 4),
                    "macro_regime": round(components["macro_regime"], 4),
                },
            }
        ),
        "results": [
            {
                "ticker": ticker_norm,
                "asset": position.get("asset"),
                "direction": position.get("direction"),
                "sector": sector_resolution.sector,
                "risk_score": round(score, 4),
                "risk_level": level,
                "evidence": evidence,
            }
        ],
        "aggregate": {
            "exact": quality == "ok",
            "confidence": confidence,
            "position_count": 1,
            "average_risk_score": round(score, 4),
            "risk_buckets": {
                "high": 1 if level == "high" else 0,
                "medium": 1 if level == "medium" else 0,
                "low": 1 if level == "low" else 0,
            },
        },
        "_meta": {
            "intent": "position_risk_refresh",
            "required_modules": list(REQUIRED_MODULES),
            "optional_modules": list(OPTIONAL_MODULES),
            "freshness_policy": {
                "name": "market_day",
                "timezone": "America/New_York",
                "after_close_cutoff": _AFTER_CLOSE_FRESHNESS_CUTOFF.strftime("%H:%M"),
            },
        },
    }
    return snapshot


def risk_engine_enabled() -> bool:
    return _flag_enabled("RISK_ENGINE_ENABLED", default=True)


def risk_compat_projections_enabled() -> bool:
    return _flag_enabled("RISK_COMPAT_PROJECTIONS_ENABLED", default=True)


def risk_recommendation_gate_enabled() -> bool:
    return _flag_enabled("RISK_RECOMMENDATION_GATE_ENABLED", default=False)


def _flag_enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _load_module(config: ModuleConfig, *, now: datetime) -> tuple[dict[str, Any], Any | None]:
    record = read_snapshot(config.snapshot_key)
    cached_state, cached_data = _evaluate_record(config, record, now=now)
    if cached_state.get("accepted"):
        return cached_state, cached_data

    if not config.required:
        cached_state["used"] = False
        return cached_state, None

    refreshed_state, refreshed_data = _refresh_module(config, now=now)
    if refreshed_state.get("accepted"):
        refreshed_state["refreshed"] = True
        return refreshed_state, refreshed_data

    fallback_data = cached_data if cached_data is not None and cached_state.get("scoring_fields_valid") else None
    state = dict(refreshed_state)
    state["required"] = config.required
    state["used"] = fallback_data is not None
    state["accepted"] = False
    state["fallback_used"] = fallback_data is not None
    if fallback_data is not None:
        state["fallback_reason"] = "targeted_refresh_failed; using prior non-fresh payload for degraded score"
        state["prior_snapshot"] = {
            "status": cached_state.get("status"),
            "quality": cached_state.get("quality"),
            "as_of": cached_state.get("as_of"),
            "fetched_at": cached_state.get("fetched_at"),
            "freshness": cached_state.get("freshness"),
            "payload_hash": cached_state.get("payload_hash"),
        }
    return state, fallback_data


def _evaluate_record(
    config: ModuleConfig,
    record: SnapshotRecord | None,
    *,
    now: datetime,
) -> tuple[dict[str, Any], Any | None]:
    adapter = config.adapter_factory()
    base_state: dict[str, Any] = {
        "status": "missing",
        "quality": "missing",
        "required": config.required,
        "source_name": config.name,
        "source_version": getattr(adapter, "source_version", "1"),
        "snapshot_key": config.snapshot_key,
        "accepted": False,
        "used": False,
        "scoring_fields_valid": False,
    }
    if record is None:
        base_state["detail"] = "snapshot not found"
        base_state["freshness"] = _freshness_state(None, now=now)
        return base_state, None
    if record.payload is None:
        state = {
            **base_state,
            "status": "error" if record.status == "error" else "missing",
            "snapshot_status": record.status,
            "error": record.error,
            "detail": record.error or "snapshot payload missing",
            "as_of": record.as_of_date,
            "fetched_at": record.fetched_at,
            "version": record.version,
            "freshness": _freshness_state(record.as_of_date or record.fetched_at, now=now),
            "scoring_fields_valid": False,
        }
        return state, None

    result = adapter.normalize(record.payload)
    valid, invalid_detail = _valid_for_scoring(config.name, record.payload, result.data)
    freshness = _freshness_state(record.as_of_date or result.as_of or record.fetched_at, now=now)
    status = result.status
    quality = result.quality
    detail = result.detail
    if record.status != "ok":
        status = "error"
        quality = "degraded" if result.data is not None else "missing"
        detail = record.error or result.detail or "snapshot refresh previously failed"
    elif not freshness["fresh"]:
        status = "stale"
        quality = "degraded"
        detail = freshness.get("reason") or "snapshot is stale"
    elif not valid:
        status = "error"
        quality = "missing"
        detail = invalid_detail

    accepted = status == "ok" and valid and bool(freshness["fresh"])
    state = {
        **result.to_status_dict(),
        "status": status,
        "quality": quality,
        "required": config.required,
        "snapshot_key": config.snapshot_key,
        "snapshot_status": record.status,
        "as_of": record.as_of_date or result.as_of,
        "fetched_at": record.fetched_at,
        "version": record.version,
        "freshness": freshness,
        "accepted": accepted,
        "used": accepted,
        "scoring_fields_valid": valid,
        "payload_hash": payload_fingerprint(record.payload),
    }
    if record.error:
        state["error"] = record.error
    if detail:
        state["detail"] = detail
    if invalid_detail and status == "ok":
        state["detail"] = invalid_detail
    return state, result.data


def _refresh_module(config: ModuleConfig, *, now: datetime) -> tuple[dict[str, Any], Any | None]:
    adapter = config.adapter_factory()
    try:
        raw = adapter.fetch()
        payload = _snapshot_payload(config.name, serialize_value(raw))
        result = adapter.normalize(payload)
        valid, invalid_detail = _valid_for_scoring(config.name, payload, result.data)
        if result.status == "ok" and valid:
            record = write_snapshot_success(
                config.snapshot_key,
                payload,
                as_of_date=_payload_as_of(payload) or result.as_of,
                version=SNAPSHOT_SCHEMA_VERSION,
            )
            return _evaluate_record(config, record, now=now)

        detail = result.detail or invalid_detail or f"{config.name} refresh returned {result.status}"
        record = write_snapshot_failure(config.snapshot_key, detail, version=SNAPSHOT_SCHEMA_VERSION)
        state, _ = _evaluate_record(config, record, now=now)
        state["refreshed"] = True
        state["detail"] = detail
        state["status"] = "error" if result.status == "error" else "partial"
        state["accepted"] = False
        state["used"] = False
        return state, None
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        log.warning("position risk targeted refresh failed for %s: %s", config.name, detail, exc_info=True)
        record = write_snapshot_failure(config.snapshot_key, detail, version=SNAPSHOT_SCHEMA_VERSION)
        state, _ = _evaluate_record(config, record, now=now)
        state["refreshed"] = True
        state["detail"] = detail
        state["status"] = "error"
        state["accepted"] = False
        state["used"] = False
        return state, None


def _compute_components(module_data: dict[str, Any]) -> dict[str, Any]:
    volatility_cluster, _ = compute_volatility_cluster(
        module_data.get("vix_term_structure"),
        module_data.get("sentiment"),
    )
    breadth_stress, _ = compute_breadth_stress(
        module_data.get("market_breadth"),
        module_data.get("top50_breadth"),
    )
    sector_scores, _ = compute_sector_stress_map(module_data.get("sector_metrics"))
    macro_regime, _ = compute_macro_regime(
        liquidity=module_data.get("liquidity"),
        positioning=module_data.get("positioning_summary"),
        economic_growth=module_data.get("economic_growth"),
        labor_market=module_data.get("labor_market"),
    )
    return {
        "volatility_cluster": float(volatility_cluster),
        "breadth_stress": float(breadth_stress),
        "sector_scores": sector_scores,
        "macro_regime": float(macro_regime),
    }


def _risk_evidence(
    *,
    sector: str,
    volatility_cluster: float,
    breadth_stress: float,
    sector_stress: float,
    macro_regime: float,
) -> list[dict[str, Any]]:
    rows = [
        {
            "component": "volatility_cluster",
            "source": "vix_term_structure",
            "name": "Volatility Cluster",
            "value": round(volatility_cluster, 4),
            "threshold": "higher => more stress",
            "direction": "deteriorating" if volatility_cluster >= 0.6 else "stable",
            "contribution": round(W_VOLATILITY * volatility_cluster, 4),
        },
        {
            "component": "breadth_stress",
            "source": "market_breadth",
            "name": "Breadth Stress",
            "value": round(breadth_stress, 4),
            "threshold": "higher => weaker participation",
            "direction": "deteriorating" if breadth_stress >= 0.6 else "stable",
            "contribution": round(W_BREADTH * breadth_stress, 4),
        },
        {
            "component": "sector_stress",
            "source": "sector_metrics",
            "name": f"{sector} Sector Stress",
            "value": round(sector_stress, 4),
            "threshold": "higher => weaker sector backdrop",
            "direction": "deteriorating" if sector_stress >= 0.6 else "stable",
            "contribution": round(W_SECTOR * sector_stress, 4),
        },
        {
            "component": "macro_regime",
            "source": "liquidity",
            "name": "Macro Regime",
            "value": round(macro_regime, 4),
            "threshold": "higher => tighter macro conditions",
            "direction": "deteriorating" if macro_regime >= 0.6 else "stable",
            "contribution": round(W_MACRO * macro_regime, 4),
        },
    ]
    rows.sort(key=lambda row: float(row.get("contribution") or 0), reverse=True)
    return rows


def _load_portfolio_position(ticker: str, *, now: datetime) -> dict[str, Any]:
    from api.exceptions import NotFoundError

    for row in _load_portfolio_positions(now=now):
        if _ticker(row.get("ticker")) != ticker:
            continue
        return row
    raise NotFoundError("Portfolio position", ticker)


def _load_portfolio_positions(*, now: datetime) -> list[dict[str, Any]]:
    from portfolio.portfolio_db import get_positions

    rows: list[dict[str, Any]] = []
    for raw in get_positions(include_hedges=False):
        ticker = _ticker(raw.get("ticker"))
        if not ticker:
            continue
        rows.append(
            {
                "ticker": ticker,
                "asset": str(raw.get("asset") or "equity").strip().lower(),
                "direction": str(raw.get("direction") or "long").strip().lower(),
                "shares": _float_or_none(raw.get("shares")),
                "cost_basis": _float_or_none(raw.get("cost_basis")),
                "conviction": _int_or_none(raw.get("conviction")),
                "contrarian": bool(raw.get("contrarian")),
                "role": str(raw.get("role") or "position"),
                "as_of": now.isoformat(),
            }
        )
    return rows


def _portfolio_source_status(position: dict[str, Any], now: datetime) -> dict[str, Any]:
    return {
        "status": "ok",
        "quality": "ok",
        "required": True,
        "source_name": "portfolio",
        "source_version": "1",
        "as_of": position.get("as_of") or now.isoformat(),
        "fetched_at": now.isoformat(),
        "accepted": True,
        "used": True,
        "freshness": {
            "policy": "request_time",
            "fresh": True,
            "basis": "portfolio_db",
            "observed_as_of_date": str(now.astimezone(_EASTERN).date()),
        },
    }


def _portfolio_aggregate_source_status(positions: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    return {
        "status": "ok",
        "quality": "ok",
        "required": True,
        "source_name": "portfolio",
        "source_version": "1",
        "as_of": now.isoformat(),
        "fetched_at": now.isoformat(),
        "accepted": True,
        "used": True,
        "position_count": len(positions),
        "freshness": {
            "policy": "request_time",
            "fresh": True,
            "basis": "portfolio_db",
            "observed_as_of_date": str(now.astimezone(_EASTERN).date()),
        },
    }


def _risk_bucket_counts(snapshots: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for row in snapshots:
        level = str(row.get("risk_level") or "low").lower()
        if level not in counts:
            level = "low"
        counts[level] += 1
    return counts


def _top_risk_contributors(snapshots: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    rows = sorted(
        snapshots,
        key=lambda row: (_float_or_none(row.get("risk_score")) or 0.0, str(row.get("ticker") or "")),
        reverse=True,
    )
    contributors: list[dict[str, Any]] = []
    for row in rows[:limit]:
        evidence = row.get("evidence") if isinstance(row.get("evidence"), list) else []
        top_driver = evidence[0] if evidence and isinstance(evidence[0], dict) else None
        contributors.append(
            {
                "ticker": row.get("ticker"),
                "asset": row.get("asset"),
                "direction": row.get("direction"),
                "sector": row.get("sector"),
                "risk_score": row.get("risk_score"),
                "risk_level": row.get("risk_level"),
                "risk_snapshot_id": row.get("result_id"),
                "quality": row.get("quality"),
                "confidence": row.get("confidence"),
                "top_driver": top_driver,
            }
        )
    return contributors


def _valid_for_scoring(module_name: str, payload: Any, data: Any) -> tuple[bool, str | None]:
    if data is None:
        return False, "normalized module data is missing"
    if module_name == "market_breadth":
        fields = ("pct_above_200dma", "pct_above_20dma", "pct_at_20day_low", "pct_at_52wk_low")
        return _has_attr_values(data, fields), "market breadth scoring fields are missing"
    if module_name == "top50_breadth":
        fields = ("pct_below_50dma", "pct_3plus_dist", "pct_broke_20low")
        return _has_attr_values(data, fields), "top50 breadth scoring fields are missing"
    if module_name == "vix_term_structure":
        return (
            _get_attr(data, "ratio") is not None or _get_attr(data, "vix") is not None,
            "VIX term structure latest row is missing ratio and VIX",
        )
    if module_name == "sector_metrics":
        rows = _get_attr(data, "rows")
        return bool(rows), "sector metric rows are missing"
    if module_name == "liquidity":
        if isinstance(payload, dict) and payload.get("regime") is None:
            return False, "liquidity regime is missing"
        return bool(str(_get_attr(data, "regime") or "").strip()), "liquidity regime is missing"
    return True, None


def _freshness_state(value: Any, *, now: datetime) -> dict[str, Any]:
    observed = _observed_date(value)
    expected = _expected_market_date(now)
    if observed is None:
        return {
            "policy": "market_day",
            "fresh": False,
            "basis": "as_of_or_fetched_at",
            "expected_market_date": expected.isoformat(),
            "observed_as_of_date": None,
            "reason": "snapshot has no parseable as-of date",
        }
    fresh = observed >= expected
    return {
        "policy": "market_day",
        "fresh": fresh,
        "basis": "as_of_or_fetched_at",
        "expected_market_date": expected.isoformat(),
        "observed_as_of_date": observed.isoformat(),
        "reason": None
        if fresh
        else f"snapshot as-of {observed.isoformat()} is older than required market day {expected.isoformat()}",
    }


def _expected_market_date(now: datetime) -> date:
    local = now.astimezone(_EASTERN)
    current = local.date()
    if local.weekday() >= 5:
        return _previous_business_day(current)
    if local.time() < _AFTER_CLOSE_FRESHNESS_CUTOFF:
        return _previous_business_day(current)
    return current


def _previous_business_day(value: date) -> date:
    cur = value - timedelta(days=1)
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def _observed_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


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


def _snapshot_payload(module_name: str, payload: Any) -> Any:
    if module_name == "positioning_summary" and isinstance(payload, list):
        return {"rows": payload}
    return payload


def _degraded_modules(source_status: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for module, state in source_status.items():
        status = str(state.get("status") or "missing").lower()
        freshness = state.get("freshness") if isinstance(state.get("freshness"), dict) else {}
        if status == "ok" and state.get("accepted", True) and freshness.get("fresh", True):
            continue
        rows.append(
            {
                "module": module,
                "required": bool(state.get("required")),
                "status": status,
                "quality": state.get("quality"),
                "reason": state.get("detail") or state.get("error") or freshness.get("reason") or "module unavailable",
                "as_of": state.get("as_of"),
                "fetched_at": state.get("fetched_at"),
            }
        )
    rows.sort(key=lambda row: (not row["required"], row["module"]))
    return rows


def _confidence(source_status: dict[str, dict[str, Any]]) -> float:
    confidence = 1.0
    for module, state in source_status.items():
        if module == "portfolio":
            continue
        accepted = bool(state.get("accepted"))
        status = str(state.get("status") or "missing").lower()
        required = bool(state.get("required"))
        if accepted and status == "ok":
            continue
        if required:
            confidence -= 0.12 if state.get("used") else 0.18
        else:
            confidence -= 0.03
    return round(max(0.25, min(1.0, confidence)), 2)


def _market_snapshot_as_of(source_status: dict[str, dict[str, Any]]) -> str | None:
    observed: list[str] = []
    for module in REQUIRED_MODULES[1:]:
        state = source_status.get(module) or {}
        if state.get("used") and state.get("as_of"):
            observed.append(str(state["as_of"]))
    return min(observed) if observed else None


def _has_attr_values(value: Any, fields: tuple[str, ...]) -> bool:
    return any(_get_attr(value, field) is not None for field in fields)


def _get_attr(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
