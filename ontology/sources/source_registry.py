from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from api.snapshot_keys import (
    DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
    SNAPSHOT_ECONOMIC_GROWTH,
    SNAPSHOT_HOUSING,
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

LONG_CACHE_TTL_SECONDS = 60 * 60
DAILY_CACHE_TTL_SECONDS = 24 * 60 * 60

KNOWN_DATASET_DOMAINS = frozenset(
    {
        "document",
        "fundamental",
        "macro",
        "market",
        "news",
        "portfolio",
        "retrieval",
        "risk",
        "snapshots",
    }
)


@dataclass(frozen=True, slots=True)
class SourceRegistryEntry:
    source_id: str
    vendor_name: str
    dataset_domain: str
    authority_rank: int
    freshness_sla_seconds: int | None
    required: bool
    fallback_source_id: str | None = None
    reliability_tier: str | None = None
    snapshot_key: str | None = None
    raw_module: str | None = None
    raw_function: str | None = None

    def to_dict(self) -> dict[str, Any]:
        from ontology.sources.reliability import derive_reliability_tier

        data = asdict(self)
        data["reliability_tier"] = derive_reliability_tier(self)
        return data


def _entry(
    source_id: str,
    vendor_name: str,
    dataset_domain: str,
    authority_rank: int,
    freshness_sla_seconds: int | None,
    required: bool,
    *,
    fallback_source_id: str | None = None,
    reliability_tier: str | None = None,
    snapshot_key: str | None = None,
    raw_module: str | None = None,
    raw_function: str | None = None,
) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id=source_id,
        vendor_name=vendor_name,
        dataset_domain=dataset_domain,
        authority_rank=authority_rank,
        freshness_sla_seconds=freshness_sla_seconds,
        required=required,
        fallback_source_id=fallback_source_id,
        reliability_tier=reliability_tier,
        snapshot_key=snapshot_key,
        raw_module=raw_module,
        raw_function=raw_function,
    )


def _build_source_registry() -> dict[str, SourceRegistryEntry]:
    entries = [
        _entry(
            "portfolio",
            "yfinance",
            "portfolio",
            1,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            True,
            raw_module="portfolio.portfolio_dashboard",
            raw_function="get_data",
        ),
        _entry(
            "market_breadth",
            "yfinance",
            "market",
            1,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            True,
            snapshot_key=SNAPSHOT_MARKET_BREADTH,
            raw_module="equities.market_technicals.market_breadth",
            raw_function="get_data",
        ),
        _entry(
            "top50_breadth",
            "yfinance",
            "market",
            1,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            True,
            snapshot_key=SNAPSHOT_TOP50_BREADTH,
            raw_module="equities.market_technicals.top50_breadth",
            raw_function="get_data",
        ),
        _entry(
            "vix_term_structure",
            "yfinance",
            "market",
            1,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            True,
            snapshot_key=SNAPSHOT_VIX_TERM_STRUCTURE,
            raw_module="equities.market_technicals.vix_term_structure",
            raw_function="get_data",
        ),
        _entry(
            "sector_metrics",
            "yfinance",
            "market",
            1,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            True,
            snapshot_key=SNAPSHOT_SECTOR_METRICS,
            raw_module="equities.sector_metrics.sector_metrics",
            raw_function="get_data",
        ),
        _entry(
            "liquidity",
            "fred+ecb_sdmx+oecd",
            "macro",
            1,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            True,
            snapshot_key=SNAPSHOT_LIQUIDITY,
            raw_module="macro.liquidity.liquidity",
            raw_function="get_snapshot",
        ),
        _entry(
            "momentum",
            "yfinance",
            "market",
            2,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            False,
            snapshot_key=SNAPSHOT_MOMENTUM,
            raw_module="portfolio.momentum.price_momentum.momentum",
            raw_function="get_data",
        ),
        _entry(
            "market_regime",
            "internal",
            "market",
            2,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            True,
            snapshot_key=SNAPSHOT_SIGNAL_AGGREGATOR,
            raw_module="api.signal_aggregator",
            raw_function="build_signal_aggregator",
        ),
        _entry(
            "sentiment",
            "yfinance+survey_sources",
            "market",
            2,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            False,
            snapshot_key=SNAPSHOT_SENTIMENT,
            raw_module="macro.sentiment.sentiment",
            raw_function="get_put_call/get_surveys/get_volatility",
        ),
        _entry(
            "positioning_summary",
            "cftc",
            "risk",
            2,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            False,
            snapshot_key=SNAPSHOT_POSITIONING_SUMMARY,
            raw_module="macro.positioning.positioning",
            raw_function="fetch_multiple_instruments",
        ),
        _entry(
            "economic_growth",
            "yfinance+managed_crb_upload",
            "macro",
            2,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            False,
            snapshot_key=SNAPSHOT_ECONOMIC_GROWTH,
            raw_module="macro.economic_growth.economic_growth",
            raw_function="get_data",
        ),
        _entry(
            "labor_market",
            "fred",
            "macro",
            2,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            False,
            snapshot_key=SNAPSHOT_LABOR_MARKET,
            raw_module="macro.labor_market.labor_market",
            raw_function="get_data",
        ),
        _entry(
            "housing",
            "fred",
            "macro",
            2,
            DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
            False,
            snapshot_key=SNAPSHOT_HOUSING,
            raw_module="api.routers.housing",
            raw_function="load_housing_payload",
        ),
        _entry(
            "sec_edgar_companyfacts",
            "sec_edgar",
            "fundamental",
            1,
            DAILY_CACHE_TTL_SECONDS,
            True,
            fallback_source_id="yfinance_fundamentals",
            raw_module="portfolio.momentum.fundamental_momentum.edgar_fetcher",
            raw_function="fetch_companyfacts_by_cik",
        ),
        _entry(
            "sec_edgar_submissions",
            "sec_edgar",
            "fundamental",
            1,
            DAILY_CACHE_TTL_SECONDS,
            False,
            fallback_source_id="yfinance_fundamentals",
            raw_module="portfolio.momentum.fundamental_momentum.edgar_fetcher",
            raw_function="fetch_submissions_by_cik",
        ),
        _entry(
            "yfinance_fundamentals",
            "yfinance",
            "fundamental",
            2,
            LONG_CACHE_TTL_SECONDS,
            False,
            raw_module="yfinance",
            raw_function="Ticker",
        ),
        _entry(
            "financials",
            "sec_edgar",
            "fundamental",
            1,
            LONG_CACHE_TTL_SECONDS,
            True,
            fallback_source_id="yfinance_fundamentals",
            raw_module="portfolio.momentum.fundamental_momentum.financials_single",
            raw_function="get_data",
        ),
        _entry(
            "dcf_historical",
            "sec_edgar+yfinance",
            "fundamental",
            1,
            LONG_CACHE_TTL_SECONDS,
            True,
            fallback_source_id="yfinance_fundamentals",
            raw_module="equities.valuation.dcf",
            raw_function="get_historical_data",
        ),
        _entry(
            "dcf_valuation",
            "yfinance",
            "fundamental",
            2,
            LONG_CACHE_TTL_SECONDS,
            True,
            raw_module="equities.valuation.dcf",
            raw_function="run_valuation",
        ),
        _entry(
            "fundamental_momentum",
            "sec_edgar",
            "fundamental",
            1,
            LONG_CACHE_TTL_SECONDS,
            True,
            fallback_source_id="yfinance_fundamentals",
            raw_module="api.routers.fundamental_momentum",
            raw_function="_compute_fundamental_momentum",
        ),
        _entry(
            "portfolio_news_digest",
            "user_upload",
            "news",
            1,
            None,
            False,
            raw_module="portfolio.news_digests",
            raw_function="save_digest/list_digests/get_digest",
        ),
        _entry(
            "source_ingestion_document",
            "user_upload",
            "document",
            1,
            None,
            False,
            raw_module="ontology.source_ingestion",
            raw_function="upload_artifact",
        ),
        _entry(
            "source_ingestion_media",
            "user_upload",
            "document",
            1,
            None,
            False,
            raw_module="ontology.source_ingestion",
            raw_function="upload_artifact",
        ),
        _entry(
            "source_extraction",
            "internal",
            "document",
            2,
            None,
            False,
            raw_module="ontology.source_ingestion",
            raw_function="run_extractions",
        ),
        _entry(
            "retrieval_index",
            "internal",
            "retrieval",
            2,
            None,
            False,
            raw_module="api.retrieval",
            raw_function="index_document/search",
        ),
    ]
    return {entry.source_id: entry for entry in entries}


_SOURCE_REGISTRY = _build_source_registry()
_SNAPSHOT_TO_SOURCE_ID = {
    entry.snapshot_key: entry.source_id for entry in _SOURCE_REGISTRY.values() if entry.snapshot_key
}


def validate_source_registry(entries: Mapping[str, SourceRegistryEntry] | None = None) -> None:
    registry = dict(entries or _SOURCE_REGISTRY)
    errors: list[str] = []
    for source_id, entry in registry.items():
        if source_id != entry.source_id:
            errors.append(f"{source_id}: key must match source_id {entry.source_id}")
        if not entry.source_id.strip():
            errors.append(f"{source_id}: source_id is required")
        if not entry.vendor_name.strip():
            errors.append(f"{source_id}: vendor_name is required")
        if entry.dataset_domain not in KNOWN_DATASET_DOMAINS:
            errors.append(f"{source_id}: dataset_domain must be one of {sorted(KNOWN_DATASET_DOMAINS)}")
        if entry.authority_rank < 1:
            errors.append(f"{source_id}: authority_rank must be positive")
        if entry.freshness_sla_seconds is not None and entry.freshness_sla_seconds < 0:
            errors.append(f"{source_id}: freshness_sla_seconds must be non-negative")
        if entry.fallback_source_id == entry.source_id:
            errors.append(f"{source_id}: fallback_source_id cannot point to itself")
        if entry.fallback_source_id and entry.fallback_source_id not in registry:
            errors.append(f"{source_id}: fallback_source_id {entry.fallback_source_id} is not registered")
        if entry.reliability_tier is not None:
            from ontology.sources.reliability import RELIABILITY_TIERS

            tier = str(entry.reliability_tier).strip().lower()
            if tier not in RELIABILITY_TIERS:
                errors.append(f"{source_id}: reliability_tier must be one of {sorted(RELIABILITY_TIERS)}")
    if errors:
        raise ValueError("; ".join(errors))


validate_source_registry()


def all_source_registry_entries() -> dict[str, SourceRegistryEntry]:
    return dict(_SOURCE_REGISTRY)


def get_source_registry_entry(source_id: str | None) -> SourceRegistryEntry | None:
    if not source_id:
        return None
    return _SOURCE_REGISTRY.get(str(source_id).strip())


def source_id_for_snapshot(snapshot_key: str | None) -> str | None:
    if not snapshot_key:
        return None
    return _SNAPSHOT_TO_SOURCE_ID.get(str(snapshot_key).strip())


def get_source_registry_entry_for_snapshot(snapshot_key: str | None) -> SourceRegistryEntry | None:
    source_id = source_id_for_snapshot(snapshot_key)
    return get_source_registry_entry(source_id)


def source_registry_metadata(
    source_id: str | None,
    *,
    required: bool | None = None,
    freshness_sla_seconds: int | None = None,
    fallback_source_id: str | None = None,
) -> dict[str, Any] | None:
    entry = get_source_registry_entry(source_id)
    if entry is None:
        return None
    if required is not None:
        entry = replace(entry, required=required)
    if freshness_sla_seconds is not None:
        entry = replace(entry, freshness_sla_seconds=freshness_sla_seconds)
    if fallback_source_id is not None:
        entry = replace(entry, fallback_source_id=fallback_source_id)
    return entry.to_dict()


def source_registry_metadata_for_snapshot(snapshot_key: str | None) -> dict[str, Any] | None:
    entry = get_source_registry_entry_for_snapshot(snapshot_key)
    return entry.to_dict() if entry else None


def attach_source_registry_metadata(
    payload: dict[str, Any],
    *,
    source_id: str | None = None,
    snapshot_key: str | None = None,
    required: bool | None = None,
    freshness_sla_seconds: int | None = None,
) -> dict[str, Any]:
    metadata = (
        source_registry_metadata_for_snapshot(snapshot_key)
        if snapshot_key
        else source_registry_metadata(source_id, required=required, freshness_sla_seconds=freshness_sla_seconds)
    )
    if metadata is None:
        return payload
    out = copy.deepcopy(payload)
    raw_meta = out.get("_meta")
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    meta["source_registry"] = metadata
    out["_meta"] = meta
    return out


def attach_registry_to_status(
    source_id: str,
    status: dict[str, Any],
    *,
    snapshot_key: str | None = None,
    required: bool | None = None,
) -> dict[str, Any]:
    metadata = (
        source_registry_metadata_for_snapshot(snapshot_key)
        if snapshot_key
        else source_registry_metadata(source_id, required=required)
    )
    out = dict(status)
    if metadata is not None:
        out["source_registry"] = metadata
    return out
