from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from ontology.sources.base import SourceAdapter, SourceResult, error_result, run_source_adapter
from ontology.sources.liquidity import LiquidityAdapter
from ontology.sources.macro import EconomicGrowthAdapter, LaborMarketAdapter, PositioningAdapter, SentimentAdapter
from ontology.sources.market_technicals import MarketBreadthAdapter, Top50BreadthAdapter, VixTermStructureAdapter
from ontology.sources.portfolio import PortfolioAdapter
from ontology.sources.sector_metrics import SectorMetricsAdapter

ADAPTER_TIMEOUT_SECONDS = 120
log = logging.getLogger(__name__)


def build_adapter_registry(
    *,
    timeframe: str,
) -> tuple[dict[str, SourceAdapter[Any]], dict[str, SourceAdapter[Any]], dict[str, SourceAdapter[Any]]]:
    required: dict[str, SourceAdapter[Any]] = {
        "portfolio": PortfolioAdapter(timeframe=timeframe),
        "market_breadth": MarketBreadthAdapter(),
        "top50_breadth": Top50BreadthAdapter(),
        "vix_term_structure": VixTermStructureAdapter(),
        "sector_metrics": SectorMetricsAdapter(),
        "liquidity": LiquidityAdapter(),
    }
    optional: dict[str, SourceAdapter[Any]] = {
        "sentiment": SentimentAdapter(),
        "positioning_summary": PositioningAdapter(),
        "economic_growth": EconomicGrowthAdapter(),
        "labor_market": LaborMarketAdapter(),
    }
    # Deep modules previously only populated source_status and did not contribute to graph construction.
    # They remain out of scope until source-specific normalized DTOs are introduced for them.
    deep: dict[str, SourceAdapter[Any]] = {}
    return required, optional, deep


def run_adapters(
    adapters: dict[str, SourceAdapter[Any]],
    *,
    provenance_parent_event_id: str | None = None,
    ontology_run_id: str | None = None,
) -> dict[str, SourceResult[Any]]:
    out: dict[str, SourceResult[Any]] = {}
    if not adapters:
        return out

    with ThreadPoolExecutor(max_workers=min(len(adapters), 10)) as pool:
        futures = {
            pool.submit(
                run_source_adapter,
                adapter,
                provenance_parent_event_id=provenance_parent_event_id,
                ontology_run_id=ontology_run_id,
            ): name
            for name, adapter in adapters.items()
        }
        try:
            for fut in as_completed(futures, timeout=ADAPTER_TIMEOUT_SECONDS):
                name = futures[fut]
                adapter = adapters[name]
                try:
                    out[name] = fut.result()
                except Exception as exc:
                    out[name] = error_result(adapter, str(exc))
        except FuturesTimeoutError:
            for fut, name in futures.items():
                if fut.done():
                    continue
                result = error_result(adapters[name], "module timed out")
                out[name] = result
                log.warning(
                    "ontology_source_adapter source=%s version=%s status=%s quality=%s duration_ms=%.1f as_of=%s "
                    "drift_count=%d detail=%s",
                    adapters[name].source_name,
                    adapters[name].source_version,
                    result.status,
                    result.quality,
                    ADAPTER_TIMEOUT_SECONDS * 1000.0,
                    result.as_of,
                    len(result.schema_drift),
                    result.detail,
                )
    return out


def source_status_from_results(results: dict[str, SourceResult[Any]]) -> dict[str, dict[str, Any]]:
    return {name: result.to_status_dict() for name, result in results.items()}
