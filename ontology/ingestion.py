"""Ingest source data into authoritative temporal ontology versions.

During the migration window this module also persists legacy snapshot runs as a
compatibility artifact for existing query paths.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any, cast

from api.postgres import use_postgres_state
from ontology.domain_write_service import ontology_primary_writes_enabled, ontology_read_model_enabled
from ontology.models import OntologyEdge, OntologyNode
from ontology.object_service import OntologyObjectService
from ontology.read_model import TemporalReadModelRepository
from ontology.repository import OntologyRepository
from ontology.risk import (
    compute_breadth_stress,
    compute_macro_regime,
    compute_sector_stress_map,
    compute_volatility_cluster,
    risk_level,
    score_position,
)
from ontology.schemas.registry import normalize_graph
from ontology.schemas.relations import (
    AFFECTED_BY,
    BELONGS_TO_SECTOR,
    EMITS_SIGNAL,
    EVALUATED_BY,
    EXPOSED_TO_SIGNAL,
    HAS_CATALYST,
    HAS_THESIS,
    REFERENCES_ASSET,
)
from ontology.sector_mapper import SectorMapper
from ontology.source_records import write_source_result_records
from ontology.sources.dtos import (
    EconomicGrowthSnapshot,
    LaborMarketSnapshot,
    LiquiditySnapshot,
    MarketBreadthSnapshot,
    PortfolioSnapshot,
    PositioningSnapshot,
    SectorMetricsSnapshot,
    SentimentSnapshot,
    Top50BreadthSnapshot,
    VixTermStructureSnapshot,
)
from ontology.sources.registry import build_adapter_registry, run_adapters, source_status_from_results
from ontology.temporal_repository import TemporalOntologyRepository

SNAPSHOT_RETENTION_DAYS = 90
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionOutput:
    run_id: str
    as_of: str
    source_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    required_modules: list[str] = field(default_factory=list)
    optional_modules: list[str] = field(default_factory=list)
    component_scores: dict[str, float] = field(default_factory=dict)
    provenance_event_id: str | None = None


def ingest_into_repository(
    repo: OntologyRepository,
    timeframe: str,
    include_deep_modules: bool,
) -> IngestionOutput:
    """Build and persist one materialized ontology snapshot run."""
    run_id = datetime.now(UTC).isoformat()
    source_status: dict[str, dict[str, Any]] = {}
    provenance_event_id: str | None = None
    try:
        from api import provenance

        provenance_event_id = provenance.deterministic_id("pv:ontology_run", run_id)
        provenance.start_event(
            event_id=provenance_event_id,
            event_type="ontology_run",
            event_name="ontology.ingest",
            ontology_run_id=run_id,
            summary={
                "run_id": run_id,
                "timeframe": timeframe,
                "include_deep_modules": include_deep_modules,
            },
            metadata={"retention_days": SNAPSHOT_RETENTION_DAYS},
        )
    except Exception:
        provenance_event_id = None

    required_adapters, optional_adapters, deep_adapters = build_adapter_registry(timeframe=timeframe)

    source_results = _run_adapters_with_provenance(
        {**required_adapters, **optional_adapters},
        provenance_parent_event_id=provenance_event_id,
        ontology_run_id=run_id,
    )
    source_status.update(source_status_from_results(source_results))
    if include_deep_modules and deep_adapters:
        deep_results = _run_adapters_with_provenance(
            deep_adapters,
            provenance_parent_event_id=provenance_event_id,
            ontology_run_id=run_id,
        )
        source_results.update(deep_results)
        source_status.update(source_status_from_results(deep_results))
    _link_source_adapter_events(source_results, provenance_event_id, run_id)
    _record_source_record_refs(source_results)
    temporal_repo: TemporalOntologyRepository | None = None
    if use_postgres_state():
        temporal_repo = TemporalOntologyRepository()
        _record_temporal_source_versions(source_results, repository=temporal_repo)

    nodes: dict[str, OntologyNode] = {}
    edges: dict[tuple[str, str, str], OntologyEdge] = {}

    def add_node(node: OntologyNode) -> None:
        nodes[node.id] = node

    def add_edge(edge: OntologyEdge) -> None:
        edges[(edge.source_id, edge.target_id, edge.relation_type)] = edge

    def ensure_sector_node(sector_name: str, sector_source: str) -> str:
        sector_label = str(sector_name or "Unknown Equity").strip() or "Unknown Equity"
        sector_id = f"sector:{_slug(sector_label)}"
        if sector_id not in nodes:
            add_node(
                OntologyNode(
                    id=sector_id,
                    type="Sector",
                    label=sector_label,
                    properties={
                        "name": sector_label,
                        "sector_source": str(sector_source or "unknown").strip() or "unknown",
                    },
                )
            )
        return sector_id

    portfolio = _source_data(source_results, "portfolio")
    if not isinstance(portfolio, PortfolioSnapshot):
        portfolio = PortfolioSnapshot(positions={}, timeframe=timeframe, timestamp=None)
    sector_mapper = SectorMapper()
    position_ids: list[str] = []

    # Core entity graph: Position -> Asset -> Sector
    portfolio_timestamp = portfolio.timestamp

    for ticker_norm, position in portfolio.positions.items():
        ticker_norm = str(ticker_norm).strip().upper()
        if not ticker_norm:
            continue

        asset_class = str(position.asset or "unknown").strip().lower()
        direction = str(position.direction or "unknown").strip().lower()

        position_id = f"position:{ticker_norm}"
        asset_id = f"asset:{ticker_norm}"
        latest_price = position.latest_price

        sector = sector_mapper.resolve_sector(ticker_norm, asset_class)
        sector_id = ensure_sector_node(sector.sector, sector.source)

        add_node(
            OntologyNode(
                id=position_id,
                type="Position",
                label=ticker_norm,
                properties={
                    "ticker": ticker_norm,
                    "asset": asset_class,
                    "direction": direction,
                    "instrument_type": getattr(position, "instrument_type", "security"),
                    "price_symbol": getattr(position, "price_symbol", None) or ticker_norm,
                    "quantity": getattr(position, "quantity", None),
                    "contract_multiplier": getattr(position, "contract_multiplier", 1.0),
                    "latest_price": latest_price,
                    "timeframe": timeframe,
                    "as_of": portfolio_timestamp,
                    "ontology_run_id": run_id,
                },
            )
        )
        add_node(
            OntologyNode(
                id=asset_id,
                type="Asset",
                label=ticker_norm,
                properties={
                    "ticker": ticker_norm,
                    "asset": asset_class,
                    "instrument_type": getattr(position, "instrument_type", "security"),
                    "price_symbol": getattr(position, "price_symbol", None) or ticker_norm,
                },
            )
        )
        add_edge(
            OntologyEdge(
                source_id=position_id,
                target_id=asset_id,
                relation_type=REFERENCES_ASSET,
                properties={"ontology_run_id": run_id},
            )
        )
        add_edge(
            OntologyEdge(
                source_id=asset_id,
                target_id=sector_id,
                relation_type=BELONGS_TO_SECTOR,
                properties={"source": sector.source, "ontology_run_id": run_id},
            )
        )
        position_ids.append(position_id)

    # Thesis + Evaluation entities: Position -> has_thesis -> Thesis
    _ingest_thesis_entities(add_node, add_edge, run_id, position_ids)

    # Compute global component scores from module outputs
    vix_data = _source_data(source_results, "vix_term_structure")
    breadth_data = _source_data(source_results, "market_breadth")
    top50_data = _source_data(source_results, "top50_breadth")
    sector_metrics_data = _source_data(source_results, "sector_metrics")
    liquidity_data = _source_data(source_results, "liquidity")

    sentiment_data = _source_data(source_results, "sentiment")
    positioning_data = _source_data(source_results, "positioning_summary")
    economic_growth_data = _source_data(source_results, "economic_growth")
    labor_market_data = _source_data(source_results, "labor_market")

    volatility_cluster, vol_evidence = compute_volatility_cluster(vix_data, sentiment_data)
    breadth_stress, breadth_evidence = compute_breadth_stress(breadth_data, top50_data)
    sector_scores, sector_evidence = compute_sector_stress_map(sector_metrics_data)
    macro_regime, macro_evidence = compute_macro_regime(
        liquidity=liquidity_data,
        positioning=positioning_data,
        economic_growth=economic_growth_data,
        labor_market=labor_market_data,
    )

    # Add macro indicators + global signals
    global_signal_ids = _add_indicator_signals(
        add_node=add_node,
        add_edge=add_edge,
        run_id=run_id,
        indicator_id="macro_indicator:vix_term_structure",
        indicator_label="VIX Term Structure",
        evidence=vol_evidence,
    )
    breadth_signal_ids = _add_indicator_signals(
        add_node=add_node,
        add_edge=add_edge,
        run_id=run_id,
        indicator_id="macro_indicator:market_breadth",
        indicator_label="Market Breadth",
        evidence=breadth_evidence,
    )
    macro_signal_ids = _add_indicator_signals(
        add_node=add_node,
        add_edge=add_edge,
        run_id=run_id,
        indicator_id="macro_indicator:macro_regime",
        indicator_label="Macro Regime",
        evidence=macro_evidence,
    )

    sector_signal_by_name: dict[str, str] = {}
    add_node(
        OntologyNode(
            id="macro_indicator:sector_metrics",
            type="MacroIndicator",
            label="Sector Metrics",
            properties={"as_of": run_id, "ontology_run_id": run_id},
        )
    )

    for item in sector_evidence:
        sector_name = str(item.get("sector") or "Unknown Equity")
        sector_id = ensure_sector_node(sector_name, "sector_metrics")
        signal_id = f"signal:sector_metrics:{_slug(sector_name)}"
        sector_signal_by_name[sector_name] = signal_id
        add_node(
            OntologyNode(
                id=signal_id,
                type="Signal",
                label=f"Sector Stress: {sector_name}",
                properties={
                    **item,
                    "component": "sector_stress",
                    "ontology_run_id": run_id,
                },
            )
        )
        add_edge(
            OntologyEdge(
                source_id="macro_indicator:sector_metrics",
                target_id=signal_id,
                relation_type=EMITS_SIGNAL,
                properties={"ontology_run_id": run_id},
            )
        )
        add_edge(
            OntologyEdge(
                source_id=sector_id,
                target_id="macro_indicator:sector_metrics",
                relation_type=AFFECTED_BY,
                properties={"ontology_run_id": run_id},
            )
        )

    # Attach risk to each position and create position->signal exposure edges
    for position_id in position_ids:
        node = nodes[position_id]
        props = dict(node.properties)
        sector_name = _resolve_sector_name_from_edges(edges, position_id, nodes)
        sector_stress = sector_scores.get(sector_name, sector_scores.get("Unknown Equity", 0.5))
        if sector_name not in sector_signal_by_name:
            signal_id = f"signal:sector_metrics:{_slug(sector_name)}"
            sector_signal_by_name[sector_name] = signal_id
            sector_id = ensure_sector_node(sector_name, "sector_metrics")
            add_node(
                OntologyNode(
                    id=signal_id,
                    type="Signal",
                    label=f"Sector Stress: {sector_name}",
                    properties={
                        "name": f"{sector_name} sector stress",
                        "source": "sector_metrics",
                        "value": round(sector_stress, 4),
                        "threshold": "higher => weaker sector backdrop",
                        "direction": "deteriorating" if sector_stress >= 0.6 else "stable",
                        "raw_signal": "deteriorating" if sector_stress >= 0.6 else "stable",
                        "sector": sector_name,
                        "component": "sector_stress",
                        "ontology_run_id": run_id,
                    },
                )
            )
            add_edge(
                OntologyEdge(
                    source_id="macro_indicator:sector_metrics",
                    target_id=signal_id,
                    relation_type=EMITS_SIGNAL,
                    properties={"ontology_run_id": run_id},
                )
            )
            add_edge(
                OntologyEdge(
                    source_id=sector_id,
                    target_id="macro_indicator:sector_metrics",
                    relation_type=AFFECTED_BY,
                    properties={"ontology_run_id": run_id},
                )
            )

        risk_score = score_position(
            volatility_cluster=volatility_cluster,
            breadth_stress=breadth_stress,
            sector_stress=sector_stress,
            macro_regime=macro_regime,
        )

        contributions = [
            {
                "signal_id": global_signal_ids[0] if global_signal_ids else "signal:vix_term_structure:core",
                "component": "volatility_cluster",
                "source": "vix_term_structure",
                "name": "Volatility Cluster",
                "value": volatility_cluster,
                "threshold": "higher => more stress",
                "direction": "deteriorating" if volatility_cluster >= 0.6 else "stable",
                "contribution": round(0.35 * volatility_cluster, 4),
            },
            {
                "signal_id": breadth_signal_ids[0] if breadth_signal_ids else "signal:market_breadth:core",
                "component": "breadth_stress",
                "source": "market_breadth",
                "name": "Breadth Stress",
                "value": breadth_stress,
                "threshold": "higher => weaker participation",
                "direction": "deteriorating" if breadth_stress >= 0.6 else "stable",
                "contribution": round(0.25 * breadth_stress, 4),
            },
            {
                "signal_id": sector_signal_by_name.get(sector_name, "signal:sector_metrics:unknown"),
                "component": "sector_stress",
                "source": "sector_metrics",
                "name": f"{sector_name} Sector Stress",
                "value": sector_stress,
                "threshold": "higher => weaker sector backdrop",
                "direction": "deteriorating" if sector_stress >= 0.6 else "stable",
                "contribution": round(0.25 * sector_stress, 4),
            },
            {
                "signal_id": macro_signal_ids[0] if macro_signal_ids else "signal:macro_regime:core",
                "component": "macro_regime",
                "source": "liquidity",
                "name": "Macro Regime",
                "value": macro_regime,
                "threshold": "higher => tighter macro conditions",
                "direction": "deteriorating" if macro_regime >= 0.6 else "stable",
                "contribution": round(0.15 * macro_regime, 4),
            },
        ]

        contributions.sort(key=lambda r: _to_float(r.get("contribution")) or 0.0, reverse=True)
        compact = contributions[:4]

        # Update position node with scored fields
        props.update(
            {
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level(risk_score),
                "volatility_cluster": round(volatility_cluster, 4),
                "breadth_stress": round(breadth_stress, 4),
                "sector_stress": round(sector_stress, 4),
                "macro_regime": round(macro_regime, 4),
                "ontology_run_id": run_id,
            }
        )
        add_node(
            OntologyNode(
                id=position_id,
                type="Position",
                label=node.label,
                properties=props,
            )
        )

        for ev in compact:
            add_edge(
                OntologyEdge(
                    source_id=position_id,
                    target_id=str(ev["signal_id"]),
                    relation_type=EXPOSED_TO_SIGNAL,
                    properties={
                        "component": ev["component"],
                        "source": ev["source"],
                        "name": ev["name"],
                        "value": ev["value"],
                        "threshold": ev["threshold"],
                        "direction": ev["direction"],
                        "contribution": ev["contribution"],
                        "ontology_run_id": run_id,
                    },
                )
            )

    normalized_graph = normalize_graph(
        list(nodes.values()),
        list(edges.values()),
        run_id=run_id,
        allow_legacy=True,
        skip_optional_invalid=True,
    )
    if normalized_graph.warnings:
        source_status["thesis_entities"] = {
            "status": "partial",
            "detail": "; ".join(normalized_graph.warnings[:3]),
        }
    snapshot_nodes = normalized_graph.nodes
    snapshot_edges = normalized_graph.edges

    as_of = str(portfolio_timestamp or datetime.now(UTC).isoformat())
    required_modules = list(required_adapters.keys())
    optional_modules = list(optional_adapters.keys()) + (list(deep_adapters.keys()) if include_deep_modules else [])
    component_scores = {
        "volatility_cluster": round(volatility_cluster, 4),
        "breadth_stress": round(breadth_stress, 4),
        "macro_regime": round(macro_regime, 4),
    }

    if temporal_repo is not None:
        _write_temporal_graph_versions(
            nodes=snapshot_nodes,
            edges=snapshot_edges,
            as_of=as_of,
            provenance_event_id=provenance_event_id,
            repository=temporal_repo,
        )
        _refresh_temporal_read_models_after_ingestion()

    repo.save_snapshot(
        run_id=run_id,
        as_of=as_of,
        source_status=source_status,
        required_modules=required_modules,
        optional_modules=optional_modules,
        component_scores=component_scores,
        nodes=snapshot_nodes,
        edges=snapshot_edges,
    )
    repo.prune_runs_older_than(days=SNAPSHOT_RETENTION_DAYS)
    if provenance_event_id:
        try:
            from api import provenance

            provenance.finish_event(
                provenance_event_id,
                status="succeeded",
                output_value={
                    "run_id": run_id,
                    "node_count": len(snapshot_nodes),
                    "edge_count": len(snapshot_edges),
                    "source_status": source_status,
                },
                summary={
                    "run_id": run_id,
                    "as_of": as_of,
                    "node_count": len(snapshot_nodes),
                    "edge_count": len(snapshot_edges),
                    "required_module_count": len(required_modules),
                    "optional_module_count": len(optional_modules),
                },
                metadata={"component_scores": component_scores},
            )
        except Exception:
            pass

    return IngestionOutput(
        run_id=run_id,
        as_of=as_of,
        source_status=source_status,
        required_modules=required_modules,
        optional_modules=optional_modules,
        component_scores=component_scores,
        provenance_event_id=provenance_event_id,
    )


def _run_adapters_with_provenance(
    adapters: dict[str, Any],
    *,
    provenance_parent_event_id: str | None,
    ontology_run_id: str,
) -> dict[str, Any]:
    try:
        return run_adapters(
            adapters,
            provenance_parent_event_id=provenance_parent_event_id,
            ontology_run_id=ontology_run_id,
        )
    except TypeError:
        # Compatibility with tests or external monkeypatches using the old signature.
        return run_adapters(adapters)


def _link_source_adapter_events(
    source_results: dict[str, Any],
    ontology_event_id: str | None,
    ontology_run_id: str,
) -> None:
    if not ontology_event_id:
        return
    try:
        from api import provenance
    except Exception:
        return
    for source_name, result in source_results.items():
        adapter_event_id = getattr(getattr(result, "lineage", None), "provenance_event_id", None)
        if not adapter_event_id:
            continue
        provenance.link_refs(
            event_id=adapter_event_id,
            source_ref_type=provenance.REF_ONTOLOGY_RUN,
            source_ref_id=ontology_run_id,
            target_ref_type=provenance.REF_SOURCE_ADAPTER_RUN,
            target_ref_id=str(adapter_event_id),
            link_type=provenance.LINK_USED,
            metadata={"source_name": source_name},
        )


def _record_source_record_refs(source_results: dict[str, Any]) -> None:
    try:
        from api import provenance
    except Exception:
        return
    for source_name, result in source_results.items():
        event_id = getattr(getattr(result, "lineage", None), "provenance_event_id", None)
        if not event_id:
            continue
        data = getattr(result, "data", None)
        as_of = getattr(result, "as_of", None)
        if isinstance(data, PortfolioSnapshot):
            for ticker, position in data.positions.items():
                provenance.record_source_ref(
                    adapter_run_event_id=event_id,
                    source_name=source_name,
                    record_kind="portfolio_position",
                    record_key=str(ticker).upper(),
                    record_value={
                        "ticker": position.ticker,
                        "asset": position.asset,
                        "direction": position.direction,
                        "latest_price": position.latest_price,
                        "series_points": position.series_points,
                        "as_of": position.as_of,
                    },
                    as_of=position.as_of or as_of,
                    summary={
                        "ticker": position.ticker,
                        "asset": position.asset,
                        "direction": position.direction,
                        "series_points": position.series_points,
                    },
                )
            continue
        if isinstance(data, SectorMetricsSnapshot):
            for row in data.rows:
                provenance.record_source_ref(
                    adapter_run_event_id=event_id,
                    source_name=source_name,
                    record_kind="sector_metric",
                    record_key=row.sector,
                    record_value={
                        "sector": row.sector,
                        "weight_now": row.weight_now,
                        "relperf_3m_pp": row.relperf_3m_pp,
                        "relperf_12m_pp": row.relperf_12m_pp,
                        "pct_above_200dma": row.pct_above_200dma,
                    },
                    as_of=data.timestamp or as_of,
                    summary={"sector": row.sector},
                )
            continue
        row_records = _source_rows_for_provenance(data, source_name, as_of)
        if row_records:
            for record_kind, record_key, record_value, record_as_of, summary in row_records:
                provenance.record_source_ref(
                    adapter_run_event_id=event_id,
                    source_name=source_name,
                    record_kind=record_kind,
                    record_key=record_key,
                    record_value=record_value,
                    as_of=record_as_of,
                    summary=summary,
                )
            continue
        provenance.record_source_ref(
            adapter_run_event_id=event_id,
            source_name=source_name,
            record_kind="snapshot",
            record_key=source_name,
            record_value={
                "source_name": source_name,
                "status": getattr(result, "status", None),
                "quality": getattr(result, "quality", None),
                "payload_fingerprint": getattr(getattr(result, "lineage", None), "payload_fingerprint", None),
            },
            as_of=as_of,
            summary={
                "source_name": source_name,
                "status": getattr(result, "status", None),
                "quality": getattr(result, "quality", None),
            },
        )


def _record_temporal_source_versions(
    source_results: dict[str, Any],
    *,
    repository: TemporalOntologyRepository,
) -> None:
    for source_name, result in source_results.items():
        write_source_result_records(
            source_name,
            result,
            repository=repository,
        )


def _write_temporal_graph_versions(
    *,
    nodes: list[OntologyNode],
    edges: list[OntologyEdge],
    as_of: str,
    provenance_event_id: str | None,
    repository: TemporalOntologyRepository,
) -> None:
    service = OntologyObjectService(repository=repository)
    actor = {"actor_type": "system", "actor_id": "ontology.ingestion"}
    provenance = {"provenance_event_id": provenance_event_id} if provenance_event_id else None

    for node in nodes:
        service.write_object(
            object_type=str(node.type),
            business_key=node.id,
            properties=node.properties,
            valid_from=as_of,
            actor=actor,
            provenance=provenance,
        )

    for edge in edges:
        service.write_relation(
            source_uid=edge.source_id,
            target_uid=edge.target_id,
            relation_type=str(edge.relation_type),
            properties=edge.properties,
            valid_from=as_of,
            actor=actor,
            provenance=provenance,
        )


def _refresh_temporal_read_models_after_ingestion() -> None:
    if not ontology_read_model_enabled():
        return
    try:
        TemporalReadModelRepository().refresh()
    except Exception:
        if ontology_primary_writes_enabled():
            raise
        logger.exception("ontology read model refresh failed after ingestion")


def _source_rows_for_provenance(
    data: Any,
    source_name: str,
    as_of: str | None,
) -> list[tuple[str, str, dict[str, Any], str | None, dict[str, Any]]]:
    if isinstance(data, PositioningSnapshot):
        return [
            (
                "positioning_row",
                row.instrument,
                _safe_record_value(row),
                row.report_date or as_of,
                {"instrument": row.instrument, "report_date": row.report_date},
            )
            for row in data.rows
        ]
    if isinstance(data, LaborMarketSnapshot):
        return [
            (
                "labor_indicator",
                key,
                _safe_record_value(indicator),
                indicator.date or data.timestamp or as_of,
                {"indicator": key, "date": indicator.date, "label": indicator.label},
            )
            for key, indicator in data.latest.items()
        ]
    if isinstance(data, LiquiditySnapshot):
        rows: list[tuple[str, str, dict[str, Any], str | None, dict[str, Any]]] = []
        for region, value in data.regional_scores.items():
            rows.append(
                (
                    "liquidity_region",
                    str(region),
                    {"region": region, "value": value},
                    data.latest_date or as_of,
                    {"region": region},
                )
            )
        for idx, component in enumerate(data.components):
            if not isinstance(component, dict):
                continue
            key = str(component.get("label") or component.get("name") or idx)
            rows.append(
                (
                    "liquidity_component",
                    key,
                    dict(component),
                    data.latest_date or as_of,
                    {"component": key},
                )
            )
        return rows
    if isinstance(data, SentimentSnapshot):
        rows = []
        for name, value in data.put_call.items():
            if isinstance(value, dict):
                rows.append(
                    ("sentiment_put_call", str(name), dict(value), str(value.get("as_of") or as_of), {"series": name})
                )
        for name, value in data.surveys.items():
            if isinstance(value, dict):
                rows.append(("sentiment_survey", str(name), dict(value), as_of, {"series": name}))
        for idx, value in enumerate(data.volatility):
            if isinstance(value, dict):
                key = str(value.get("date") or idx)
                rows.append(("sentiment_volatility", key, dict(value), str(value.get("date") or as_of), {"date": key}))
        return rows
    if isinstance(data, EconomicGrowthSnapshot):
        return [
            (
                "economic_growth_bucket",
                key,
                value if isinstance(value, dict) else {"value": value},
                data.timestamp or as_of,
                {"bucket": key},
            )
            for key, value in {
                "commodities": data.commodities,
                "equities": data.equities,
                "equity_relative_returns": data.equity_relative_returns,
                "currencies": data.currencies,
            }.items()
            if value
        ]
    if isinstance(data, VixTermStructureSnapshot):
        value = _safe_record_value(data)
        return [("snapshot", source_name, value, data.date or as_of, {"source_name": source_name, "date": data.date})]
    if isinstance(data, MarketBreadthSnapshot):
        value = _safe_record_value(data)
        return [
            (
                "snapshot",
                source_name,
                value,
                data.as_of_date or as_of,
                {"source_name": source_name, "as_of_date": data.as_of_date},
            )
        ]
    if isinstance(data, Top50BreadthSnapshot):
        return [("snapshot", source_name, _safe_record_value(data), as_of, {"source_name": source_name})]
    return []


def _safe_record_value(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return cast(dict[str, Any], asdict(cast(Any, value)))
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def _source_data(source_results: dict[str, Any], name: str) -> Any:
    result = source_results.get(name)
    return getattr(result, "data", None)


def _ingest_thesis_entities(
    add_node: Callable[[OntologyNode], None],
    add_edge: Callable[[OntologyEdge], None],
    run_id: str,
    position_ids: list[str],
) -> None:
    """Create Thesis and Evaluation nodes linked to existing Position nodes."""
    import logging

    from ontology.runtime_read_service import OntologyRuntimeReadService

    log = logging.getLogger(__name__)

    try:
        reads = OntologyRuntimeReadService()
        all_meta = reads.theses(limit=1000)
        latest_evals = reads.latest_evaluations(limit=1000)
    except Exception:
        log.warning("Could not load thesis data for ontology ingestion", exc_info=True)
        return

    meta_by_ticker = {str(m["ticker"]).upper(): m for m in all_meta}
    eval_by_ticker = {str(e["ticker"]).upper(): e for e in latest_evals}
    position_tickers = {pid.split(":")[-1].upper() for pid in position_ids}

    for ticker in position_tickers:
        meta = meta_by_ticker.get(ticker)
        if meta is None:
            continue

        position_id = f"position:{ticker}"
        thesis_id = f"thesis:{ticker}"

        add_node(
            OntologyNode(
                id=thesis_id,
                type="Thesis",
                label=f"Thesis: {ticker}",
                properties={
                    "ticker": ticker,
                    "status": meta.get("status"),
                    "created_at": meta.get("created_at"),
                    "updated_at": meta.get("updated_at"),
                    "ontology_run_id": run_id,
                },
            )
        )
        add_edge(
            OntologyEdge(
                source_id=position_id,
                target_id=thesis_id,
                relation_type=HAS_THESIS,
                properties={"ontology_run_id": run_id},
            )
        )

        evaluation = eval_by_ticker.get(ticker)
        if evaluation:
            eval_id = f"evaluation:{ticker}:{evaluation.get('evaluated_at', 'latest')}"
            add_node(
                OntologyNode(
                    id=eval_id,
                    type="Evaluation",
                    label=f"Eval: {ticker}",
                    properties={
                        "ticker": ticker,
                        "evaluated_at": evaluation.get("evaluated_at"),
                        "thesis_status": evaluation.get("thesis_status"),
                        "technical_read": evaluation.get("technical_read"),
                        "fundamental_read": evaluation.get("fundamental_read"),
                        "action": evaluation.get("action"),
                        "confidence": evaluation.get("confidence"),
                        "risk_flag": evaluation.get("risk_flag"),
                        "ontology_run_id": run_id,
                    },
                )
            )
            add_edge(
                OntologyEdge(
                    source_id=thesis_id,
                    target_id=eval_id,
                    relation_type=EVALUATED_BY,
                    properties={"ontology_run_id": run_id},
                )
            )

        # Parse catalyst nodes from thesis markdown
        catalysts = _parse_catalysts_from_thesis(ticker)
        for i, catalyst in enumerate(catalysts):
            catalyst_id = f"catalyst:{ticker}:{i}"
            add_node(
                OntologyNode(
                    id=catalyst_id,
                    type="Catalyst",
                    label=catalyst["name"],
                    properties={
                        "ticker": ticker,
                        "name": catalyst["name"],
                        "description": catalyst["description"],
                        "ontology_run_id": run_id,
                    },
                )
            )
            add_edge(
                OntologyEdge(
                    source_id=thesis_id,
                    target_id=catalyst_id,
                    relation_type=HAS_CATALYST,
                    properties={"ontology_run_id": run_id},
                )
            )


def _parse_catalysts_from_thesis(ticker: str) -> list[dict[str, str]]:
    """Extract catalyst bullet points from a thesis markdown file."""
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    theses_dir = repo_root / "investment_theses"

    # Try exact match, then case-insensitive
    thesis_path = theses_dir / f"{ticker}.md"
    if not thesis_path.exists():
        for p in theses_dir.glob("*.md"):
            if p.stem.upper() == ticker.upper():
                thesis_path = p
                break
    if not thesis_path.exists():
        return []

    try:
        content = thesis_path.read_text(encoding="utf-8")
    except Exception:
        return []

    # Find ## Key Catalysts section
    catalyst_match = re.search(
        r"##\s+Key\s+Catalysts\s*\n(.*?)(?=\n##|\Z)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if not catalyst_match:
        return []

    section = catalyst_match.group(1)
    catalysts: list[dict[str, str]] = []

    # Parse bullet points: - **Name**: Description  or  - Name: Description
    for line in section.split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        line = line[2:].strip()
        if not line or line.startswith("<!--"):
            continue

        # Try **bold**: description format
        bold_match = re.match(r"\*\*(.+?)\*\*[:\s]*(.*)$", line)
        if bold_match:
            catalysts.append(
                {
                    "name": bold_match.group(1).strip(),
                    "description": bold_match.group(2).strip(),
                }
            )
        else:
            # Plain text bullet
            catalysts.append(
                {
                    "name": line[:80],
                    "description": line,
                }
            )

    return catalysts


def _extract_latest_price(series: Any) -> float | None:
    if isinstance(series, list) and series:
        last = series[-1]
        if isinstance(last, dict):
            return _to_float(last.get("value"))
    return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_sector_name_from_edges(
    edge_map: dict[tuple[str, str, str], OntologyEdge],
    position_id: str,
    nodes: dict[str, OntologyNode],
) -> str:
    asset_id = None
    for (src, tgt, rel), _edge in edge_map.items():
        if src == position_id and rel == REFERENCES_ASSET:
            asset_id = tgt
            break
    if asset_id is None:
        return "Unknown Equity"

    sector_id = None
    for (src, tgt, rel), _edge in edge_map.items():
        if src == asset_id and rel == BELONGS_TO_SECTOR:
            sector_id = tgt
            break
    if sector_id is None:
        return "Unknown Equity"

    node = nodes.get(sector_id)
    if node and isinstance(node.properties.get("name"), str):
        return str(node.properties["name"])
    return "Unknown Equity"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _slug(text: str) -> str:
    value = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    value = "_".join(part for part in value.split("_") if part)
    return value or "unknown"


def _add_indicator_signals(
    add_node: Callable[[OntologyNode], None],
    add_edge: Callable[[OntologyEdge], None],
    run_id: str,
    indicator_id: str,
    indicator_label: str,
    evidence: list[dict[str, Any]],
) -> list[str]:
    add_node(
        OntologyNode(
            id=indicator_id,
            type="MacroIndicator",
            label=indicator_label,
            properties={"ontology_run_id": run_id, "as_of": run_id},
        )
    )

    signal_ids: list[str] = []
    for ev in evidence:
        source = str(ev.get("source") or "indicator")
        name = str(ev.get("name") or "signal")
        signal_id = f"signal:{_slug(source)}:{_slug(name)}"
        signal_ids.append(signal_id)

        add_node(
            OntologyNode(
                id=signal_id,
                type="Signal",
                label=name,
                properties={
                    **ev,
                    "ontology_run_id": run_id,
                },
            )
        )
        add_edge(
            OntologyEdge(
                source_id=indicator_id,
                target_id=signal_id,
                relation_type=EMITS_SIGNAL,
                properties={"ontology_run_id": run_id},
            )
        )

    return signal_ids
