from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ontology.models import OntologyEdge, OntologyNode
from ontology.repository import OntologyRepository
from ontology.risk import (
    compute_breadth_stress,
    compute_macro_regime,
    compute_sector_stress_map,
    compute_volatility_cluster,
    risk_level,
    score_position,
)
from ontology.sector_mapper import SectorMapper

ModuleFetcher = Callable[[], dict[str, Any] | list[Any]]
SNAPSHOT_RETENTION_DAYS = 90


@dataclass(slots=True)
class IngestionOutput:
    run_id: str
    as_of: str
    source_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    required_modules: list[str] = field(default_factory=list)
    optional_modules: list[str] = field(default_factory=list)
    component_scores: dict[str, float] = field(default_factory=dict)


def ingest_into_repository(
    repo: OntologyRepository,
    timeframe: str,
    include_deep_modules: bool,
) -> IngestionOutput:
    run_id = datetime.now(UTC).isoformat()
    source_status: dict[str, dict[str, Any]] = {}

    required_fetchers, optional_fetchers, deep_fetchers = _build_fetchers(timeframe=timeframe)

    required_data = _run_fetchers(required_fetchers, source_status)
    optional_data = _run_fetchers(optional_fetchers, source_status)
    if include_deep_modules:
        _run_fetchers(deep_fetchers, source_status)

    nodes: dict[str, OntologyNode] = {}
    edges: dict[tuple[str, str, str], OntologyEdge] = {}

    def add_node(node: OntologyNode) -> None:
        nodes[node.id] = node

    def add_edge(edge: OntologyEdge) -> None:
        edges[(edge.source_id, edge.target_id, edge.relation_type)] = edge

    portfolio = _as_dict(required_data.get("portfolio"))
    sector_mapper = SectorMapper()
    position_ids: list[str] = []

    # Core entity graph: Position -> Asset -> Sector
    metadata = _as_dict(portfolio.get("metadata"))
    positions = _as_dict(portfolio.get("positions"))
    portfolio_timestamp = portfolio.get("timestamp")

    for ticker, meta_obj in metadata.items():
        ticker_norm = str(ticker).strip().upper()
        if not ticker_norm:
            continue

        meta = meta_obj if isinstance(meta_obj, dict) else {}
        asset_class = str(meta.get("asset") or "unknown").strip().lower()
        direction = str(meta.get("direction") or "unknown").strip().lower()

        position_id = f"position:{ticker_norm}"
        asset_id = f"asset:{ticker_norm}"
        latest_price = _extract_latest_price(positions.get(ticker_norm) or positions.get(ticker))

        sector = sector_mapper.resolve_sector(ticker_norm, asset_class)
        sector_id = f"sector:{_slug(sector.sector)}"

        add_node(
            OntologyNode(
                id=position_id,
                type="Position",
                label=ticker_norm,
                properties={
                    "ticker": ticker_norm,
                    "asset": asset_class,
                    "direction": direction,
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
                },
            )
        )
        add_node(
            OntologyNode(
                id=sector_id,
                type="Sector",
                label=sector.sector,
                properties={
                    "name": sector.sector,
                    "sector_source": sector.source,
                },
            )
        )
        add_edge(
            OntologyEdge(
                source_id=position_id,
                target_id=asset_id,
                relation_type="references_asset",
                properties={"ontology_run_id": run_id},
            )
        )
        add_edge(
            OntologyEdge(
                source_id=asset_id,
                target_id=sector_id,
                relation_type="belongs_to_sector",
                properties={"source": sector.source, "ontology_run_id": run_id},
            )
        )
        position_ids.append(position_id)

    # Compute global component scores from module outputs
    vix_data = _as_dict(required_data.get("vix_term_structure"))
    breadth_data = _as_dict(required_data.get("market_breadth"))
    top50_data = _as_dict(required_data.get("top50_breadth"))
    sector_metrics_data = _as_dict(required_data.get("sector_metrics"))
    liquidity_data = _as_dict(required_data.get("liquidity"))

    sentiment_data = _as_dict(optional_data.get("sentiment"))
    positioning_data = _as_dict(optional_data.get("positioning_summary"))
    economic_growth_data = _as_dict(optional_data.get("economic_growth"))
    labor_market_data = _as_dict(optional_data.get("labor_market"))

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
                relation_type="emits_signal",
                properties={"ontology_run_id": run_id},
            )
        )
        add_edge(
            OntologyEdge(
                source_id=f"sector:{_slug(sector_name)}",
                target_id="macro_indicator:sector_metrics",
                relation_type="affected_by",
                properties={"ontology_run_id": run_id},
            )
        )

    # Attach risk to each position and create position->signal exposure edges
    for position_id in position_ids:
        node = nodes[position_id]
        props = dict(node.properties)
        sector_name = _resolve_sector_name_from_edges(edges, position_id, nodes)
        sector_stress = sector_scores.get(sector_name, sector_scores.get("Unknown Equity", 0.5))

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
                    relation_type="exposed_to_signal",
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

    snapshot_nodes = list(nodes.values())
    snapshot_edges = list(edges.values())

    as_of = str(portfolio_timestamp or datetime.now(UTC).isoformat())
    required_modules = list(required_fetchers.keys())
    optional_modules = list(optional_fetchers.keys()) + (list(deep_fetchers.keys()) if include_deep_modules else [])
    component_scores = {
        "volatility_cluster": round(volatility_cluster, 4),
        "breadth_stress": round(breadth_stress, 4),
        "macro_regime": round(macro_regime, 4),
    }

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

    return IngestionOutput(
        run_id=run_id,
        as_of=as_of,
        source_status=source_status,
        required_modules=required_modules,
        optional_modules=optional_modules,
        component_scores=component_scores,
    )


def _build_fetchers(
    timeframe: str,
) -> tuple[dict[str, ModuleFetcher], dict[str, ModuleFetcher], dict[str, ModuleFetcher]]:
    from api.routers.breakout import get_breakout
    from api.routers.central_banks import get_central_banks
    from api.routers.commodities import get_commodities
    from api.routers.commodities_curve import get_commodities_curve
    from api.routers.country_dashboard import get_country_dashboard
    from api.routers.economic_growth import get_economic_growth
    from api.routers.fx_dashboard import get_fx_dashboard
    from api.routers.fx_model import list_pairs
    from api.routers.index_dashboard import get_index_dashboard
    from api.routers.industry import get_industry_monitor
    from api.routers.labor_market import get_labor_market
    from api.routers.liquidity import get_liquidity
    from api.routers.market_technicals import (
        get_market_breadth,
        get_price_volume_signals,
        get_top50_breadth,
        get_vix_term_structure,
    )
    from api.routers.momentum import get_momentum
    from api.routers.portfolio import get_portfolio
    from api.routers.portfolio_news import get_portfolio_news
    from api.routers.positioning import get_positioning_summary
    from api.routers.sector_metrics import get_sector_metrics
    from api.routers.sentiment import get_put_call, get_surveys, get_volatility
    from api.routers.yield_curve import get_yield_curve

    required: dict[str, ModuleFetcher] = {
        "portfolio": lambda: get_portfolio(timeframe=timeframe, all_timeframes=False),
        "market_breadth": get_market_breadth,
        "top50_breadth": get_top50_breadth,
        "vix_term_structure": get_vix_term_structure,
        "sector_metrics": get_sector_metrics,
        "liquidity": get_liquidity,
    }

    optional: dict[str, ModuleFetcher] = {
        "sentiment": lambda: {
            "put_call": get_put_call(lookback_days=180),
            "surveys": get_surveys(),
            "volatility": get_volatility(lookback_days=365),
        },
        "positioning_summary": get_positioning_summary,
        "economic_growth": get_economic_growth,
        "labor_market": get_labor_market,
    }

    deep: dict[str, ModuleFetcher] = {
        "index_dashboard": lambda: get_index_dashboard(timeframe=timeframe),
        "fx_dashboard": lambda: get_fx_dashboard(timeframe=timeframe),
        "commodities": lambda: get_commodities(timeframe=timeframe),
        "price_volume_signals": get_price_volume_signals,
        "momentum": get_momentum,
        "country_dashboard": lambda: get_country_dashboard(metric="Inflation"),
        "breakout": get_breakout,
        "yield_curve": lambda: get_yield_curve(lookback_days=90),
        "central_banks": lambda: get_central_banks(refresh=False),
        "industry_monitor": lambda: get_industry_monitor(refresh=False),
        "portfolio_news": lambda: get_portfolio_news(refresh=False),
        "commodities_curve": lambda: get_commodities_curve(commodity="CL", lookback_days=30),
        "fx_model_pairs": list_pairs,
    }

    return required, optional, deep


def _run_fetchers(fetchers: dict[str, ModuleFetcher], source_status: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not fetchers:
        return out

    with ThreadPoolExecutor(max_workers=min(len(fetchers), 10)) as pool:
        futures = {pool.submit(fn): name for name, fn in fetchers.items()}
        for fut in as_completed(futures, timeout=120):
            name = futures[fut]
            try:
                data = fut.result(timeout=90)
                out[name] = data
                if _is_partial(name, data):
                    source_status[name] = {"status": "partial", "detail": "incomplete payload"}
                else:
                    source_status[name] = {"status": "ok"}
            except Exception as exc:
                out[name] = {}
                source_status[name] = {"status": "error", "detail": str(exc)}
    return out


def _is_partial(name: str, data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if name == "sentiment":
        surveys = data.get("surveys")
        if isinstance(surveys, dict):
            errs = surveys.get("errors")
            if isinstance(errs, dict) and errs:
                return True
    errors = data.get("errors")
    if isinstance(errors, dict) and errors:
        return True
    return False


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
        if src == position_id and rel == "references_asset":
            asset_id = tgt
            break
    if asset_id is None:
        return "Unknown Equity"

    sector_id = None
    for (src, tgt, rel), _edge in edge_map.items():
        if src == asset_id and rel == "belongs_to_sector":
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
                relation_type="emits_signal",
                properties={"ontology_run_id": run_id},
            )
        )

    return signal_ids
