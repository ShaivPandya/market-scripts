from __future__ import annotations

from collections import defaultdict
from typing import Any

from ontology.ingestion import ingest_into_repository
from ontology.parser import parse_hybrid_query
from ontology.repository import OntologyRepository

VALID_TIMEFRAMES = {"This Week", "Daily", "Weekly", "Monthly"}
KNOWN_SECTORS = {
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
    "Emerging Markets ETF",
    "South Korea ETF",
    "Brazil ETF",
    "Commodities",
    "FX",
    "Rates",
    "Other Assets",
    "Unknown Equity",
}


class OntologyRunNotFoundError(Exception):
    """Raised when a requested ontology snapshot run_id does not exist."""

    def __init__(self, run_id: str):
        super().__init__(f"Ontology run not found: {run_id}")
        self.run_id = run_id


class OntologyQueryService:
    def __init__(self, repository: OntologyRepository | None = None):
        self.repo = repository or OntologyRepository()

    def query(
        self,
        query: str | None,
        intent: str | None,
        filters: dict[str, Any] | None,
        timeframe: str = "Daily",
        include_graph: bool = False,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        tf = timeframe if timeframe in VALID_TIMEFRAMES else "Daily"

        interpreted = parse_hybrid_query(
            query=query,
            intent=intent,
            filters=filters,
            known_sectors=KNOWN_SECTORS,
        )

        if run_id:
            run = self.repo.get_run(run_id)
            if run is None:
                raise OntologyRunNotFoundError(run_id)
            resolved_run_id = str(run["run_id"])
            as_of = str(run["as_of"])
            source_status = _as_dict(run.get("source_status"))
            required_modules = _as_str_list(run.get("required_modules"))
        else:
            deep_fetch = include_graph or interpreted.intent == "entity_context"
            ingestion = ingest_into_repository(
                repo=self.repo,
                timeframe=tf,
                include_deep_modules=deep_fetch,
            )
            resolved_run_id = ingestion.run_id
            as_of = ingestion.as_of
            source_status = ingestion.source_status
            required_modules = ingestion.required_modules

        effective_filters = dict(interpreted.filters)
        if interpreted.intent == "entity_context" and interpreted.entity:
            if "tickers" not in effective_filters and "sectors" not in effective_filters:
                token = str(interpreted.entity).strip()
                if token.upper() == token and any(ch.isalpha() for ch in token):
                    effective_filters["tickers"] = [token.upper()]
                else:
                    effective_filters["sectors"] = [token]

        rows = self.repo.fetch_snapshot_position_asset_sector_rows(run_id=resolved_run_id)
        results = []
        for row in rows:
            pos = _as_dict(row.get("position_props"))
            position_id = str(row.get("position_id") or "")
            ticker = str(pos.get("ticker") or position_id.split(":")[-1])
            asset = str(pos.get("asset") or "unknown")
            direction = str(pos.get("direction") or "unknown")
            risk_score = _to_float(pos.get("risk_score")) or 0.0
            risk_level = str(pos.get("risk_level") or _risk_level_from_score(risk_score))
            sector = "Unknown Equity"
            sector_props = _as_dict(row.get("sector_props"))
            if isinstance(sector_props.get("name"), str):
                sector = str(sector_props.get("name"))

            evidence = self._build_evidence(position_id=position_id, run_id=resolved_run_id)

            results.append(
                {
                    "ticker": ticker,
                    "asset": asset,
                    "direction": direction,
                    "sector": sector,
                    "risk_score": round(risk_score, 4),
                    "risk_level": risk_level,
                    "evidence": evidence,
                }
            )

        if interpreted.intent == "positions_in_deteriorating_macro":
            results = [r for r in results if (_to_float(r.get("risk_score")) or 0.0) >= 0.6]

        results = _apply_filters(results, effective_filters)
        results.sort(key=lambda r: _to_float(r.get("risk_score")) or 0.0, reverse=True)
        max_results = _to_int(effective_filters.get("max_results"))
        if max_results is not None and max_results > 0:
            results = results[:max_results]

        aggregate = _build_aggregate(results, source_status, required_modules)
        response: dict[str, Any] = {
            "run_id": resolved_run_id,
            "intent": interpreted.intent,
            "interpreted_query": {
                "source": interpreted.source,
                "query": interpreted.original_query,
                "entity": interpreted.entity,
                "filters": effective_filters,
            },
            "as_of": as_of,
            "source_status": source_status,
            "results": results,
            "aggregate": aggregate,
        }

        if include_graph:
            response["graph"] = self.repo.fetch_snapshot_graph(run_id=resolved_run_id)

        return response

    def _build_evidence(self, position_id: str, run_id: str) -> list[dict[str, Any]]:
        raw = self.repo.fetch_snapshot_position_signal_evidence(run_id=run_id, position_id=position_id)
        evidence = []
        for row in raw:
            edge = _as_dict(row.get("edge_props"))
            evidence.append(
                {
                    "source": edge.get("source"),
                    "name": edge.get("name"),
                    "value": edge.get("value"),
                    "threshold": edge.get("threshold"),
                    "direction": edge.get("direction"),
                    "contribution": edge.get("contribution"),
                    "signal_id": row.get("signal_id"),
                }
            )

        evidence.sort(key=lambda r: _to_float(r.get("contribution")) or 0.0, reverse=True)
        return evidence[:4]


def _apply_filters(results: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    out = list(results)

    tickers = filters.get("tickers") if isinstance(filters.get("tickers"), list) else None
    sectors = filters.get("sectors") if isinstance(filters.get("sectors"), list) else None
    assets = filters.get("assets") if isinstance(filters.get("assets"), list) else None

    if tickers:
        wanted = {str(t).upper() for t in tickers}
        out = [r for r in out if str(r.get("ticker", "")).upper() in wanted]

    if sectors:
        wanted = {str(s).lower() for s in sectors}
        out = [r for r in out if str(r.get("sector", "")).lower() in wanted]

    if assets:
        wanted = {str(a).lower() for a in assets}
        out = [r for r in out if str(r.get("asset", "")).lower() in wanted]

    min_risk = _to_float(filters.get("min_risk_score"))
    if min_risk is not None:
        out = [r for r in out if (_to_float(r.get("risk_score")) or 0.0) >= min_risk]

    return out


def _build_aggregate(
    results: list[dict[str, Any]],
    source_status: dict[str, dict[str, Any]],
    required_modules: list[str],
) -> dict[str, Any]:
    by_level = {"high": 0, "medium": 0, "low": 0}
    by_asset: dict[str, int] = defaultdict(int)
    scores: list[float] = []

    for row in results:
        lvl = str(row.get("risk_level") or "low")
        if lvl not in by_level:
            lvl = "low"
        by_level[lvl] += 1

        asset = str(row.get("asset") or "unknown").lower()
        by_asset[asset] += 1

        s = _to_float(row.get("risk_score"))
        if s is not None:
            scores.append(s)

    confidence = _compute_confidence(source_status, required_modules)

    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {
        "position_count": len(results),
        "risk_buckets": by_level,
        "asset_exposure_counts": dict(by_asset),
        "average_risk_score": avg_score,
        "confidence": round(confidence, 4),
    }


def _compute_confidence(source_status: dict[str, dict[str, Any]], required_modules: list[str]) -> float:
    required = set(required_modules)
    req_errors = 0
    req_partials = 0
    opt_errors = 0
    opt_partials = 0

    for module, state in source_status.items():
        status = str((state or {}).get("status") or "error")
        if module in required:
            if status == "error":
                req_errors += 1
            elif status == "partial":
                req_partials += 1
        else:
            if status == "error":
                opt_errors += 1
            elif status == "partial":
                opt_partials += 1

    score = 1.0
    score -= req_errors * 0.12
    score -= req_partials * 0.06
    score -= opt_errors * 0.03
    score -= opt_partials * 0.015
    return max(0.2, min(1.0, score))


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, (str, int, float))]


def _risk_level_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"
