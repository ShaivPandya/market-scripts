from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
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
SNAPSHOT_REUSE_MAX_AGE = timedelta(minutes=15)


class OntologyRunNotFoundError(Exception):
    """Raised when a requested ontology snapshot run_id does not exist."""

    def __init__(self, run_id: str):
        super().__init__(f"Ontology run not found: {run_id}")
        self.run_id = run_id


class OntologyQueryService:
    def __init__(self, repository: OntologyRepository | None = None):
        self.repo = repository or OntologyRepository()

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.repo.list_runs(limit=limit)

    def query(
        self,
        query: str | None,
        intent: str | None,
        filters: dict[str, Any] | None,
        timeframe: str = "Daily",
        include_graph: bool = False,
        run_id: str | None = None,
        refresh_snapshot: bool = False,
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
            latest_run = self.repo.get_latest_run() if not refresh_snapshot else None
            if latest_run is not None and self._can_reuse_run(latest_run):
                resolved_run_id = str(latest_run["run_id"])
                as_of = str(latest_run["as_of"])
                source_status = _as_dict(latest_run.get("source_status"))
                required_modules = _as_str_list(latest_run.get("required_modules"))
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
        all_evidence = self.repo.fetch_snapshot_all_position_signal_evidence(run_id=resolved_run_id)
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

            evidence = self._build_evidence_from_batch(all_evidence.get(position_id, []))

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

        if interpreted.intent == "thesis_review":
            results = _enrich_with_thesis(results, resolved_run_id, self.repo)

        if interpreted.intent == "temporal_comparison":
            diff = self._auto_temporal_diff(resolved_run_id)
            if diff:
                return {
                    "run_id": resolved_run_id,
                    "intent": "temporal_comparison",
                    "interpreted_query": {
                        "source": interpreted.source,
                        "query": interpreted.original_query,
                        "entity": interpreted.entity,
                        "filters": effective_filters,
                    },
                    "as_of": as_of,
                    "source_status": source_status,
                    "diff": diff,
                    "results": results,
                    "aggregate": _build_aggregate(results, source_status, required_modules),
                }

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

    def _can_reuse_run(self, run: dict[str, Any]) -> bool:
        if not _run_is_fresh(run.get("created_at"), max_age=SNAPSHOT_REUSE_MAX_AGE):
            return False

        source_status = _as_dict(run.get("source_status"))
        required_modules = _as_str_list(run.get("required_modules"))

        # Reuse only healthy required modules; avoid pinning to degraded snapshots.
        for module in required_modules:
            state = _as_dict(source_status.get(module))
            if str(state.get("status") or "error") != "ok":
                return False

        run_id = str(run.get("run_id") or "")
        if not run_id:
            return False

        rows = self.repo.fetch_snapshot_position_asset_sector_rows(run_id=run_id)
        return len(rows) > 0

    def compare_snapshots(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Diff two ontology snapshots. Returns position changes, risk score deltas, and signal transitions."""
        run_a = self.repo.get_run(run_id_a)
        run_b = self.repo.get_run(run_id_b)
        if run_a is None:
            raise OntologyRunNotFoundError(run_id_a)
        if run_b is None:
            raise OntologyRunNotFoundError(run_id_b)

        rows_a = self.repo.fetch_snapshot_position_asset_sector_rows(run_id=run_id_a)
        rows_b = self.repo.fetch_snapshot_position_asset_sector_rows(run_id=run_id_b)

        def _positions_map(rows: list[dict]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for row in rows:
                pos = _as_dict(row.get("position_props"))
                ticker = str(pos.get("ticker") or "").upper()
                if ticker:
                    out[ticker] = pos
            return out

        positions_a = _positions_map(rows_a)
        positions_b = _positions_map(rows_b)

        added = sorted(set(positions_b) - set(positions_a))
        removed = sorted(set(positions_a) - set(positions_b))
        common = sorted(set(positions_a) & set(positions_b))

        risk_changes: list[dict[str, Any]] = []
        signal_transitions: list[dict[str, Any]] = []

        for ticker in common:
            pa = positions_a[ticker]
            pb = positions_b[ticker]
            score_a = _to_float(pa.get("risk_score")) or 0.0
            score_b = _to_float(pb.get("risk_score")) or 0.0
            delta = round(score_b - score_a, 4)

            if abs(delta) >= 0.02:
                risk_changes.append(
                    {
                        "ticker": ticker,
                        "risk_score_before": round(score_a, 4),
                        "risk_score_after": round(score_b, 4),
                        "delta": delta,
                        "level_before": str(pa.get("risk_level") or _risk_level_from_score(score_a)),
                        "level_after": str(pb.get("risk_level") or _risk_level_from_score(score_b)),
                    }
                )

            # Check component-level transitions
            for component in ("volatility_cluster", "breadth_stress", "sector_stress", "macro_regime"):
                va = _to_float(pa.get(component))
                vb = _to_float(pb.get(component))
                if va is not None and vb is not None:
                    cd = round(vb - va, 4)
                    if abs(cd) >= 0.05:
                        dir_a = "deteriorating" if va >= 0.6 else "stable"
                        dir_b = "deteriorating" if vb >= 0.6 else "stable"
                        if dir_a != dir_b:
                            signal_transitions.append(
                                {
                                    "ticker": ticker,
                                    "component": component,
                                    "before": round(va, 4),
                                    "after": round(vb, 4),
                                    "transition": f"{dir_a} -> {dir_b}",
                                }
                            )

        risk_changes.sort(key=lambda r: abs(r.get("delta", 0)), reverse=True)

        # Component score diffs
        scores_a = _as_dict(run_a.get("component_scores"))
        scores_b = _as_dict(run_b.get("component_scores"))
        component_diffs: dict[str, dict[str, float]] = {}
        for key in set(list(scores_a.keys()) + list(scores_b.keys())):
            va = _to_float(scores_a.get(key))
            vb = _to_float(scores_b.get(key))
            if va is not None and vb is not None:
                component_diffs[key] = {
                    "before": round(va, 4),
                    "after": round(vb, 4),
                    "delta": round(vb - va, 4),
                }

        return {
            "run_id_before": run_id_a,
            "run_id_after": run_id_b,
            "as_of_before": str(run_a.get("as_of", "")),
            "as_of_after": str(run_b.get("as_of", "")),
            "positions_added": added,
            "positions_removed": removed,
            "risk_changes": risk_changes[:20],
            "signal_transitions": signal_transitions[:20],
            "component_diffs": component_diffs,
            "total_positions": {"before": len(positions_a), "after": len(positions_b)},
        }

    def _auto_temporal_diff(self, current_run_id: str) -> dict[str, Any] | None:
        """Find the most recent prior run and diff against it."""
        runs = self.repo.list_runs(limit=10)
        prior_run_id: str | None = None
        for run in runs:
            rid = str(run.get("run_id", ""))
            if rid and rid != current_run_id:
                prior_run_id = rid
                break
        if not prior_run_id:
            return None
        try:
            return self.compare_snapshots(prior_run_id, current_run_id)
        except Exception:
            return None

    def _build_evidence(self, position_id: str, run_id: str) -> list[dict[str, Any]]:
        raw = self.repo.fetch_snapshot_position_signal_evidence(run_id=run_id, position_id=position_id)
        return self._build_evidence_from_batch(raw)

    def _build_evidence_from_batch(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _enrich_with_thesis(
    results: list[dict[str, Any]],
    run_id: str,
    repo: OntologyRepository,
) -> list[dict[str, Any]]:
    """Enrich position results with thesis metadata from the ontology snapshot."""
    graph = repo.fetch_snapshot_graph(run_id=run_id)
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []

    # Build lookup: position_id -> thesis node properties
    thesis_by_position: dict[str, dict[str, Any]] = {}
    eval_by_thesis: dict[str, dict[str, Any]] = {}

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if edge.get("relation_type") == "has_thesis":
            thesis_by_position[edge["source_id"]] = edge["target_id"]
        if edge.get("relation_type") == "evaluated_by":
            eval_by_thesis[edge["source_id"]] = edge["target_id"]

    node_props: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if isinstance(node, dict):
            node_props[node.get("id", "")] = node.get("properties", {})

    for result in results:
        ticker = str(result.get("ticker") or "").upper()
        position_id = f"position:{ticker}"
        thesis_id = thesis_by_position.get(position_id)

        if thesis_id and isinstance(thesis_id, str):
            t_props = _as_dict(node_props.get(thesis_id))
            result["thesis"] = {
                "status": t_props.get("status"),
                "created_at": t_props.get("created_at"),
                "updated_at": t_props.get("updated_at"),
            }
            eval_id = eval_by_thesis.get(thesis_id)
            if eval_id and isinstance(eval_id, str):
                e_props = _as_dict(node_props.get(eval_id))
                result["latest_evaluation"] = {
                    "evaluated_at": e_props.get("evaluated_at"),
                    "thesis_status": e_props.get("thesis_status"),
                    "technical_read": e_props.get("technical_read"),
                    "fundamental_read": e_props.get("fundamental_read"),
                    "action": e_props.get("action"),
                    "confidence": e_props.get("confidence"),
                    "risk_flag": e_props.get("risk_flag"),
                }
        else:
            result["thesis"] = None
            result["latest_evaluation"] = None

    return results


def _run_is_fresh(created_at: Any, *, max_age: timedelta) -> bool:
    created_dt = _parse_run_created_at(created_at)
    if created_dt is None:
        return False
    age = datetime.now(UTC) - created_dt
    return age <= max_age


def _parse_run_created_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
