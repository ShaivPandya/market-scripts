from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from api.audit import emit_audit_event
from ontology.action_registry import get_tool_exposure
from ontology.ingestion import ingest_into_repository
from ontology.parser import parse_hybrid_query
from ontology.policy import (
    DEFAULT_ONTOLOGY_POLICY,
    Actor,
    EdgeResource,
    NodeResource,
    OntologyAction,
    OntologyPolicy,
    PolicyDenied,
    admin_actor,
    filter_graph,
    redact_properties,
    require_allowed,
)
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
GRAPH_PAGE_NODE_LIMIT = 500
GRAPH_PAGE_EDGE_LIMIT = 1000


class OntologyRunNotFoundError(Exception):
    """Raised when a requested ontology snapshot run_id does not exist."""

    def __init__(self, run_id: str):
        super().__init__(f"Ontology run not found: {run_id}")
        self.run_id = run_id


class OntologyQueryService:
    def __init__(
        self,
        repository: OntologyRepository | None = None,
        policy: OntologyPolicy | None = None,
    ):
        self.repo = repository or OntologyRepository()
        self.policy = policy or DEFAULT_ONTOLOGY_POLICY

    def list_runs(self, limit: int = 100, actor: Actor | None = None) -> list[dict[str, Any]]:
        actor = actor or admin_actor(source="service")
        try:
            require_allowed(self.policy.check_action(actor, OntologyAction.RUNS_LIST, {"limit": limit}))
        except PolicyDenied as exc:
            _emit_ontology_read_audit(
                "ontology.runs.list",
                actor=actor,
                status="denied",
                metadata={"limit": limit},
                error=exc.reason,
            )
            raise
        runs = self.repo.list_runs(limit=limit)
        _emit_ontology_read_audit(
            "ontology.runs.list",
            actor=actor,
            status="succeeded",
            metadata={"limit": limit},
            after_summary={"run_count": len(runs)},
        )
        return runs

    def query(
        self,
        query: str | None,
        intent: str | None,
        filters: dict[str, Any] | None,
        timeframe: str = "Daily",
        include_graph: bool = False,
        run_id: str | None = None,
        refresh_snapshot: bool = False,
        page: int = 1,
        page_size: int = 25,
        schema_mode: str = "upgraded",
        actor: Actor | None = None,
    ) -> dict[str, Any]:
        if schema_mode != "upgraded":
            raise ValueError("Ontology semantic queries require schema_mode='upgraded'")
        actor = actor or admin_actor(source="service")
        query_tool = get_tool_exposure("query_ontology")
        query_policy = query_tool.policy_spec
        try:
            action_context = {"intent": intent, "run_id": run_id}
            required_actions = query_policy.ontology_actions if query_policy else (OntologyAction.QUERY,)
            for action_name in required_actions:
                require_allowed(self.policy.check_action(actor, action_name, action_context))
            dynamic_actions = (
                query_policy.dynamic_ontology_actions(
                    {"include_graph": include_graph, "refresh_snapshot": refresh_snapshot}
                )
                if query_policy and query_policy.dynamic_ontology_actions
                else ()
            )
            for action_name in dynamic_actions:
                require_allowed(self.policy.check_action(actor, action_name, {"run_id": run_id}))
        except PolicyDenied as exc:
            _emit_ontology_read_audit(
                "ontology.query",
                actor=actor,
                status="denied",
                metadata={"intent": intent, "run_id": run_id, "include_graph": include_graph},
                error=exc.reason,
            )
            raise
        tf = timeframe if timeframe in VALID_TIMEFRAMES else "Daily"
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 100))
        auth_stats = _empty_auth_stats()

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
                require_allowed(self.policy.check_action(actor, OntologyAction.SNAPSHOT_REFRESH, {"timeframe": tf}))
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
        applied_filters = _query_filters_for_sql(effective_filters, interpreted.intent)
        page_data = self.repo.query_snapshot_positions_page(
            run_id=resolved_run_id,
            filters=applied_filters,
            page=safe_page,
            page_size=safe_page_size,
            schema_mode="upgraded",
        )
        rows = page_data["rows"]
        position_ids = [str(row.get("position_id") or "") for row in rows if row.get("position_id")]
        evidence_by_position = self.repo.fetch_snapshot_position_signal_evidence_batch(
            run_id=resolved_run_id,
            position_ids=position_ids,
            schema_mode="upgraded",
        )
        thesis_context = (
            self.repo.fetch_snapshot_position_thesis_context_batch(
                run_id=resolved_run_id,
                position_ids=position_ids,
                schema_mode="upgraded",
            )
            if include_graph or interpreted.intent == "thesis_review"
            else {}
        )
        results: list[dict[str, Any]] = []
        result_position_ids: list[str] = []
        visible_rows: list[dict[str, Any]] = []
        for row in rows:
            position_resource = _position_resource_from_row(row)
            if not self.policy.check_object(actor, position_resource).allowed:
                auth_stats["filtered_objects"] += 1
                continue
            raw_pos = _as_dict(row.get("position_props"))
            pos, redacted = redact_properties(actor, self.policy, position_resource, raw_pos)
            auth_stats["redacted_fields"] += redacted
            position_id = str(row.get("position_id") or "")
            ticker = str(pos.get("ticker")) if pos.get("ticker") is not None else None
            if ticker is None and _field_visible(actor, self.policy, position_resource, "ticker"):
                ticker = position_id.split(":")[-1]
            asset = _resolved_asset(row, pos, position_resource, actor, self.policy, auth_stats)
            direction = str(pos.get("direction")) if pos.get("direction") is not None else None
            risk_score = _to_float(pos.get("risk_score")) or 0.0
            risk_level = None
            if pos.get("risk_level") is not None:
                risk_level = str(pos.get("risk_level"))
            elif "risk_score" in pos and _field_visible(actor, self.policy, position_resource, "risk_level"):
                risk_level = _risk_level_from_score(risk_score)
            sector = _resolved_sector(row, position_resource, actor, self.policy, auth_stats)

            evidence = self._build_evidence_from_batch(
                evidence_by_position.get(position_id, []),
                actor=actor,
                position_resource=position_resource,
                auth_stats=auth_stats,
            )

            results.append(
                {
                    "ticker": ticker,
                    "asset": asset,
                    "direction": direction,
                    "sector": sector,
                    "risk_score": round(risk_score, 4) if "risk_score" in pos else None,
                    "risk_level": risk_level,
                    "evidence": evidence,
                }
            )
            result_position_ids.append(position_id)
            visible_rows.append(row)

        if interpreted.intent == "thesis_review":
            _enrich_with_thesis_context(
                results,
                position_ids=result_position_ids,
                thesis_context=thesis_context,
                actor=actor,
                policy=self.policy,
                auth_stats=auth_stats,
            )

        exact_totals = _has_exact_query_totals(actor, self.policy)
        aggregate = self.repo.aggregate_snapshot_positions(run_id=resolved_run_id, filters=applied_filters)
        aggregate["confidence"] = round(_compute_confidence(source_status, required_modules), 4)
        aggregate["exact"] = exact_totals
        _sanitize_aggregate_for_policy(actor, self.policy, aggregate)
        pagination_meta = _build_pagination_meta(
            page=safe_page,
            page_size=safe_page_size,
            returned_results=len(results),
            total_results=int(page_data["total_results"] or 0),
            exact_total=exact_totals,
        )
        response: dict[str, Any] = {
            "run_id": resolved_run_id,
            "intent": interpreted.intent,
            "interpreted_query": {
                "source": interpreted.source,
                "query": interpreted.original_query,
                "entity": interpreted.entity,
                "filters": applied_filters,
            },
            "as_of": as_of,
            "source_status": source_status,
            "results": results,
            "aggregate": aggregate,
        }

        if interpreted.intent == "temporal_comparison":
            diff = self._auto_temporal_diff(resolved_run_id, actor=actor)
            if diff:
                response["diff"] = diff

        if include_graph:
            raw_graph, graph_meta = _build_page_graph(
                visible_rows,
                evidence_by_position,
                thesis_context,
                run_id=resolved_run_id,
            )
            graph, graph_stats = filter_graph(
                actor,
                self.policy,
                raw_graph,
            )
            response["graph"] = graph
            _merge_auth_stats(auth_stats, graph_stats)
            graph_meta["node_count"] = len(graph.get("nodes", []))
            graph_meta["edge_count"] = len(graph.get("edges", []))
        else:
            graph_meta = None

        response["_meta"] = {
            "authorization": dict(auth_stats),
            "pagination": pagination_meta,
        }
        if graph_meta is not None:
            response["_meta"]["graph"] = graph_meta

        _emit_ontology_read_audit(
            "ontology.query",
            actor=actor,
            status="succeeded",
            object_refs=[{"type": "ontology_run", "id": resolved_run_id}],
            metadata={
                "intent": interpreted.intent,
                "include_graph": include_graph,
                "refresh_snapshot": refresh_snapshot,
                "page": safe_page,
                "page_size": safe_page_size,
            },
            after_summary={
                "run_id": resolved_run_id,
                "intent": interpreted.intent,
                "result_count": len(results),
                "total_results": pagination_meta["total_results"],
                "include_graph": include_graph,
                "page": safe_page,
                "page_size": safe_page_size,
                "authorization": dict(auth_stats),
            },
            source_lineage={"run_id": resolved_run_id, "as_of": as_of, "source_status": source_status},
        )
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

        return self.repo.snapshot_has_positions(run_id)

    def compare_snapshots(self, run_id_a: str, run_id_b: str, actor: Actor | None = None) -> dict[str, Any]:
        """Diff two ontology snapshots. Returns position changes, risk score deltas, and signal transitions."""
        actor = actor or admin_actor(source="service")
        try:
            require_allowed(
                self.policy.check_action(
                    actor,
                    OntologyAction.SNAPSHOTS_COMPARE,
                    {"run_id_before": run_id_a, "run_id_after": run_id_b},
                )
            )
        except PolicyDenied as exc:
            _emit_ontology_read_audit(
                "ontology.snapshots.compare",
                actor=actor,
                status="denied",
                object_refs=[{"type": "ontology_run", "id": run_id_a}, {"type": "ontology_run", "id": run_id_b}],
                error=exc.reason,
            )
            raise
        auth_stats = _empty_auth_stats()
        run_a = self.repo.get_run(run_id_a)
        run_b = self.repo.get_run(run_id_b)
        if run_a is None:
            raise OntologyRunNotFoundError(run_id_a)
        if run_b is None:
            raise OntologyRunNotFoundError(run_id_b)

        rows_a = self.repo.fetch_snapshot_position_asset_sector_rows(run_id=run_id_a, schema_mode="upgraded")
        rows_b = self.repo.fetch_snapshot_position_asset_sector_rows(run_id=run_id_b, schema_mode="upgraded")

        def _positions_map(rows: list[dict]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for row in rows:
                resource = _position_resource_from_row(row)
                if not self.policy.check_object(actor, resource).allowed:
                    auth_stats["filtered_objects"] += 1
                    continue
                pos, redacted = redact_properties(actor, self.policy, resource, _as_dict(row.get("position_props")))
                auth_stats["redacted_fields"] += redacted
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

        can_read_risk_score = _position_field_visible(actor, self.policy, "risk_score")
        for ticker in common:
            pa = positions_a[ticker]
            pb = positions_b[ticker]
            score_a = _to_float(pa.get("risk_score")) or 0.0
            score_b = _to_float(pb.get("risk_score")) or 0.0
            delta = round(score_b - score_a, 4)

            if can_read_risk_score and abs(delta) >= 0.02:
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

        response = {
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
            "_meta": {"authorization": dict(auth_stats)},
        }
        _emit_ontology_read_audit(
            "ontology.snapshots.compare",
            actor=actor,
            status="succeeded",
            object_refs=[{"type": "ontology_run", "id": run_id_a}, {"type": "ontology_run", "id": run_id_b}],
            after_summary={
                "run_id_before": run_id_a,
                "run_id_after": run_id_b,
                "positions_added_count": len(added),
                "positions_removed_count": len(removed),
                "risk_change_count": len(risk_changes),
                "signal_transition_count": len(signal_transitions),
                "authorization": dict(auth_stats),
            },
        )
        return response

    def _auto_temporal_diff(self, current_run_id: str, *, actor: Actor | None = None) -> dict[str, Any] | None:
        """Find the most recent prior run and diff against it."""
        runs = self.list_runs(limit=10, actor=actor)
        prior_run_id: str | None = None
        for run in runs:
            rid = str(run.get("run_id", ""))
            if rid and rid != current_run_id:
                prior_run_id = rid
                break
        if not prior_run_id:
            return None
        try:
            return self.compare_snapshots(prior_run_id, current_run_id, actor=actor)
        except Exception:
            return None

    def _build_evidence(self, position_id: str, run_id: str, actor: Actor | None = None) -> list[dict[str, Any]]:
        raw = self.repo.fetch_snapshot_position_signal_evidence_batch(
            run_id=run_id,
            position_ids=[position_id],
            schema_mode="upgraded",
        ).get(position_id, [])
        position_resource = NodeResource(id=position_id, type="Position")
        return self._build_evidence_from_batch(raw, actor=actor, position_resource=position_resource)

    def _build_evidence_from_batch(
        self,
        raw: list[dict[str, Any]],
        actor: Actor | None = None,
        position_resource: NodeResource | None = None,
        auth_stats: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        actor = actor or admin_actor(source="service")
        auth_stats = auth_stats if auth_stats is not None else _empty_auth_stats()
        evidence = []
        for row in raw:
            signal_resource = _signal_resource_from_evidence(row)
            if not self.policy.check_object(actor, signal_resource).allowed:
                auth_stats["filtered_objects"] += 1
                continue
            edge_resource = EdgeResource(
                source_id=(position_resource.id if position_resource is not None else ""),
                target_id=signal_resource.id,
                relation_type="exposed_to_signal",
                properties=_as_dict(row.get("edge_props")),
                schema_name=str(row["edge_schema_name"]) if row.get("edge_schema_name") is not None else None,
                schema_version=int(row["edge_schema_version"]) if row.get("edge_schema_version") is not None else None,
            )
            if not self.policy.check_relationship(
                actor, edge_resource, source=position_resource, target=signal_resource
            ).allowed:
                auth_stats["filtered_relationships"] += 1
                continue
            edge, redacted = redact_properties(actor, self.policy, edge_resource, edge_resource.properties)
            auth_stats["redacted_fields"] += redacted
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


def _empty_auth_stats() -> dict[str, int]:
    return {"filtered_objects": 0, "filtered_relationships": 0, "redacted_fields": 0}


def _emit_ontology_read_audit(
    action_name: str,
    *,
    actor: Actor,
    status: str,
    object_refs: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    after_summary: dict[str, Any] | None = None,
    source_lineage: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    emit_audit_event(
        action_name,
        "ontology_read",
        status,
        actor=actor,
        object_refs=object_refs,
        metadata=metadata,
        after_summary=after_summary,
        source_lineage=source_lineage,
        error=error,
    )


def _merge_auth_stats(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value or 0)


def _field_visible(
    actor: Actor | None,
    policy: OntologyPolicy,
    resource: NodeResource | EdgeResource,
    field_name: str,
) -> bool:
    allowed = policy.allowed_fields(actor, resource)
    return allowed is None or field_name in allowed


def _position_field_visible(actor: Actor | None, policy: OntologyPolicy, field_name: str) -> bool:
    return _field_visible(actor, policy, NodeResource(id="position:*", type="Position"), field_name)


def _position_resource_from_row(row: dict[str, Any]) -> NodeResource:
    position_id = str(row.get("position_id") or "")
    return NodeResource(
        id=position_id,
        type="Position",
        label=str(row.get("position_label") or position_id or ""),
        properties=_as_dict(row.get("position_props")),
        schema_name=str(row["position_schema_name"]) if row.get("position_schema_name") is not None else None,
        schema_version=(
            int(row["position_schema_version"]) if row.get("position_schema_version") is not None else None
        ),
    )


def _asset_resource_from_row(row: dict[str, Any]) -> NodeResource | None:
    asset_id = row.get("asset_id")
    if asset_id is None:
        return None
    return NodeResource(
        id=str(asset_id),
        type="Asset",
        label=str(row.get("asset_label") or asset_id),
        properties=_as_dict(row.get("asset_props")),
        schema_name=str(row["asset_schema_name"]) if row.get("asset_schema_name") is not None else None,
        schema_version=int(row["asset_schema_version"]) if row.get("asset_schema_version") is not None else None,
    )


def _sector_resource_from_row(row: dict[str, Any]) -> NodeResource | None:
    sector_id = row.get("sector_id")
    sector_props = _as_dict(row.get("sector_props"))
    if sector_id is None and not sector_props:
        return None
    fallback_id = f"sector:{str(sector_props.get('name') or 'unknown').lower().replace(' ', '_')}"
    return NodeResource(
        id=str(sector_id or fallback_id),
        type="Sector",
        label=str(row.get("sector_label") or sector_props.get("name") or sector_id or fallback_id),
        properties=sector_props,
        schema_name=str(row["sector_schema_name"]) if row.get("sector_schema_name") is not None else None,
        schema_version=int(row["sector_schema_version"]) if row.get("sector_schema_version") is not None else None,
    )


def _signal_resource_from_evidence(row: dict[str, Any]) -> NodeResource:
    signal_id = str(row.get("signal_id") or "")
    return NodeResource(
        id=signal_id,
        type="Signal",
        label=str(row.get("signal_label") or signal_id),
        properties=_as_dict(row.get("signal_props")),
        schema_name=str(row["signal_schema_name"]) if row.get("signal_schema_name") is not None else None,
        schema_version=int(row["signal_schema_version"]) if row.get("signal_schema_version") is not None else None,
    )


def _relationship_allowed(
    actor: Actor | None,
    policy: OntologyPolicy,
    edge: EdgeResource,
    source: NodeResource | None,
    target: NodeResource | None,
    stats: dict[str, int],
) -> bool:
    if policy.check_relationship(actor, edge, source=source, target=target).allowed:
        return True
    stats["filtered_relationships"] += 1
    return False


def _resolved_asset(
    row: dict[str, Any],
    pos: dict[str, Any],
    position_resource: NodeResource,
    actor: Actor,
    policy: OntologyPolicy,
    stats: dict[str, int],
) -> str | None:
    if not _field_visible(actor, policy, position_resource, "asset"):
        return None
    asset_resource = _asset_resource_from_row(row)
    if asset_resource is None:
        return str(pos.get("asset")) if pos.get("asset") is not None else "unknown"
    if not policy.check_object(actor, asset_resource).allowed:
        stats["filtered_objects"] += 1
        return None
    edge_props = _as_dict(row.get("position_asset_edge_props")) or {"ontology_run_id": pos.get("ontology_run_id")}
    edge = EdgeResource(
        source_id=position_resource.id,
        target_id=asset_resource.id,
        relation_type="references_asset",
        properties=edge_props,
        schema_name=(
            str(row["position_asset_edge_schema_name"])
            if row.get("position_asset_edge_schema_name") is not None
            else None
        ),
        schema_version=(
            int(row["position_asset_edge_schema_version"])
            if row.get("position_asset_edge_schema_version") is not None
            else None
        ),
    )
    if not _relationship_allowed(actor, policy, edge, position_resource, asset_resource, stats):
        return None
    return (
        str(pos.get("asset"))
        if pos.get("asset") is not None
        else str(asset_resource.properties.get("asset") or "unknown")
    )


def _resolved_sector(
    row: dict[str, Any],
    position_resource: NodeResource,
    actor: Actor,
    policy: OntologyPolicy,
    stats: dict[str, int],
) -> str | None:
    sector_resource = _sector_resource_from_row(row)
    if sector_resource is None:
        return "Unknown Equity"
    if not policy.check_object(actor, sector_resource).allowed:
        stats["filtered_objects"] += 1
        return None
    asset_resource = _asset_resource_from_row(row)
    if asset_resource is not None and row.get("sector_id") is not None:
        edge_props = _as_dict(row.get("asset_sector_edge_props")) or {
            "ontology_run_id": position_resource.properties.get("ontology_run_id")
        }
        edge = EdgeResource(
            source_id=asset_resource.id,
            target_id=sector_resource.id,
            relation_type="belongs_to_sector",
            properties=edge_props,
            schema_name=(
                str(row["asset_sector_edge_schema_name"])
                if row.get("asset_sector_edge_schema_name") is not None
                else None
            ),
            schema_version=(
                int(row["asset_sector_edge_schema_version"])
                if row.get("asset_sector_edge_schema_version") is not None
                else None
            ),
        )
        if not _relationship_allowed(actor, policy, edge, asset_resource, sector_resource, stats):
            return None
    sector_props, redacted = redact_properties(actor, policy, sector_resource, sector_resource.properties)
    stats["redacted_fields"] += redacted
    if not _field_visible(actor, policy, sector_resource, "name"):
        return None
    return str(sector_props.get("name")) if isinstance(sector_props.get("name"), str) else "Unknown Equity"


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
    actor: Actor | None = None,
    policy: OntologyPolicy | None = None,
    auth_stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    actor = actor or admin_actor(source="service")
    policy = policy or DEFAULT_ONTOLOGY_POLICY
    auth_stats = auth_stats if auth_stats is not None else _empty_auth_stats()
    position_ids = [f"position:{str(result.get('ticker') or '').upper()}" for result in results if result.get("ticker")]
    thesis_context = repo.fetch_snapshot_position_thesis_context_batch(
        run_id=run_id,
        position_ids=position_ids,
        schema_mode="upgraded",
    )
    _enrich_with_thesis_context(
        results,
        position_ids=position_ids,
        thesis_context=thesis_context,
        actor=actor,
        policy=policy,
        auth_stats=auth_stats,
    )
    return results


def _query_filters_for_sql(filters: dict[str, Any], intent: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("tickers", "sectors", "assets"):
        value = filters.get(key)
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                out[key] = cleaned
    min_risk = _to_float(filters.get("min_risk_score"))
    if intent == "positions_in_deteriorating_macro":
        min_risk = max(min_risk or 0.0, 0.6)
    if min_risk is not None:
        out["min_risk_score"] = round(min_risk, 4)
    return out


def _has_exact_query_totals(actor: Actor | None, policy: OntologyPolicy) -> bool:
    if policy is not DEFAULT_ONTOLOGY_POLICY or actor is None:
        return False
    roles = {role.lower() for role in actor.roles}
    return actor.actor_type == "system" or "admin" in roles


def _build_pagination_meta(
    *,
    page: int,
    page_size: int,
    returned_results: int,
    total_results: int,
    exact_total: bool,
) -> dict[str, Any]:
    total_pages = (total_results + page_size - 1) // page_size if page_size > 0 else 0
    return {
        "page": page,
        "page_size": page_size,
        "returned_results": returned_results,
        "total_results": total_results,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "sort": "risk_score_desc_then_position_id_asc",
        "exact_total": exact_total,
    }


def _sanitize_aggregate_for_policy(actor: Actor | None, policy: OntologyPolicy, aggregate: dict[str, Any]) -> None:
    if not _position_field_visible(actor, policy, "risk_score"):
        aggregate["average_risk_score"] = 0.0
    if not _position_field_visible(actor, policy, "risk_level"):
        aggregate["risk_buckets"] = {"high": 0, "medium": 0, "low": 0}
    if not _position_field_visible(actor, policy, "asset"):
        aggregate["asset_exposure_counts"] = {}


def _enrich_with_thesis_context(
    results: list[dict[str, Any]],
    *,
    position_ids: list[str],
    thesis_context: dict[str, dict[str, Any]],
    actor: Actor | None,
    policy: OntologyPolicy,
    auth_stats: dict[str, int],
) -> None:
    for position_id, result in zip(position_ids, results, strict=False):
        context = thesis_context.get(position_id) if isinstance(thesis_context, dict) else None
        if not isinstance(context, dict):
            result["thesis"] = None
            result["latest_evaluation"] = None
            continue

        thesis_bundle = context.get("thesis")
        thesis_node = thesis_bundle.get("node") if isinstance(thesis_bundle, dict) else None
        thesis_edge = thesis_bundle.get("edge") if isinstance(thesis_bundle, dict) else None
        if not _graph_node_visible(actor, policy, thesis_node, auth_stats):
            result["thesis"] = None
            result["latest_evaluation"] = None
            continue
        if not _graph_edge_visible(
            actor, policy, thesis_edge, source_id=position_id, target=thesis_node, auth_stats=auth_stats
        ):
            result["thesis"] = None
            result["latest_evaluation"] = None
            continue

        thesis_props = _graph_node_properties(actor, policy, thesis_node, auth_stats)
        result["thesis"] = {
            "status": thesis_props.get("status"),
            "created_at": thesis_props.get("created_at"),
            "updated_at": thesis_props.get("updated_at"),
        }

        evaluations = context.get("evaluations") if isinstance(context.get("evaluations"), list) else []
        latest = _select_latest_visible_evaluation(
            evaluations,
            actor=actor,
            policy=policy,
            thesis_node=thesis_node,
            auth_stats=auth_stats,
        )
        if latest is None:
            result["latest_evaluation"] = None
            continue
        eval_props = _graph_node_properties(actor, policy, latest["node"], auth_stats)
        result["latest_evaluation"] = {
            "evaluated_at": eval_props.get("evaluated_at"),
            "thesis_status": eval_props.get("thesis_status"),
            "technical_read": eval_props.get("technical_read"),
            "fundamental_read": eval_props.get("fundamental_read"),
            "action": eval_props.get("action"),
            "confidence": eval_props.get("confidence"),
            "risk_flag": eval_props.get("risk_flag"),
        }


def _select_latest_visible_evaluation(
    evaluations: list[dict[str, Any]],
    *,
    actor: Actor | None,
    policy: OntologyPolicy,
    thesis_node: dict[str, Any],
    auth_stats: dict[str, int],
) -> dict[str, Any] | None:
    visible: list[dict[str, Any]] = []
    for item in evaluations:
        node = item.get("node") if isinstance(item, dict) else None
        edge = item.get("edge") if isinstance(item, dict) else None
        if not _graph_node_visible(actor, policy, node, auth_stats):
            continue
        if not _graph_edge_visible(actor, policy, edge, source=thesis_node, target=node, auth_stats=auth_stats):
            continue
        visible.append(item)
    if not visible:
        return None
    visible.sort(key=_evaluation_sort_key, reverse=True)
    return visible[0]


def _evaluation_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    node = item.get("node") if isinstance(item, dict) else {}
    props = _as_dict(node.get("properties") if isinstance(node, dict) else {})
    return (str(props.get("evaluated_at") or ""), str(node.get("id") or ""))


def _build_page_graph(
    rows: list[dict[str, Any]],
    evidence_by_position: dict[str, list[dict[str, Any]]],
    thesis_context: dict[str, dict[str, Any]],
    *,
    run_id: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    builder = _PageGraphBuilder(max_nodes=GRAPH_PAGE_NODE_LIMIT, max_edges=GRAPH_PAGE_EDGE_LIMIT)

    for row in rows:
        builder.add_node(_row_graph_node(row, kind="position"))

    for row in rows:
        builder.add_node(_row_graph_node(row, kind="asset"))
        builder.add_edge(_row_graph_edge(row, kind="position_asset", run_id=run_id))
        builder.add_node(_row_graph_node(row, kind="sector"))
        builder.add_edge(_row_graph_edge(row, kind="asset_sector", run_id=run_id))

    for row in rows:
        position_id = str(row.get("position_id") or "")
        for evidence in evidence_by_position.get(position_id, []):
            builder.add_node(_signal_graph_node(evidence))
            builder.add_edge(_signal_graph_edge(position_id, evidence))

    for row in rows:
        position_id = str(row.get("position_id") or "")
        context = thesis_context.get(position_id) if isinstance(thesis_context, dict) else None
        if not isinstance(context, dict):
            continue
        thesis_bundle = context.get("thesis")
        if isinstance(thesis_bundle, dict):
            builder.add_node(thesis_bundle.get("node"))
            builder.add_edge(thesis_bundle.get("edge"))
        for evaluation in context.get("evaluations") if isinstance(context.get("evaluations"), list) else []:
            if isinstance(evaluation, dict):
                builder.add_node(evaluation.get("node"))
                builder.add_edge(evaluation.get("edge"))
        for catalyst in context.get("catalysts") if isinstance(context.get("catalysts"), list) else []:
            if isinstance(catalyst, dict):
                builder.add_node(catalyst.get("node"))
                builder.add_edge(catalyst.get("edge"))

    return builder.graph(), {
        "scope": "page",
        "node_count": len(builder.nodes),
        "edge_count": len(builder.edges),
        "truncated": builder.truncated,
        "max_nodes": GRAPH_PAGE_NODE_LIMIT,
        "max_edges": GRAPH_PAGE_EDGE_LIMIT,
    }


class _PageGraphBuilder:
    def __init__(self, *, max_nodes: int, max_edges: int):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self._node_ids: set[str] = set()
        self._edge_keys: set[tuple[str, str, str]] = set()
        self.truncated = False

    def add_node(self, node: dict[str, Any] | None) -> bool:
        if not isinstance(node, dict):
            return False
        node_id = str(node.get("id") or "")
        if not node_id:
            return False
        if node_id in self._node_ids:
            return True
        if len(self.nodes) >= self.max_nodes:
            self.truncated = True
            return False
        self._node_ids.add(node_id)
        self.nodes.append(dict(node))
        return True

    def add_edge(self, edge: dict[str, Any] | None) -> bool:
        if not isinstance(edge, dict):
            return False
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        relation_type = str(edge.get("relation_type") or "")
        if not source_id or not target_id or not relation_type:
            return False
        if source_id not in self._node_ids or target_id not in self._node_ids:
            return False
        key = (source_id, target_id, relation_type)
        if key in self._edge_keys:
            return True
        if len(self.edges) >= self.max_edges:
            self.truncated = True
            return False
        self._edge_keys.add(key)
        self.edges.append(dict(edge))
        return True

    def graph(self) -> dict[str, list[dict[str, Any]]]:
        return {"nodes": self.nodes, "edges": self.edges}


def _row_graph_node(row: dict[str, Any], *, kind: str) -> dict[str, Any] | None:
    if kind == "position":
        node_id = row.get("position_id")
        if node_id is None:
            return None
        return {
            "id": str(node_id),
            "type": "Position",
            "label": str(row.get("position_label") or node_id),
            "properties": _as_dict(row.get("position_props")),
            "schema_name": row.get("position_schema_name"),
            "schema_version": row.get("position_schema_version"),
            "updated_at": row.get("position_updated_at"),
        }
    if kind == "asset":
        node_id = row.get("asset_id")
        if node_id is None:
            return None
        return {
            "id": str(node_id),
            "type": "Asset",
            "label": str(row.get("asset_label") or node_id),
            "properties": _as_dict(row.get("asset_props")),
            "schema_name": row.get("asset_schema_name"),
            "schema_version": row.get("asset_schema_version"),
            "updated_at": row.get("asset_updated_at"),
        }
    if kind == "sector":
        node_id = row.get("sector_id")
        if node_id is None:
            return None
        return {
            "id": str(node_id),
            "type": "Sector",
            "label": str(row.get("sector_label") or node_id),
            "properties": _as_dict(row.get("sector_props")),
            "schema_name": row.get("sector_schema_name"),
            "schema_version": row.get("sector_schema_version"),
            "updated_at": row.get("sector_updated_at"),
        }
    return None


def _row_graph_edge(row: dict[str, Any], *, kind: str, run_id: str) -> dict[str, Any] | None:
    if kind == "position_asset":
        source_id = row.get("position_id")
        target_id = row.get("asset_id")
        if source_id is None or target_id is None:
            return None
        return {
            "source_id": str(source_id),
            "target_id": str(target_id),
            "relation_type": "references_asset",
            "properties": _as_dict(row.get("position_asset_edge_props")) or {"ontology_run_id": run_id},
            "schema_name": row.get("position_asset_edge_schema_name"),
            "schema_version": row.get("position_asset_edge_schema_version"),
            "relation_schema_name": row.get("position_asset_edge_relation_schema_name"),
            "relation_schema_version": row.get("position_asset_edge_relation_schema_version"),
            "updated_at": row.get("position_asset_edge_updated_at"),
        }
    if kind == "asset_sector":
        source_id = row.get("asset_id")
        target_id = row.get("sector_id")
        if source_id is None or target_id is None:
            return None
        return {
            "source_id": str(source_id),
            "target_id": str(target_id),
            "relation_type": "belongs_to_sector",
            "properties": _as_dict(row.get("asset_sector_edge_props")) or {"ontology_run_id": run_id},
            "schema_name": row.get("asset_sector_edge_schema_name"),
            "schema_version": row.get("asset_sector_edge_schema_version"),
            "relation_schema_name": row.get("asset_sector_edge_relation_schema_name"),
            "relation_schema_version": row.get("asset_sector_edge_relation_schema_version"),
            "updated_at": row.get("asset_sector_edge_updated_at"),
        }
    return None


def _signal_graph_node(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("signal_id") or ""),
        "type": "Signal",
        "label": str(row.get("signal_label") or row.get("signal_id") or ""),
        "properties": _as_dict(row.get("signal_props")),
        "schema_name": row.get("signal_schema_name"),
        "schema_version": row.get("signal_schema_version"),
        "updated_at": row.get("signal_updated_at"),
    }


def _signal_graph_edge(position_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": position_id,
        "target_id": str(row.get("signal_id") or ""),
        "relation_type": "exposed_to_signal",
        "properties": _as_dict(row.get("edge_props")),
        "schema_name": row.get("edge_schema_name"),
        "schema_version": row.get("edge_schema_version"),
        "relation_schema_name": row.get("edge_relation_schema_name"),
        "relation_schema_version": row.get("edge_relation_schema_version"),
        "updated_at": row.get("edge_updated_at"),
    }


def _graph_node_visible(
    actor: Actor | None,
    policy: OntologyPolicy,
    node: dict[str, Any] | None,
    auth_stats: dict[str, int],
) -> bool:
    if not isinstance(node, dict):
        return False
    resource = NodeResource(
        id=str(node.get("id") or ""),
        type=str(node.get("type") or ""),
        label=str(node["label"]) if node.get("label") is not None else None,
        properties=_as_dict(node.get("properties")),
        schema_name=str(node["schema_name"]) if node.get("schema_name") is not None else None,
        schema_version=int(node["schema_version"]) if node.get("schema_version") is not None else None,
    )
    if policy.check_object(actor, resource).allowed:
        return True
    auth_stats["filtered_objects"] += 1
    return False


def _graph_edge_visible(
    actor: Actor | None,
    policy: OntologyPolicy,
    edge: dict[str, Any] | None,
    *,
    source: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
    source_id: str | None = None,
    auth_stats: dict[str, int],
) -> bool:
    if not isinstance(edge, dict):
        return False
    edge_resource = EdgeResource(
        source_id=str(edge.get("source_id") or source_id or ""),
        target_id=str(edge.get("target_id") or ""),
        relation_type=str(edge.get("relation_type") or ""),
        properties=_as_dict(edge.get("properties")),
        schema_name=str(edge["schema_name"]) if edge.get("schema_name") is not None else None,
        schema_version=int(edge["schema_version"]) if edge.get("schema_version") is not None else None,
    )
    source_resource = (
        _node_resource_from_graph_node(source)
        if isinstance(source, dict)
        else (NodeResource(id=source_id, type="Position") if source_id else None)
    )
    target_resource = _node_resource_from_graph_node(target) if isinstance(target, dict) else None
    if policy.check_relationship(actor, edge_resource, source=source_resource, target=target_resource).allowed:
        return True
    auth_stats["filtered_relationships"] += 1
    return False


def _graph_node_properties(
    actor: Actor | None,
    policy: OntologyPolicy,
    node: dict[str, Any] | None,
    auth_stats: dict[str, int],
) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    resource = _node_resource_from_graph_node(node)
    props, redacted = redact_properties(actor, policy, resource, _as_dict(node.get("properties")))
    auth_stats["redacted_fields"] += redacted
    return props


def _node_resource_from_graph_node(node: dict[str, Any]) -> NodeResource:
    return NodeResource(
        id=str(node.get("id") or ""),
        type=str(node.get("type") or ""),
        label=str(node["label"]) if node.get("label") is not None else None,
        properties=_as_dict(node.get("properties")),
        schema_name=str(node["schema_name"]) if node.get("schema_name") is not None else None,
        schema_version=int(node["schema_version"]) if node.get("schema_version") is not None else None,
    )


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
