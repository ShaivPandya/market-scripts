"""Postgres temporal read models for ontology semantic queries."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from api.postgres import connect

ConnectionFactory = Callable[[], Any]

TEMPORAL_READ_MODEL_RUN_ID = "temporal:read_model"
OPERATIONAL_READ_MODEL_VIEW = "ontology_current_operational_object_read_model"


class TemporalReadModelUnavailable(RuntimeError):
    """Raised when temporal read-model tables or connections are unavailable."""


class TemporalReadModelRepository:
    """Read optimized ontology projections backed by authoritative temporal tables."""

    def __init__(self, connection_factory: ConnectionFactory | None = None):
        self._connection_factory = connection_factory or connect

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        with self._connection_factory() as conn:
            yield conn

    def refresh(self) -> None:
        """Refresh the current temporal read models after ontology object/relation writes."""
        with self._connect() as conn:
            conn.execute("SELECT refresh_ontology_temporal_read_models()")
            commit = getattr(conn, "commit", None)
            if callable(commit):
                commit()

    def query_positions_page(
        self,
        *,
        filters: Mapping[str, Any] | None,
        page: int,
        page_size: int,
        as_of: str | None = None,
        tx_as_of: str | None = None,
        include_history: bool = False,
    ) -> dict[str, Any]:
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 100))
        offset = (safe_page - 1) * safe_page_size
        source_sql, source_params = _position_source_sql(
            as_of=as_of,
            tx_as_of=tx_as_of,
            include_history=include_history,
        )
        where_sql, where_params = _position_filter_sql(filters)
        params = [*source_params, *where_params]

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS total_results FROM ({source_sql}) rm WHERE {where_sql}",
                tuple(params),
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT *
                FROM ({source_sql}) rm
                WHERE {where_sql}
                ORDER BY risk_score_value DESC NULLS LAST, position_id ASC
                LIMIT %s OFFSET %s
                """,
                tuple([*params, safe_page_size, offset]),
            ).fetchall()

        return {
            "rows": [_position_row(row) for row in rows],
            "total_results": int(_row_value(total, "total_results", 0) or 0),
            "page": safe_page,
            "page_size": safe_page_size,
        }

    def aggregate_positions(
        self,
        *,
        filters: Mapping[str, Any] | None,
        as_of: str | None = None,
        tx_as_of: str | None = None,
        include_history: bool = False,
    ) -> dict[str, Any]:
        source_sql, source_params = _position_source_sql(
            as_of=as_of,
            tx_as_of=tx_as_of,
            include_history=include_history,
        )
        where_sql, where_params = _position_filter_sql(filters)
        params = [*source_params, *where_params]

        with self._connect() as conn:
            counts = conn.execute(
                f"""
                SELECT
                  COUNT(*) AS position_count,
                  SUM(CASE WHEN risk_score_value >= 0.75 THEN 1 ELSE 0 END) AS high_count,
                  SUM(CASE WHEN risk_score_value >= 0.5 AND risk_score_value < 0.75 THEN 1 ELSE 0 END)
                    AS medium_count,
                  SUM(CASE WHEN risk_score_value < 0.5 OR risk_score_value IS NULL THEN 1 ELSE 0 END)
                    AS low_count,
                  AVG(risk_score_value) AS average_risk_score
                FROM ({source_sql}) rm
                WHERE {where_sql}
                """,
                tuple(params),
            ).fetchone()
            asset_rows = conn.execute(
                f"""
                SELECT COALESCE(asset, 'unknown') AS asset_name, COUNT(*) AS asset_count
                FROM ({source_sql}) rm
                WHERE {where_sql}
                GROUP BY COALESCE(asset, 'unknown')
                ORDER BY COALESCE(asset, 'unknown')
                """,
                tuple(params),
            ).fetchall()

        return {
            "position_count": int(_row_value(counts, "position_count", 0) or 0),
            "risk_buckets": {
                "high": int(_row_value(counts, "high_count", 0) or 0),
                "medium": int(_row_value(counts, "medium_count", 0) or 0),
                "low": int(_row_value(counts, "low_count", 0) or 0),
            },
            "asset_exposure_counts": {
                str(_row_value(row, "asset_name", "unknown") or "unknown"): int(_row_value(row, "asset_count", 0) or 0)
                for row in asset_rows
            },
            "average_risk_score": round(float(_row_value(counts, "average_risk_score", 0.0) or 0.0), 4),
        }

    def fetch_position_signal_evidence_batch(
        self,
        position_ids: Sequence[str],
        *,
        as_of: str | None = None,
        tx_as_of: str | None = None,
        include_history: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        normalized_ids = _normalized_ids(position_ids)
        if not normalized_ids:
            return {}
        source_sql, params = _evidence_source_sql(as_of=as_of, tx_as_of=tx_as_of, include_history=include_history)
        where, id_params = _in_clause("position_id", normalized_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM ({source_sql}) ev
                WHERE {where}
                ORDER BY position_id, signal_id
                """,
                tuple([*params, *id_params]),
            ).fetchall()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            item = _row_dict(row)
            item.setdefault("edge_relation_schema_name", item.get("relation_schema_name"))
            item.setdefault("edge_relation_schema_version", item.get("relation_schema_version"))
            position_id = str(item.get("position_id") or "")
            grouped.setdefault(position_id, []).append(item)
        return grouped

    def fetch_position_thesis_context_batch(
        self,
        position_ids: Sequence[str],
        *,
        as_of: str | None = None,
        tx_as_of: str | None = None,
        include_history: bool = False,
    ) -> dict[str, dict[str, Any]]:
        normalized_ids = _normalized_ids(position_ids)
        if not normalized_ids:
            return {}
        source_sql, params = _thesis_source_sql(as_of=as_of, tx_as_of=tx_as_of, include_history=include_history)
        where, id_params = _in_clause("position_id", normalized_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM ({source_sql}) tc
                WHERE {where}
                ORDER BY position_id, context_type, target_id
                """,
                tuple([*params, *id_params]),
            ).fetchall()

        grouped: dict[str, dict[str, Any]] = {
            position_id: {"evaluations": [], "catalysts": []} for position_id in normalized_ids
        }
        for row_raw in rows:
            row = _row_dict(row_raw)
            position_id = str(row.get("position_id") or "")
            context_type = str(row.get("context_type") or "")
            bundle = {
                "node": _context_node(row),
                "edge": _context_edge(row),
            }
            if context_type == "thesis":
                grouped.setdefault(position_id, {"evaluations": [], "catalysts": []})["thesis"] = bundle
            elif context_type == "evaluation":
                grouped.setdefault(position_id, {"evaluations": [], "catalysts": []})["evaluations"].append(bundle)
            elif context_type == "catalyst":
                grouped.setdefault(position_id, {"evaluations": [], "catalysts": []})["catalysts"].append(bundle)
        return grouped

    def fetch_workspace_bundle(self) -> dict[str, Any]:
        """Fetch the ontology-backed workspace landing-page payload in bounded indexed reads."""
        with self._connect() as conn:
            latest_evaluations = _latest_by_ticker(
                _fetch_operational_objects(
                    conn,
                    "Evaluation",
                    limit=1000,
                    order_by="evaluated_at_sort DESC, updated_sort DESC, object_uid ASC",
                )
            )
            theses = _fetch_operational_objects(
                conn,
                "Thesis",
                limit=1000,
                order_by="ticker ASC NULLS LAST, updated_sort DESC, object_uid ASC",
            )
            pending_approvals = _fetch_operational_objects(
                conn,
                "Approval",
                filters={"status": "pending"},
                limit=200,
                order_by="created_at_sort DESC, updated_sort DESC, object_uid ASC",
            )
            latest_daily_recommendation = _first_row(
                _fetch_operational_objects(
                    conn,
                    "Recommendation",
                    filters={"report_type": "daily"},
                    limit=1,
                    order_by="as_of_sort DESC, updated_sort DESC, object_uid ASC",
                )
            )
            latest_weekly_recommendation = _first_row(
                _fetch_operational_objects(
                    conn,
                    "Recommendation",
                    filters={"report_type": "weekly"},
                    limit=1,
                    order_by="as_of_sort DESC, updated_sort DESC, object_uid ASC",
                )
            )
            pending_actionable_recommendations = _fetch_operational_objects(
                conn,
                "Recommendation",
                filters={"approval_status": "pending"},
                limit=5,
                order_by="as_of_sort DESC, updated_sort DESC, object_uid ASC",
            )
            pending_course_of_actions = _fetch_operational_objects(
                conn,
                "CourseOfAction",
                filters={"approval_status": "pending"},
                limit=10,
                order_by="as_of_sort DESC, updated_sort DESC, object_uid ASC",
            )
            recent_course_of_actions = _fetch_operational_objects(
                conn,
                "CourseOfAction",
                limit=10,
                order_by="as_of_sort DESC, updated_sort DESC, object_uid ASC",
            )
            open_course_of_action_comparisons = _fetch_operational_objects(
                conn,
                "CourseOfActionComparison",
                filters={"status": "open"},
                limit=10,
                order_by="as_of_sort DESC, updated_sort DESC, object_uid ASC",
            )
            open_action_items = _fetch_operational_objects(
                conn,
                "ActionItem",
                filters={"status": "open"},
                limit=100,
                order_by="created_at_sort DESC, updated_sort DESC, object_uid ASC",
            )
            optimizer_alerts = _fetch_operational_objects(
                conn,
                "OptimizationAlert",
                filters={"status": "open"},
                limit=5,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            _attach_optimization_alert_context(conn, optimizer_alerts)
            active_monitor_definitions = _fetch_operational_objects(
                conn,
                "MonitorDefinition",
                filters={"status": "active"},
                limit=100,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            active_mission_definitions = _fetch_operational_objects(
                conn,
                "MissionDefinition",
                filters={"status": "active"},
                limit=100,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            active_watch_triggers = _fetch_operational_objects(
                conn,
                "WatchTrigger",
                filters={"status": "active"},
                limit=100,
                order_by="created_at_sort DESC, updated_sort DESC, object_uid ASC",
            )
            recent_monitor_hits = _fetch_operational_objects(
                conn,
                "MonitorHit",
                filters={"status": "open"},
                limit=20,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            open_opportunity_candidates = _fetch_operational_objects(
                conn,
                "OpportunityCandidate",
                filters={"status": "open"},
                limit=50,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            recent_workflow_runs = _fetch_operational_objects(
                conn,
                "WorkflowRun",
                limit=3,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            recent_report_runs = _fetch_operational_objects(
                conn,
                "ReportRun",
                limit=5,
                order_by="as_of_sort DESC, updated_sort DESC, object_uid ASC",
            )
            challenged_claims = _fetch_operational_objects(
                conn,
                "ThesisClaim",
                filters={"status": "challenged"},
                limit=5,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            disconfirmed_claims = _fetch_operational_objects(
                conn,
                "ThesisClaim",
                filters={"status": "disconfirmed"},
                limit=5,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            pending_draft_decision_outcomes = _fetch_operational_objects(
                conn,
                "DecisionOutcome",
                filters={"final_label_status": "draft", "outcome_status": "evaluated"},
                limit=10,
                order_by="as_of_sort DESC, updated_sort DESC, object_uid ASC",
            )
            recent_finalized_decision_outcomes = _fetch_operational_objects(
                conn,
                "DecisionOutcome",
                limit=10,
                order_by="updated_sort DESC, as_of_sort DESC, object_uid ASC",
            )

        return {
            "latest_evaluations": latest_evaluations,
            "theses": theses,
            "pending_approvals": pending_approvals,
            "latest_daily_recommendation": latest_daily_recommendation,
            "latest_weekly_recommendation": latest_weekly_recommendation,
            "pending_actionable_recommendations": pending_actionable_recommendations,
            "pending_course_of_actions": pending_course_of_actions,
            "recent_course_of_actions": recent_course_of_actions,
            "open_course_of_action_comparisons": open_course_of_action_comparisons,
            "open_action_items": open_action_items,
            "optimizer_alerts": optimizer_alerts,
            "active_monitor_definitions": active_monitor_definitions,
            "active_mission_definitions": active_mission_definitions,
            "active_watch_triggers": active_watch_triggers,
            "recent_monitor_hits": recent_monitor_hits,
            "open_opportunity_candidates": open_opportunity_candidates,
            "recent_workflow_runs": recent_workflow_runs,
            "recent_report_runs": recent_report_runs,
            "challenged_claims": challenged_claims,
            "disconfirmed_claims": disconfirmed_claims,
            "pending_draft_decision_outcomes": pending_draft_decision_outcomes,
            "recent_finalized_decision_outcomes": recent_finalized_decision_outcomes,
        }

    def fetch_dossier_bundle(self, ticker: str) -> dict[str, Any]:
        """Fetch the ontology-backed dossier payload for one ticker without scanning all positions."""
        normalized = str(ticker or "").strip().upper()
        with self._connect() as conn:
            position = _first_row(
                _fetch_operational_objects(
                    conn,
                    "Position",
                    filters={"ticker": normalized},
                    limit=1,
                    order_by="updated_sort DESC, object_uid ASC",
                )
            )
            thesis = _fetch_dossier_thesis(conn, normalized)
            management_quality_assessment = _first_row(
                _fetch_operational_objects(
                    conn,
                    "ManagementQualityAssessment",
                    filters={"ticker": normalized},
                    limit=20,
                    order_by=(
                        "CASE WHEN status = 'active' THEN 0 ELSE 1 END, "
                        "updated_sort DESC, created_at_sort DESC, object_uid ASC"
                    ),
                )
            )
            if management_quality_assessment:
                _attach_management_quality_children(conn, management_quality_assessment)
            evaluations = _fetch_operational_objects(
                conn,
                "Evaluation",
                filters={"ticker": normalized},
                limit=52,
                order_by="evaluated_at_sort DESC, updated_sort DESC, object_uid ASC",
            )
            catalysts = _fetch_operational_objects(
                conn,
                "Catalyst",
                filters={"ticker": normalized},
                limit=100,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            kill_conditions = _fetch_operational_objects(
                conn,
                "KillCondition",
                filters={"ticker": normalized},
                limit=100,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            thesis_claims = _fetch_operational_objects(
                conn,
                "ThesisClaim",
                filters={"ticker": normalized},
                limit=100,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            workflow_runs = _fetch_operational_objects(
                conn,
                "WorkflowRun",
                filters={"ticker": normalized},
                limit=10,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            action_items = _fetch_operational_objects(
                conn,
                "ActionItem",
                filters={"ticker": normalized, "status": "open"},
                limit=100,
                order_by="created_at_sort DESC, updated_sort DESC, object_uid ASC",
            )
            watch_triggers = _fetch_operational_objects(
                conn,
                "WatchTrigger",
                filters={"ticker": normalized},
                limit=100,
                order_by="created_at_sort DESC, updated_sort DESC, object_uid ASC",
            )
            monitor_hits = _fetch_operational_objects(
                conn,
                "MonitorHit",
                filters={"ticker": normalized},
                limit=50,
                order_by="updated_sort DESC, created_at_sort DESC, object_uid ASC",
            )
            pending_approvals = _fetch_operational_objects(
                conn,
                "Approval",
                filters={"ticker": normalized, "status": "pending"},
                limit=200,
                order_by="created_at_sort DESC, updated_sort DESC, object_uid ASC",
            )
            decision_outcomes = _fetch_operational_objects(
                conn,
                "DecisionOutcome",
                filters={"ticker": normalized},
                limit=20,
                order_by="as_of_sort DESC, updated_sort DESC, object_uid ASC",
            )

        return {
            "position": position,
            "thesis_meta": thesis,
            "management_quality_assessment": management_quality_assessment,
            "evaluations": evaluations,
            "catalysts": catalysts,
            "kill_conditions": kill_conditions,
            "thesis_claims": thesis_claims,
            "workflow_runs": workflow_runs,
            "action_items": action_items,
            "watch_triggers": watch_triggers,
            "monitor_hits": monitor_hits,
            "pending_approvals": pending_approvals,
            "decision_outcomes": decision_outcomes,
        }

    def source_status_summary(self) -> tuple[dict[str, dict[str, Any]], list[str]]:
        sql = """
        SELECT
          source_name,
          status,
          quality,
          as_of,
          load_time,
          provenance_event_id
        FROM ontology_current_source_status_read_model
        ORDER BY source_name
        """
        with self._connect() as conn:
            rows = conn.execute(sql, ()).fetchall()
        status: dict[str, dict[str, Any]] = {}
        for row_raw in rows:
            row = _row_dict(row_raw)
            source_name = str(row.get("source_name") or "unknown")
            status[source_name] = {
                "status": str(row.get("status") or "ok"),
                "quality": str(row.get("quality") or "ok"),
                "as_of": _iso(row.get("as_of")),
                "load_time": _iso(row.get("load_time")),
                "provenance_event_id": row.get("provenance_event_id"),
            }
        return status, sorted(status.keys())


_OPERATIONAL_FILTER_COLUMNS = {
    "object_uid",
    "ticker",
    "status",
    "application_status",
    "approval_status",
    "outcome_status",
    "final_label_status",
    "report_type",
    "parent_uid",
    "assessment_id",
    "run_id",
}


def _fetch_operational_objects(
    conn: Any,
    object_type: str,
    *,
    filters: Mapping[str, Any] | None = None,
    limit: int = 100,
    order_by: str = "updated_sort DESC, object_uid ASC",
) -> list[dict[str, Any]]:
    where_parts = ["object_type = %s"]
    params: list[Any] = [object_type]
    for column, raw_value in (filters or {}).items():
        if column not in _OPERATIONAL_FILTER_COLUMNS:
            raise ValueError(f"Unsupported operational read-model filter: {column}")
        if raw_value is None or raw_value == "":
            continue
        value = str(raw_value)
        if column == "ticker":
            value = value.upper()
        elif column in {
            "status",
            "application_status",
            "approval_status",
            "outcome_status",
            "final_label_status",
            "report_type",
        }:
            value = value.lower()
        where_parts.append(f"{column} = %s")
        params.append(value)

    sql = f"""
    SELECT *
    FROM {OPERATIONAL_READ_MODEL_VIEW}
    WHERE {" AND ".join(where_parts)}
    ORDER BY {order_by}
    LIMIT %s
    """
    rows = conn.execute(sql, tuple([*params, max(1, min(int(limit), 1000))])).fetchall()
    return [_operational_object(row) for row in rows]


def _fetch_operational_by_uids(conn: Any, object_uids: Sequence[str]) -> dict[str, dict[str, Any]]:
    uids = _normalized_ids(object_uids)
    if not uids:
        return {}
    where, params = _in_clause("object_uid", uids)
    rows = conn.execute(
        f"""
        SELECT *
        FROM {OPERATIONAL_READ_MODEL_VIEW}
        WHERE {where}
        """,
        tuple(params),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row_raw in rows:
        row = _operational_object(row_raw)
        uid = str(row.get("object_uid") or "")
        if uid:
            out[uid] = row
    return out


def _fetch_operational_in(
    conn: Any,
    object_type: str,
    column: str,
    values: Sequence[str],
    *,
    limit: int = 1000,
    order_by: str = "updated_sort DESC, object_uid ASC",
) -> list[dict[str, Any]]:
    if column not in _OPERATIONAL_FILTER_COLUMNS:
        raise ValueError(f"Unsupported operational read-model IN filter: {column}")
    normalized = _normalized_ids(values)
    if not normalized:
        return []
    where, params = _in_clause(column, normalized)
    rows = conn.execute(
        f"""
        SELECT *
        FROM {OPERATIONAL_READ_MODEL_VIEW}
        WHERE object_type = %s
          AND {where}
        ORDER BY {order_by}
        LIMIT %s
        """,
        tuple([object_type, *params, max(1, min(int(limit), 5000))]),
    ).fetchall()
    return [_operational_object(row) for row in rows]


def _fetch_dossier_thesis(conn: Any, ticker: str) -> dict[str, Any] | None:
    object_uid = f"thesis:{ticker}"
    rows = conn.execute(
        f"""
        SELECT *
        FROM {OPERATIONAL_READ_MODEL_VIEW}
        WHERE object_type = 'Thesis'
          AND (object_uid = %s OR ticker = %s)
        ORDER BY CASE WHEN object_uid = %s THEN 0 ELSE 1 END,
                 updated_sort DESC,
                 object_uid ASC
        LIMIT 1
        """,
        (object_uid, ticker, object_uid),
    ).fetchall()
    return _first_row([_operational_object(row) for row in rows])


def _attach_optimization_alert_context(conn: Any, alerts: list[dict[str, Any]]) -> None:
    snapshot_ids: list[str] = []
    alert_ids: list[str] = []
    for alert in alerts:
        props = _row_dict(alert.get("properties_json"))
        current_snapshot_id = str(props.get("current_snapshot_id") or "").strip()
        previous_snapshot_id = str(props.get("previous_snapshot_id") or "").strip()
        if current_snapshot_id:
            snapshot_ids.append(current_snapshot_id)
        if previous_snapshot_id:
            snapshot_ids.append(previous_snapshot_id)
        alert_uid = str(alert.get("object_uid") or props.get("id") or "").strip()
        if alert_uid:
            alert_ids.append(alert_uid)

    snapshots = _fetch_operational_by_uids(conn, snapshot_ids)
    source_freshness = _source_freshness_by_parent(conn, alert_ids)
    for alert in alerts:
        props = _row_dict(alert.get("properties_json"))
        current_snapshot_id = str(props.get("current_snapshot_id") or "").strip()
        previous_snapshot_id = str(props.get("previous_snapshot_id") or "").strip()
        if current_snapshot_id and current_snapshot_id in snapshots:
            alert["current_snapshot"] = snapshots[current_snapshot_id]
        if previous_snapshot_id and previous_snapshot_id in snapshots:
            alert["previous_snapshot"] = snapshots[previous_snapshot_id]
        alert_uid = str(alert.get("object_uid") or props.get("id") or "").strip()
        if alert_uid and alert_uid in source_freshness:
            alert["source_freshness"] = source_freshness[alert_uid]


def _source_freshness_by_parent(conn: Any, parent_uids: Sequence[str]) -> dict[str, dict[str, dict[str, Any]]]:
    rows = _fetch_operational_in(
        conn,
        "SourceFreshness",
        "parent_uid",
        parent_uids,
        limit=max(100, len(parent_uids) * 100),
        order_by="parent_uid ASC, object_uid ASC",
    )
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        props = _row_dict(row.get("properties_json"))
        parent_uid = str(props.get("parent_uid") or row.get("parent_uid") or "").strip()
        source_name = str(props.get("source_name") or "").strip()
        if not parent_uid or not source_name:
            continue
        grouped.setdefault(parent_uid, {})[source_name] = {
            key: props.get(key)
            for key in ("status", "checked_at", "as_of", "freshness_category", "error", "metadata")
            if props.get(key) is not None
        }
    return grouped


def _attach_management_quality_children(conn: Any, assessment: dict[str, Any]) -> None:
    assessment_uid = str(assessment.get("object_uid") or "").strip()
    if not assessment_uid:
        return
    child_specs = (
        ("ManagementQualityScorecardRow", "scorecard"),
        ("ManagementQualityAccomplishment", "accomplishments"),
        ("ManagementQualitySetback", "setbacks"),
    )
    for object_type, key in child_specs:
        children = _fetch_operational_objects(
            conn,
            object_type,
            filters={"assessment_id": assessment_uid},
            limit=200,
            order_by="updated_sort ASC, object_uid ASC",
        )
        children.sort(key=lambda item: int(_row_dict(item.get("properties_json")).get("ordinal") or 0))
        assessment[key] = children


def _latest_by_ticker(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        props = _row_dict(row.get("properties_json"))
        ticker = str(props.get("ticker") or row.get("ticker") or "").strip().upper()
        if ticker and ticker not in latest:
            latest[ticker] = row
    return list(latest.values())


def _first_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _operational_object(row_raw: Any) -> dict[str, Any]:
    row = _row_dict(row_raw)
    props = _row_dict(row.get("properties_json"))
    payload = {
        "version_id": row.get("version_id"),
        "object_uid": row.get("object_uid"),
        "object_type": row.get("object_type"),
        "business_key": row.get("business_key"),
        "properties_json": props,
        "properties": props,
        "schema_name": row.get("schema_name"),
        "schema_version": row.get("schema_version"),
        "source_record_id": row.get("source_record_id"),
        "valid_from": row.get("valid_from"),
        "valid_to": row.get("valid_to"),
        "tx_from": row.get("tx_from"),
        "tx_to": row.get("tx_to"),
        "actor_id": row.get("actor_id"),
        "input_hash": row.get("input_hash"),
        "supersedes_version_id": row.get("supersedes_version_id"),
        "temporal_confidence": row.get("temporal_confidence"),
        "ticker": row.get("ticker"),
        "status": row.get("status"),
        "application_status": row.get("application_status"),
        "approval_status": row.get("approval_status"),
        "outcome_status": row.get("outcome_status"),
        "final_label_status": row.get("final_label_status"),
        "report_type": row.get("report_type"),
        "parent_uid": row.get("parent_uid"),
        "assessment_id": row.get("assessment_id"),
        "run_id": row.get("run_id"),
        "current_snapshot_id": row.get("current_snapshot_id"),
        "previous_snapshot_id": row.get("previous_snapshot_id"),
    }
    temporal = {
        "object_uid": payload.get("object_uid"),
        "version_id": str(payload.get("version_id")) if payload.get("version_id") is not None else None,
        "valid_from": _iso(payload.get("valid_from")),
        "valid_to": _iso(payload.get("valid_to")),
        "tx_from": _iso(payload.get("tx_from")),
        "tx_to": _iso(payload.get("tx_to")),
        "temporal_confidence": payload.get("temporal_confidence"),
    }
    payload["_meta"] = {"temporal": {key: value for key, value in temporal.items() if value is not None}}
    return payload


def _position_source_sql(
    *,
    as_of: str | None,
    tx_as_of: str | None,
    include_history: bool,
) -> tuple[str, list[Any]]:
    if not as_of and not tx_as_of and not include_history:
        return "SELECT * FROM ontology_current_position_risk_read_model", []

    object_where, object_params = _temporal_where(as_of=as_of, tx_as_of=tx_as_of, include_history=include_history)
    relation_where, relation_params = _temporal_where(as_of=as_of, tx_as_of=tx_as_of, include_history=include_history)
    sql = f"""
    WITH objs AS (
      SELECT *
      FROM ontology_object_versions
      WHERE {object_where}
    ),
    rels AS (
      SELECT *
      FROM ontology_relation_versions
      WHERE {relation_where}
    )
    SELECT
      p.object_uid AS position_id,
      p.business_key AS position_business_key,
      COALESCE(p.properties_json->>'ticker', replace(p.object_uid, 'position:', '')) AS ticker,
      p.properties_json->>'asset' AS asset,
      p.properties_json->>'direction' AS direction,
      NULLIF(p.properties_json->>'risk_score', '')::double precision AS risk_score_value,
      p.properties_json AS position_props,
      p.schema_name AS position_schema_name,
      p.schema_version AS position_schema_version,
      p.tx_from AS position_updated_at,
      p.version_id AS position_version_id,
      p.valid_from AS position_valid_from,
      p.valid_to AS position_valid_to,
      p.tx_from AS position_tx_from,
      p.tx_to AS position_tx_to,
      p.temporal_confidence AS position_temporal_confidence,
      COALESCE(p.properties_json->>'ticker', replace(p.object_uid, 'position:', '')) AS position_label,
      a.object_uid AS asset_id,
      a.properties_json AS asset_props,
      a.schema_name AS asset_schema_name,
      a.schema_version AS asset_schema_version,
      a.tx_from AS asset_updated_at,
      a.properties_json->>'ticker' AS asset_label,
      s.object_uid AS sector_id,
      s.properties_json AS sector_props,
      s.schema_name AS sector_schema_name,
      s.schema_version AS sector_schema_version,
      s.tx_from AS sector_updated_at,
      s.properties_json->>'name' AS sector_label,
      COALESCE(s.properties_json->>'name', 'Unknown Equity') AS sector,
      pa.properties_json AS position_asset_edge_props,
      pa.relation_schema_name AS position_asset_edge_schema_name,
      pa.relation_schema_version AS position_asset_edge_schema_version,
      pa.relation_schema_name AS position_asset_edge_relation_schema_name,
      pa.relation_schema_version AS position_asset_edge_relation_schema_version,
      pa.tx_from AS position_asset_edge_updated_at,
      ase.properties_json AS asset_sector_edge_props,
      ase.relation_schema_name AS asset_sector_edge_schema_name,
      ase.relation_schema_version AS asset_sector_edge_schema_version,
      ase.relation_schema_name AS asset_sector_edge_relation_schema_name,
      ase.relation_schema_version AS asset_sector_edge_relation_schema_version,
      ase.tx_from AS asset_sector_edge_updated_at
    FROM objs p
    LEFT JOIN rels pa
      ON pa.source_object_uid = p.object_uid
     AND pa.relation_type = 'references_asset'
    LEFT JOIN objs a
      ON a.object_uid = pa.target_object_uid
     AND a.object_type = 'Asset'
    LEFT JOIN rels ase
      ON ase.source_object_uid = a.object_uid
     AND ase.relation_type = 'belongs_to_sector'
    LEFT JOIN objs s
      ON s.object_uid = ase.target_object_uid
     AND s.object_type = 'Sector'
    WHERE p.object_type = 'Position'
    """
    return sql, [*object_params, *relation_params]


def _evidence_source_sql(
    *,
    as_of: str | None,
    tx_as_of: str | None,
    include_history: bool,
) -> tuple[str, list[Any]]:
    if not as_of and not tx_as_of and not include_history:
        return "SELECT * FROM ontology_current_position_signal_evidence_read_model", []
    object_where, object_params = _temporal_where(as_of=as_of, tx_as_of=tx_as_of, include_history=include_history)
    relation_where, relation_params = _temporal_where(as_of=as_of, tx_as_of=tx_as_of, include_history=include_history)
    sql = f"""
    WITH objs AS (
      SELECT *
      FROM ontology_object_versions
      WHERE {object_where}
    ),
    rels AS (
      SELECT *
      FROM ontology_relation_versions
      WHERE {relation_where}
    )
    SELECT
      ps.source_object_uid AS position_id,
      s.object_uid AS signal_id,
      COALESCE(s.properties_json->>'name', s.business_key, s.object_uid) AS signal_label,
      s.properties_json AS signal_props,
      s.schema_name AS signal_schema_name,
      s.schema_version AS signal_schema_version,
      s.tx_from AS signal_updated_at,
      ps.properties_json AS edge_props,
      ps.relation_schema_name AS edge_schema_name,
      ps.relation_schema_version AS edge_schema_version,
      ps.relation_schema_name AS relation_schema_name,
      ps.relation_schema_version AS relation_schema_version,
      ps.tx_from AS edge_updated_at
    FROM rels ps
    JOIN objs s
      ON s.object_uid = ps.target_object_uid
     AND s.object_type = 'Signal'
    WHERE ps.relation_type = 'exposed_to_signal'
    """
    return sql, [*object_params, *relation_params]


def _thesis_source_sql(
    *,
    as_of: str | None,
    tx_as_of: str | None,
    include_history: bool,
) -> tuple[str, list[Any]]:
    if not as_of and not tx_as_of and not include_history:
        return "SELECT * FROM ontology_current_position_thesis_context_read_model", []
    object_where, object_params = _temporal_where(as_of=as_of, tx_as_of=tx_as_of, include_history=include_history)
    relation_where, relation_params = _temporal_where(as_of=as_of, tx_as_of=tx_as_of, include_history=include_history)
    sql = f"""
    WITH objs AS (
      SELECT *
      FROM ontology_object_versions
      WHERE {object_where}
    ),
    rels AS (
      SELECT *
      FROM ontology_relation_versions
      WHERE {relation_where}
    ),
    theses AS (
      SELECT
        ht.source_object_uid AS position_id,
        ht.target_object_uid AS thesis_id,
        ht.properties_json AS source_edge_props,
        ht.relation_schema_name AS source_edge_schema_name,
        ht.relation_schema_version AS source_edge_schema_version,
        ht.tx_from AS source_edge_updated_at,
        t.object_uid AS target_id,
        t.object_type AS target_type,
        t.properties_json AS target_props,
        t.schema_name AS target_schema_name,
        t.schema_version AS target_schema_version,
        t.tx_from AS target_updated_at,
        COALESCE(t.properties_json->>'ticker', t.business_key, t.object_uid) AS target_label,
        'thesis'::text AS context_type
      FROM rels ht
      JOIN objs t ON t.object_uid = ht.target_object_uid AND t.object_type = 'Thesis'
      WHERE ht.relation_type = 'has_thesis'
    )
    SELECT * FROM theses
    UNION ALL
    SELECT
      th.position_id,
      th.thesis_id,
      eb.properties_json,
      eb.relation_schema_name,
      eb.relation_schema_version,
      eb.tx_from,
      e.object_uid,
      e.object_type,
      e.properties_json,
      e.schema_name,
      e.schema_version,
      e.tx_from,
      COALESCE(e.properties_json->>'evaluated_at', e.business_key, e.object_uid),
      'evaluation'::text
    FROM theses th
    JOIN rels eb ON eb.source_object_uid = th.thesis_id AND eb.relation_type = 'evaluated_by'
    JOIN objs e ON e.object_uid = eb.target_object_uid AND e.object_type = 'Evaluation'
    UNION ALL
    SELECT
      th.position_id,
      th.thesis_id,
      hc.properties_json,
      hc.relation_schema_name,
      hc.relation_schema_version,
      hc.tx_from,
      c.object_uid,
      c.object_type,
      c.properties_json,
      c.schema_name,
      c.schema_version,
      c.tx_from,
      COALESCE(c.properties_json->>'name', c.business_key, c.object_uid),
      'catalyst'::text
    FROM theses th
    JOIN rels hc ON hc.source_object_uid = th.thesis_id AND hc.relation_type = 'has_catalyst'
    JOIN objs c ON c.object_uid = hc.target_object_uid AND c.object_type = 'Catalyst'
    """
    return sql, [*object_params, *relation_params]


def _temporal_where(
    *,
    as_of: str | None,
    tx_as_of: str | None,
    include_history: bool,
) -> tuple[str, list[Any]]:
    if include_history:
        return "TRUE", []
    valid_time = as_of or datetime.now(UTC).isoformat()
    parts = ["valid_from <= %s", "(valid_to IS NULL OR valid_to > %s)"]
    params: list[Any] = [valid_time, valid_time]
    if tx_as_of:
        parts.extend(["tx_from <= %s", "(tx_to IS NULL OR tx_to > %s)"])
        params.extend([tx_as_of, tx_as_of])
    else:
        parts.append("tx_to IS NULL")
    return " AND ".join(parts), params


def _position_filter_sql(filters: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
    filters = filters or {}
    parts = ["TRUE"]
    params: list[Any] = []
    tickers = _clean_list(filters.get("tickers"), upper=True)
    if tickers:
        clause, values = _in_clause("UPPER(ticker)", tickers)
        parts.append(clause)
        params.extend(values)
    sectors = _clean_list(filters.get("sectors"), lower=True)
    if sectors:
        clause, values = _in_clause("LOWER(sector)", sectors)
        parts.append(clause)
        params.extend(values)
    assets = _clean_list(filters.get("assets"), lower=True)
    if assets:
        clause, values = _in_clause("LOWER(asset)", assets)
        parts.append(clause)
        params.extend(values)
    min_risk = _to_float(filters.get("min_risk_score"))
    if min_risk is not None:
        parts.append("COALESCE(risk_score_value, 0.0) >= %s")
        params.append(min_risk)
    return " AND ".join(parts), params


def _position_row(row_raw: Any) -> dict[str, Any]:
    row = _row_dict(row_raw)
    return {
        "position_id": row.get("position_id"),
        "position_label": row.get("position_label") or row.get("ticker") or row.get("position_id"),
        "position_props": row.get("position_props") or {},
        "position_schema_name": row.get("position_schema_name"),
        "position_schema_version": int(row.get("position_schema_version") or 0),
        "position_updated_at": _iso(row.get("position_updated_at")),
        "position_version_id": _str_or_none(row.get("position_version_id")),
        "position_valid_from": _iso(row.get("position_valid_from")),
        "position_valid_to": _iso(row.get("position_valid_to")),
        "position_tx_from": _iso(row.get("position_tx_from")),
        "position_tx_to": _iso(row.get("position_tx_to")),
        "position_temporal_confidence": row.get("position_temporal_confidence"),
        "asset_id": row.get("asset_id"),
        "asset_label": row.get("asset_label") or row.get("asset_id"),
        "asset_props": row.get("asset_props") or {},
        "asset_schema_name": row.get("asset_schema_name"),
        "asset_schema_version": int(row.get("asset_schema_version") or 0) if row.get("asset_id") else None,
        "asset_updated_at": _iso(row.get("asset_updated_at")),
        "sector_id": row.get("sector_id"),
        "sector_label": row.get("sector_label") or row.get("sector_id"),
        "sector_props": row.get("sector_props") or {},
        "sector_schema_name": row.get("sector_schema_name"),
        "sector_schema_version": int(row.get("sector_schema_version") or 0) if row.get("sector_id") else None,
        "sector_updated_at": _iso(row.get("sector_updated_at")),
        "position_asset_edge_props": row.get("position_asset_edge_props") or {},
        "position_asset_edge_schema_name": row.get("position_asset_edge_schema_name"),
        "position_asset_edge_schema_version": (
            int(row.get("position_asset_edge_schema_version") or 0)
            if row.get("position_asset_edge_schema_name")
            else None
        ),
        "position_asset_edge_relation_schema_name": row.get("position_asset_edge_relation_schema_name"),
        "position_asset_edge_relation_schema_version": int(row.get("position_asset_edge_relation_schema_version") or 0),
        "position_asset_edge_updated_at": _iso(row.get("position_asset_edge_updated_at")),
        "asset_sector_edge_props": row.get("asset_sector_edge_props") or {},
        "asset_sector_edge_schema_name": row.get("asset_sector_edge_schema_name"),
        "asset_sector_edge_schema_version": (
            int(row.get("asset_sector_edge_schema_version") or 0) if row.get("asset_sector_edge_schema_name") else None
        ),
        "asset_sector_edge_relation_schema_name": row.get("asset_sector_edge_relation_schema_name"),
        "asset_sector_edge_relation_schema_version": int(row.get("asset_sector_edge_relation_schema_version") or 0),
        "asset_sector_edge_updated_at": _iso(row.get("asset_sector_edge_updated_at")),
    }


def _context_node(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("target_id"),
        "type": row.get("target_type"),
        "label": row.get("target_label") or row.get("target_id"),
        "properties": row.get("target_props") or {},
        "schema_name": row.get("target_schema_name"),
        "schema_version": int(row.get("target_schema_version") or 0),
        "updated_at": _iso(row.get("target_updated_at")),
    }


def _context_edge(row: Mapping[str, Any]) -> dict[str, Any]:
    relation_type = "has_thesis"
    if row.get("context_type") == "evaluation":
        relation_type = "evaluated_by"
    elif row.get("context_type") == "catalyst":
        relation_type = "has_catalyst"
    return {
        "source_id": row.get("position_id") if row.get("context_type") == "thesis" else row.get("thesis_id"),
        "target_id": row.get("target_id"),
        "relation_type": relation_type,
        "properties": row.get("source_edge_props") or {},
        "schema_name": row.get("source_edge_schema_name"),
        "schema_version": int(row.get("source_edge_schema_version") or 0),
        "relation_schema_name": row.get("source_edge_schema_name"),
        "relation_schema_version": int(row.get("source_edge_schema_version") or 0),
        "updated_at": _iso(row.get("source_edge_updated_at")),
    }


def _in_clause(column: str, values: Sequence[str]) -> tuple[str, list[Any]]:
    placeholders = ", ".join("%s" for _ in values)
    return f"{column} IN ({placeholders})", list(values)


def _normalized_ids(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in out:
            out.append(item)
    return out


def _clean_list(value: Any, *, upper: bool = False, lower: bool = False) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if upper:
            text = text.upper()
        if lower:
            text = text.lower()
        out.append(text)
    return out


def _row_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return {}


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    data = _row_dict(row)
    return data.get(key, default)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None
