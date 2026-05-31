"""Ontology-backed runtime read helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from api.postgres import use_postgres_state
from ontology.object_service import OntologyObjectService

logger = logging.getLogger(__name__)
_DEFAULT_OBJECT_SERVICE: ContextVar[Any | None] = ContextVar("ontology_runtime_object_service", default=None)


class _EmptyObjectService:
    def get_object(self, object_uid: str, **kwargs: Any) -> None:
        return None

    def query_objects(self, object_type: str | None = None, filters: dict[str, Any] | None = None, **kwargs: Any):
        return []

    def query_relations(self, relation_type: str | None = None, **kwargs: Any):
        return []


@contextmanager
def runtime_object_service(object_service: Any) -> Iterator[None]:
    token = _DEFAULT_OBJECT_SERVICE.set(object_service)
    try:
        yield
    finally:
        _DEFAULT_OBJECT_SERVICE.reset(token)


def _default_object_service() -> Any:
    if not use_postgres_state():
        return _EmptyObjectService()
    return OntologyObjectService()


def _cached_report_positions(*, include_hedges: bool = False) -> list[dict[str, Any]] | None:
    try:
        from auto_report.report_state import api_only_mode, load_cached_positions
    except Exception:
        return None
    if not api_only_mode():
        return None
    rows = load_cached_positions(include_hedges=include_hedges)
    if rows is None:
        raise RuntimeError("AUTO_REPORT_API_ONLY requires cached portfolio state from auto_report.fetch_state.")
    return rows


def get_positions(*, include_hedges: bool = False) -> list[dict[str, Any]]:
    """Return current ontology positions."""
    cached_positions = _cached_report_positions(include_hedges=include_hedges)
    if cached_positions is not None:
        return cached_positions
    return OntologyRuntimeReadService().positions(include_hedges=include_hedges)


def object_props(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    props = dict(row.get("properties") or row.get("properties_json") or {})
    object_uid = str(row.get("object_uid") or props.get("id") or "")
    if object_uid:
        props["id"] = object_uid
        props["object_uid"] = object_uid
    meta = row.get("_meta")
    if isinstance(meta, dict):
        props["_meta"] = meta
    return props


def get_positions_df(*, include_hedges: bool = False, fallback_to_csv: bool = False):
    """Return current ontology positions as a DataFrame."""
    cached_positions = _cached_report_positions(include_hedges=include_hedges)
    if cached_positions is not None:
        import pandas as pd

        return pd.DataFrame(cached_positions)
    return OntologyRuntimeReadService().positions_df(include_hedges=include_hedges)


def get_hedge_positions() -> list[dict[str, Any]]:
    cached_positions = _cached_report_positions(include_hedges=True)
    if cached_positions is not None:
        return [row for row in cached_positions if str(row.get("role") or "").lower() == "hedge"]
    rows = OntologyRuntimeReadService().positions(include_hedges=True)
    return [row for row in rows if str(row.get("role") or "").lower() == "hedge"]


class OntologyRuntimeReadService:
    def __init__(self, object_service: OntologyObjectService | None = None, read_model_repository: Any | None = None):
        self.objects = object_service or _DEFAULT_OBJECT_SERVICE.get() or _default_object_service()
        self.read_model_repo = read_model_repository

    def get(self, object_uid: str) -> dict[str, Any] | None:
        row = self.objects.get_object(object_uid)
        if not row:
            return None
        return self._project_object(str(row.get("object_type") or ""), object_props(row))

    def list_objects(
        self,
        object_type: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return [
            self._project_object(object_type, object_props(row))
            for row in self.objects.query_objects(object_type, filters=_clean_filters(filters), limit=limit)
        ]

    def workspace_bundle(self) -> dict[str, Any]:
        repo = self._read_model_repository()
        bundle = repo.fetch_workspace_bundle()
        recent_workflow_runs = self._merge_fresh_workflow_runs(
            self._project_read_model_rows(bundle.get("recent_workflow_runs", [])),
            ticker=None,
            limit=3,
        )
        return {
            "latest_evaluations": self._project_read_model_rows(bundle.get("latest_evaluations", [])),
            "theses": self._project_read_model_rows(bundle.get("theses", [])),
            "pending_approvals": self._project_read_model_rows(bundle.get("pending_approvals", [])),
            "latest_daily_recommendation": self._project_read_model_row(bundle.get("latest_daily_recommendation")),
            "latest_weekly_recommendation": self._project_read_model_row(bundle.get("latest_weekly_recommendation")),
            "pending_actionable_recommendations": self._project_read_model_rows(
                bundle.get("pending_actionable_recommendations", [])
            ),
            "open_action_items": self._project_read_model_rows(bundle.get("open_action_items", [])),
            "optimizer_alerts": self._project_read_model_rows(bundle.get("optimizer_alerts", [])),
            "active_watch_triggers": self._project_read_model_rows(bundle.get("active_watch_triggers", [])),
            "recent_workflow_runs": recent_workflow_runs,
            "recent_report_runs": self._project_read_model_rows(bundle.get("recent_report_runs", [])),
            "challenged_claims": self._project_read_model_rows(bundle.get("challenged_claims", [])),
            "disconfirmed_claims": self._project_read_model_rows(bundle.get("disconfirmed_claims", [])),
        }

    def dossier_bundle(self, ticker: str) -> dict[str, Any]:
        repo = self._read_model_repository()
        bundle = repo.fetch_dossier_bundle(ticker)
        workflow_runs = self._merge_fresh_workflow_runs(
            self._project_read_model_rows(bundle.get("workflow_runs", [])),
            ticker=ticker,
            limit=10,
        )
        return {
            "position": self._project_read_model_row(bundle.get("position")),
            "thesis_meta": self._project_read_model_row(bundle.get("thesis_meta")),
            "management_quality_assessment": self._project_read_model_row(bundle.get("management_quality_assessment")),
            "evaluations": self._project_read_model_rows(bundle.get("evaluations", [])),
            "catalysts": self._project_read_model_rows(bundle.get("catalysts", [])),
            "kill_conditions": self._project_read_model_rows(bundle.get("kill_conditions", [])),
            "thesis_claims": self._project_read_model_rows(bundle.get("thesis_claims", [])),
            "workflow_runs": workflow_runs,
            "action_items": self._project_read_model_rows(bundle.get("action_items", [])),
            "watch_triggers": self._project_read_model_rows(bundle.get("watch_triggers", [])),
            "pending_approvals": self._project_read_model_rows(bundle.get("pending_approvals", [])),
        }

    def management_quality_assessment(self, ticker: str) -> dict[str, Any] | None:
        normalized = _ticker(ticker)
        rows = self.list_objects("ManagementQualityAssessment", filters={"ticker": normalized}, limit=20)
        active = [row for row in rows if str(row.get("status") or "active").lower() == "active"]
        candidates = active or rows
        candidates.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
        return candidates[0] if candidates else None

    def positions(self, *, include_hedges: bool = False, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.list_objects("Position", limit=limit)
        if include_hedges:
            rows.extend(self.list_objects("HedgePosition", limit=limit))
        return sorted(rows, key=lambda row: str(row.get("ticker") or row.get("id") or ""))

    def positions_df(self, *, include_hedges: bool = False):
        import pandas as pd

        rows = self.positions(include_hedges=include_hedges)
        return pd.DataFrame(rows)

    def thesis(self, ticker: str) -> dict[str, Any] | None:
        normalized = _ticker(ticker)
        return self.get(f"thesis:{normalized}") or _first(self.list_objects("Thesis", filters={"ticker": normalized}))

    def theses(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        return self.list_objects("Thesis", limit=limit)

    def evaluations(self, ticker: str | None = None, *, limit: int = 1000) -> list[dict[str, Any]]:
        filters = {"ticker": _ticker(ticker)} if ticker else None
        rows = self.list_objects("Evaluation", filters=filters, limit=limit)
        return sorted(rows, key=lambda row: str(row.get("evaluated_at") or ""), reverse=True)

    def latest_evaluations(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.evaluations(limit=limit):
            ticker = _ticker(row.get("ticker"))
            if ticker and ticker not in latest:
                latest[ticker] = row
        return list(latest.values())

    def catalysts(
        self, ticker: str | None = None, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.list_objects("Catalyst", filters=_ticker_status_filter(ticker, status), limit=limit)

    def kill_conditions(
        self,
        ticker: str | None = None,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.list_objects("KillCondition", filters=_ticker_status_filter(ticker, status), limit=limit)

    def thesis_claims(
        self,
        ticker: str | None = None,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.list_objects("ThesisClaim", filters=_ticker_status_filter(ticker, status), limit=limit)

    def evidence_ledger(self, ticker: str) -> dict[str, Any]:
        from ontology.evidence_ledger import build_ticker_evidence_ledger

        return build_ticker_evidence_ledger(self, ticker)

    def action_items(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.list_objects("ActionItem", filters=_ticker_status_filter(ticker, status), limit=limit)

    def watch_triggers(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.list_objects("WatchTrigger", filters=_ticker_status_filter(ticker, status), limit=limit)

    def monitor_hits(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        filters = _clean_filters(
            {
                "ticker": _ticker(ticker) if ticker else None,
                "status": status,
                "entity_type": entity_type,
                "entity_id": entity_id,
            }
        )
        rows = self.list_objects("MonitorHit", filters=filters, limit=limit)
        return sorted(rows, key=lambda row: str(row.get("detected_at") or ""), reverse=True)

    def approvals(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        application_status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        filters = _clean_filters(
            {
                "ticker": _ticker(ticker) if ticker else None,
                "status": status,
                "application_status": application_status,
            }
        )
        return self.list_objects("Approval", filters=filters, limit=limit)

    def recommendations(
        self,
        *,
        report_type: str | None = None,
        status: str | None = None,
        ticker: str | None = None,
        approval_status: str | None = None,
        outcome_status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self.list_objects(
            "Recommendation",
            filters=_clean_filters(
                {
                    "report_type": report_type,
                    "status": status,
                    "ticker": _ticker(ticker) if ticker else None,
                    "approval_status": approval_status,
                    "outcome_status": outcome_status,
                }
            ),
            limit=limit,
        )
        return sorted(rows, key=lambda row: str(row.get("as_of") or ""), reverse=True)

    def decision_outcomes(
        self,
        *,
        ticker: str | None = None,
        outcome_status: str | None = None,
        final_label_status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self.list_objects(
            "DecisionOutcome",
            filters=_clean_filters(
                {
                    "ticker": _ticker(ticker) if ticker else None,
                    "outcome_status": outcome_status,
                    "final_label_status": final_label_status,
                }
            ),
            limit=limit,
        )
        return sorted(
            rows,
            key=lambda row: str(row.get("finalized_at") or row.get("as_of") or row.get("updated_at") or ""),
            reverse=True,
        )

    def latest_recommendation(self, report_type: str) -> dict[str, Any] | None:
        return _first(self.recommendations(report_type=report_type, limit=1))

    def policy_gate_results(
        self,
        *,
        decision: str | None = None,
        action_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self.list_objects("PolicyGateResult", filters=_clean_filters({"decision": decision}), limit=limit)
        if action_id:
            rows = [row for row in rows if action_id in str(row.get("gate_result_id") or row.get("id") or "")]
        if target_type:
            rows = [row for row in rows if target_type in str(row.get("gate_result_id") or row.get("id") or "")]
        if target_id:
            rows = [row for row in rows if target_id in str(row.get("gate_result_id") or row.get("id") or "")]
        return rows

    def workflow_runs(self, *, ticker: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self._fresh_workflow_runs_from_objects(ticker=ticker, limit=limit)

    def report_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.list_objects("ReportRun", limit=limit)
        return sorted(rows, key=lambda row: str(row.get("as_of") or row.get("synced_at") or ""), reverse=True)

    def _workspace_bundle_from_objects(self) -> dict[str, Any]:
        return {
            "latest_evaluations": self.latest_evaluations(),
            "theses": self.theses(),
            "pending_approvals": self.approvals(status="pending"),
            "latest_daily_recommendation": self.latest_recommendation("daily"),
            "latest_weekly_recommendation": self.latest_recommendation("weekly"),
            "pending_actionable_recommendations": self.recommendations(approval_status="pending", limit=5),
            "open_action_items": self.action_items(status="open"),
            "optimizer_alerts": self.list_objects("OptimizationAlert", filters={"status": "open"}, limit=5),
            "active_watch_triggers": self.watch_triggers(status="active"),
            "recent_monitor_hits": self.monitor_hits(status="open", limit=20),
            "recent_workflow_runs": self.workflow_runs(limit=3),
            "recent_report_runs": self.report_runs(limit=5),
            "challenged_claims": self.thesis_claims(status="challenged", limit=5),
            "disconfirmed_claims": self.thesis_claims(status="disconfirmed", limit=5),
            "pending_draft_decision_outcomes": self.decision_outcomes(
                outcome_status="evaluated",
                final_label_status="draft",
                limit=10,
            ),
            "recent_finalized_decision_outcomes": [
                row
                for row in self.decision_outcomes(limit=20)
                if str(row.get("final_label_status") or "draft") != "draft"
            ][:10],
        }

    def _dossier_bundle_from_objects(self, ticker: str) -> dict[str, Any]:
        normalized = _ticker(ticker)
        position = None
        for pos in self.positions():
            if _ticker(pos.get("ticker")) == normalized:
                position = pos
                break
        return {
            "position": position,
            "thesis_meta": self.thesis(normalized),
            "management_quality_assessment": self.management_quality_assessment(normalized),
            "evaluations": self.evaluations(normalized, limit=52),
            "catalysts": self.catalysts(normalized),
            "kill_conditions": self.kill_conditions(normalized),
            "thesis_claims": self.thesis_claims(ticker=normalized),
            "workflow_runs": self.workflow_runs(ticker=normalized, limit=10),
            "action_items": self.action_items(ticker=normalized, status="open"),
            "watch_triggers": self.watch_triggers(ticker=normalized),
            "monitor_hits": self.monitor_hits(ticker=normalized, limit=50),
            "pending_approvals": self.approvals(ticker=normalized, status="pending"),
            "decision_outcomes": self.decision_outcomes(ticker=normalized, limit=20),
        }

    def _merge_fresh_workflow_runs(
        self,
        read_model_runs: list[dict[str, Any]],
        *,
        ticker: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            direct_runs = self._fresh_workflow_runs_from_objects(ticker=ticker, limit=max(limit, 20))
        except Exception:
            logger.exception("failed to merge fresh workflow runs into operational bundle")
            direct_runs = []

        by_id: dict[str, dict[str, Any]] = {}
        for run in [*direct_runs, *read_model_runs]:
            key = _workflow_run_identity(run)
            if key and key not in by_id:
                by_id[key] = run
        return _sort_workflow_runs(list(by_id.values()))[:limit]

    def _fresh_workflow_runs_from_objects(self, *, ticker: str | None, limit: int) -> list[dict[str, Any]]:
        normalized = _ticker(ticker) if ticker else None
        filters = {"ticker": normalized} if normalized else None
        fetch_limit = 500
        rows = self.list_objects("WorkflowRun", filters=filters, limit=fetch_limit)
        return _sort_workflow_runs(rows)[:limit]

    def _read_model_repository(self):
        if self.read_model_repo is not None:
            return self.read_model_repo
        from ontology.read_model import TemporalReadModelRepository

        self.read_model_repo = TemporalReadModelRepository()
        return self.read_model_repo

    def _project_read_model_rows(self, rows: Any) -> list[dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        return [row for row in (self._project_read_model_row(item) for item in rows) if row is not None]

    def _project_read_model_row(self, row: Any) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        object_type = str(row.get("object_type") or "")
        props = object_props(row)
        for key in ("current_snapshot", "previous_snapshot"):
            nested = row.get(key)
            if isinstance(nested, dict):
                props[key] = object_props(nested)
        source_freshness = row.get("source_freshness")
        if isinstance(source_freshness, dict):
            props["source_freshness"] = source_freshness
        for key in ("scorecard", "accomplishments", "setbacks"):
            children = row.get(key)
            if isinstance(children, list):
                props[key] = [object_props(child) for child in children if isinstance(child, dict)]
        return self._project_object(object_type, props)

    def _raw_object(self, object_uid: str) -> dict[str, Any] | None:
        row = self.objects.get_object(object_uid)
        return object_props(row) if row else None

    def _raw_objects(
        self,
        object_type: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return [
            object_props(row)
            for row in self.objects.query_objects(object_type, filters=_clean_filters(filters), limit=limit)
        ]

    def _project_object(self, object_type: str, row: dict[str, Any]) -> dict[str, Any]:
        if object_type == "IdeaEvaluation":
            return self._project_idea_evaluation(row)
        if object_type == "IdeaComparisonRun":
            return self._project_idea_comparison_run(row)
        if object_type == "OptimizationRun":
            return self._project_optimization_run(row)
        if object_type == "OptimizationAlert":
            return self._project_optimization_alert(row)
        if object_type == "ManagementQualityAssessment":
            return self._project_management_quality_assessment(row)
        return row

    def _project_idea_evaluation(self, row: dict[str, Any]) -> dict[str, Any]:
        uid = str(row.get("id") or row.get("object_uid") or "")
        factor_rows = self._raw_objects("FactorScore", filters={"parent_uid": uid}, limit=100) if uid else []
        if factor_rows:
            factors: dict[str, dict[str, Any]] = {}
            for factor in sorted(factor_rows, key=lambda item: str(item.get("factor_name") or item.get("id") or "")):
                name = str(factor.get("factor_name") or "").strip()
                if not name:
                    continue
                factors[name] = {
                    key: factor.get(key)
                    for key in ("score", "status", "rationale", "missing", "weight")
                    if factor.get(key) is not None
                }
            row["factor_scores"] = factors
        missing_rows = (
            self._raw_objects("MissingInformationRequirement", filters={"parent_uid": uid}, limit=100) if uid else []
        )
        if missing_rows:
            row["missing_information"] = [
                {
                    "field": item.get("field"),
                    "severity": item.get("severity"),
                    "reason": item.get("reason"),
                    "status": item.get("status"),
                }
                for item in sorted(missing_rows, key=lambda item: str(item.get("id") or ""))
            ]
        return row

    def _project_idea_comparison_run(self, row: dict[str, Any]) -> dict[str, Any]:
        uid = str(row.get("id") or row.get("object_uid") or "")
        ranking_rows = self._raw_objects("IdeaComparisonRanking", filters={"comparison_run_id": uid}, limit=1000)
        if ranking_rows:
            rankings = sorted(
                ranking_rows, key=lambda item: (int(item.get("rank") or 0), str(item.get("ticker") or ""))
            )
            row["rankings"] = rankings
            row["ranking_count"] = len(rankings)
        return row

    def _project_optimization_run(self, row: dict[str, Any]) -> dict[str, Any]:
        uid = str(row.get("id") or row.get("object_uid") or row.get("run_id") or "")
        if uid:
            snapshots = self._raw_objects("OptimizationActionSnapshot", filters={"run_id": uid}, limit=1000)
            if snapshots:
                row["snapshots"] = sorted(
                    snapshots,
                    key=lambda item: (
                        str(item.get("ticker") or ""),
                        str(item.get("created_at") or item.get("updated_at") or ""),
                    ),
                )
            row = self._attach_source_freshness(row, uid)
        return row

    def _project_optimization_alert(self, row: dict[str, Any]) -> dict[str, Any]:
        current_uid = str(row.get("current_snapshot_id") or "").strip()
        previous_uid = str(row.get("previous_snapshot_id") or "").strip()
        if current_uid and not isinstance(row.get("current_snapshot"), dict):
            row["current_snapshot"] = self._raw_object(current_uid)
        if previous_uid and not isinstance(row.get("previous_snapshot"), dict):
            row["previous_snapshot"] = self._raw_object(previous_uid)
        uid = str(row.get("id") or row.get("object_uid") or "")
        return self._attach_source_freshness(row, uid) if uid else row

    def _attach_source_freshness(self, row: dict[str, Any], uid: str) -> dict[str, Any]:
        if isinstance(row.get("source_freshness"), dict):
            return row
        freshness_rows = self._raw_objects("SourceFreshness", filters={"parent_uid": uid}, limit=100)
        if not freshness_rows:
            return row
        source_freshness: dict[str, dict[str, Any]] = {}
        for item in freshness_rows:
            name = str(item.get("source_name") or "").strip()
            if not name:
                continue
            source_freshness[name] = {
                key: item.get(key)
                for key in ("status", "checked_at", "as_of", "freshness_category", "error", "metadata")
                if item.get(key) is not None
            }
        row["source_freshness"] = source_freshness
        return row

    def _project_management_quality_assessment(self, row: dict[str, Any]) -> dict[str, Any]:
        uid = str(row.get("id") or row.get("object_uid") or "")
        if not uid:
            return row
        for child_type, key in (
            ("ManagementQualityScorecardRow", "scorecard"),
            ("ManagementQualityAccomplishment", "accomplishments"),
            ("ManagementQualitySetback", "setbacks"),
        ):
            if isinstance(row.get(key), list):
                continue
            children = self._raw_objects(child_type, filters={"assessment_id": uid}, limit=200)
            if children:
                row[key] = sorted(children, key=lambda item: int(item.get("ordinal") or 0))
        row["parsed"] = _management_quality_parsed_from_assessment(row)
        return row


def _clean_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (filters or {}).items() if value is not None and value != ""}


def _ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _ticker_status_filter(ticker: str | None, status: str | None) -> dict[str, Any]:
    return _clean_filters({"ticker": _ticker(ticker) if ticker else None, "status": status})


def _workflow_run_identity(row: dict[str, Any]) -> str:
    return str(row.get("run_id") or row.get("object_uid") or row.get("id") or "").strip()


def _workflow_run_sort_value(row: dict[str, Any]) -> str:
    for key in ("updated_at", "completed_at", "started_at", "created_at"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    meta = row.get("_meta")
    temporal = meta.get("temporal") if isinstance(meta, dict) else None
    if isinstance(temporal, dict):
        return str(temporal.get("tx_from") or temporal.get("valid_from") or "").strip()
    return ""


def _sort_workflow_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (_workflow_run_sort_value(row), _workflow_run_identity(row)),
        reverse=True,
    )


def _first(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _management_quality_parsed_from_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "overall_rating": assessment.get("overall_rating"),
        "bottom_line": assessment.get("bottom_line"),
        "owner_mindset": {
            "rating": assessment.get("owner_mindset_rating"),
            "text": assessment.get("owner_mindset_text"),
        },
        "business_value_understanding": {
            "rating": assessment.get("business_value_understanding_rating"),
            "text": assessment.get("business_value_understanding_text"),
        },
        "follow_through": {
            "rating": assessment.get("follow_through_rating"),
            "text": assessment.get("follow_through_text"),
        },
    }
    compact_summary = {key: value for key, value in summary.items() if value not in (None, "", {})}
    return {
        "summary": compact_summary or None,
        "scorecard": [
            {
                "question": row.get("question"),
                "rating": row.get("rating"),
                "evidence": row.get("evidence"),
            }
            for row in assessment.get("scorecard", [])
            if isinstance(row, dict)
        ]
        or None,
        "accomplishments": [
            {"title": row.get("title"), "text": row.get("text")}
            for row in assessment.get("accomplishments", [])
            if isinstance(row, dict)
        ]
        or None,
        "setbacks": [
            {
                "title": row.get("title"),
                "text": row.get("text"),
                "response_rating": row.get("response_rating"),
                "response_text": row.get("response_text"),
            }
            for row in assessment.get("setbacks", [])
            if isinstance(row, dict)
        ]
        or None,
    }
