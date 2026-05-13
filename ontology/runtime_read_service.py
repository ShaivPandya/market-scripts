"""Ontology-backed runtime read helpers.

This module is the runtime replacement for the old domain SQLite readers. It
keeps API routers from depending on legacy table modules while read models are
still being filled out route by route.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, cast

from ontology.object_service import OntologyObjectService

logger = logging.getLogger(__name__)


def _ontology_primary_writes_enabled() -> bool:
    try:
        from ontology.domain_write_service import ontology_primary_writes_enabled

        return ontology_primary_writes_enabled()
    except Exception:
        return False


def _ontology_read_model_enabled() -> bool:
    try:
        from ontology.domain_write_service import ontology_read_model_enabled

        return ontology_read_model_enabled()
    except Exception:
        return False


def get_positions(*, include_hedges: bool = False) -> list[dict[str, Any]]:
    """Ontology-native replacement for legacy portfolio_db.get_positions."""
    if not _ontology_primary_writes_enabled():
        portfolio_db = importlib.import_module("portfolio.portfolio_db")

        return _legacy_positions(portfolio_db, include_hedges=include_hedges)
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
    """Ontology-native replacement for legacy portfolio_db.get_positions_df."""
    if not _ontology_primary_writes_enabled():
        _legacy_get_positions_df = importlib.import_module("portfolio.portfolio_db").get_positions_df

        return _legacy_get_positions_df(include_hedges=include_hedges, fallback_to_csv=fallback_to_csv)
    return OntologyRuntimeReadService().positions_df(include_hedges=include_hedges)


def get_hedge_positions() -> list[dict[str, Any]]:
    if not _ontology_primary_writes_enabled():
        _legacy_get_hedge_positions = importlib.import_module("portfolio.portfolio_db").get_hedge_positions

        return cast(list[dict[str, Any]], _legacy_get_hedge_positions())
    rows = OntologyRuntimeReadService().positions(include_hedges=True)
    return [row for row in rows if str(row.get("role") or "").lower() == "hedge"]


class OntologyRuntimeReadService:
    def __init__(self, object_service: OntologyObjectService | None = None, read_model_repository: Any | None = None):
        self.objects = object_service or OntologyObjectService()
        self.read_model_repo = read_model_repository

    def get(self, object_uid: str) -> dict[str, Any] | None:
        if not _ontology_primary_writes_enabled():
            return _legacy_get(object_uid)
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
        if not _ontology_primary_writes_enabled():
            return _legacy_list_objects(object_type, filters=filters, limit=limit)
        return [
            self._project_object(object_type, object_props(row))
            for row in self.objects.query_objects(object_type, filters=_clean_filters(filters), limit=limit)
        ]

    def workspace_bundle(self) -> dict[str, Any]:
        if not _ontology_primary_writes_enabled() or not _ontology_read_model_enabled():
            return self._workspace_bundle_from_objects()
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
        if not _ontology_primary_writes_enabled() or not _ontology_read_model_enabled():
            return self._dossier_bundle_from_objects(ticker)
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
        if not _ontology_primary_writes_enabled():
            return None
        normalized = _ticker(ticker)
        rows = self.list_objects("ManagementQualityAssessment", filters={"ticker": normalized}, limit=20)
        active = [row for row in rows if str(row.get("status") or "active").lower() == "active"]
        candidates = active or rows
        candidates.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
        return candidates[0] if candidates else None

    def positions(self, *, include_hedges: bool = False, limit: int = 1000) -> list[dict[str, Any]]:
        if not _ontology_primary_writes_enabled():
            portfolio_db = importlib.import_module("portfolio.portfolio_db")

            return _legacy_positions(portfolio_db, include_hedges=include_hedges)[:limit]
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
        if _ontology_primary_writes_enabled():
            return self._fresh_workflow_runs_from_objects(ticker=ticker, limit=limit)
        filters = {"ticker": _ticker(ticker)} if ticker else None
        return _sort_workflow_runs(self.list_objects("WorkflowRun", filters=filters, limit=limit))[:limit]

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
            "recent_workflow_runs": self.workflow_runs(limit=3),
            "recent_report_runs": self.report_runs(limit=5),
            "challenged_claims": self.thesis_claims(status="challenged", limit=5),
            "disconfirmed_claims": self.thesis_claims(status="disconfirmed", limit=5),
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
            "pending_approvals": self.approvals(ticker=normalized, status="pending"),
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
        fetch_limit = 500 if _ontology_primary_writes_enabled() else limit
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


def _legacy_get(object_uid: str) -> dict[str, Any] | None:
    prefix, _, raw_id = str(object_uid or "").partition(":")
    if not prefix or not raw_id:
        return None
    from portfolio import core_db

    if prefix == "action_item":
        return _find_by_id(core_db.get_action_items(), raw_id)
    if prefix == "watch_trigger":
        return _find_by_id(core_db.get_watch_triggers(), raw_id)
    if prefix == "approval":
        try:
            return core_db.get_pending_approval(int(raw_id))
        except (TypeError, ValueError):
            return None
    if prefix == "workflow_run":
        return core_db.get_workflow_run(raw_id)
    if prefix == "thesis":
        thesis_db = importlib.import_module("portfolio.thesis_db")

        return cast(dict[str, Any] | None, thesis_db.get_thesis_meta(raw_id))
    if prefix == "investment_idea":
        try:
            return core_db.get_investment_idea(int(raw_id))
        except (TypeError, ValueError):
            return None
    if prefix == "idea_evaluation":
        try:
            return core_db.get_idea_evaluation(int(raw_id))
        except (TypeError, ValueError):
            return None
    if prefix == "idea_comparison_run":
        return core_db.get_idea_comparison_run(raw_id)
    if prefix in {"optimizationmission", "optimization_mission"}:
        return core_db.get_optimization_mission(_optional_int(raw_id))
    if prefix in {"optimizationrun", "optimization_run"}:
        return core_db.get_optimization_run(raw_id)
    if prefix in {"optimizationalert", "optimization_alert"}:
        alert_id = _optional_int(raw_id)
        if alert_id is None:
            return None
        for alert in core_db.get_optimization_alerts(status=None, limit=200):
            if _optional_int(alert.get("id")) == alert_id:
                return alert
        return None
    if prefix == "recommendation":
        try:
            return core_db.get_recommendation(int(raw_id))
        except (TypeError, ValueError):
            return None
    if prefix == "thesis_claim":
        try:
            return core_db.get_thesis_claim(int(raw_id))
        except (TypeError, ValueError):
            return None
    if prefix == "document_artifact" and raw_id.startswith("news_digest:"):
        from portfolio import news_digests

        digest_id = raw_id.removeprefix("news_digest:")
        try:
            digest = news_digests.get_digest(digest_id)
        except FileNotFoundError:
            return None
        return _digest_artifact(digest)
    return None


def _legacy_list_objects(
    object_type: str,
    *,
    filters: dict[str, Any] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    filters = _clean_filters(filters)
    from portfolio import core_db

    portfolio_db = importlib.import_module("portfolio.portfolio_db")

    if object_type == "Position":
        return _legacy_positions(portfolio_db, include_hedges=False)[:limit]
    if object_type == "HedgePosition":
        return cast(list[dict[str, Any]], portfolio_db.get_hedge_positions())[:limit]
    if object_type == "Thesis":
        thesis_db = importlib.import_module("portfolio.thesis_db")
        thesis_rows = cast(list[dict[str, Any]], thesis_db.get_all_thesis_meta())
        ticker = filters.get("ticker")
        status = filters.get("status")
        return [
            row
            for row in thesis_rows
            if (not ticker or str(row.get("ticker") or "").upper() == str(ticker).upper())
            and (not status or row.get("status") == status)
        ][:limit]
    if object_type == "Evaluation":
        thesis_db = importlib.import_module("portfolio.thesis_db")
        ticker = filters.get("ticker")
        if ticker:
            return cast(list[dict[str, Any]], thesis_db.get_evaluations(str(ticker).upper(), limit=limit))
        return cast(list[dict[str, Any]], thesis_db.get_latest_evaluations())[:limit]
    if object_type == "ActionItem":
        return core_db.get_action_items(status=filters.get("status"), ticker=filters.get("ticker"))[:limit]
    if object_type == "WatchTrigger":
        return core_db.get_watch_triggers(status=filters.get("status"), ticker=filters.get("ticker"))[:limit]
    if object_type == "Catalyst":
        ticker = filters.get("ticker")
        catalyst_rows = core_db.get_catalysts(str(ticker)) if ticker else []
        status = filters.get("status")
        return [row for row in catalyst_rows if not status or row.get("status") == status][:limit]
    if object_type == "KillCondition":
        ticker = filters.get("ticker")
        condition_rows = core_db.get_kill_conditions(str(ticker)) if ticker else []
        status = filters.get("status")
        return [row for row in condition_rows if not status or row.get("status") == status][:limit]
    if object_type == "ThesisClaim":
        return core_db.get_thesis_claims(
            ticker=filters.get("ticker"),
            status=filters.get("status"),
            limit=limit,
        )
    if object_type == "Approval":
        return core_db.get_pending_approvals(
            ticker=filters.get("ticker"),
            status=filters.get("status"),
            application_status=filters.get("application_status"),
        )[:limit]
    if object_type == "Recommendation":
        return core_db.get_recommendations(
            report_type=filters.get("report_type"),
            status=filters.get("status"),
            ticker=filters.get("ticker"),
            approval_status=filters.get("approval_status"),
            outcome_status=filters.get("outcome_status"),
            limit=limit,
        )
    if object_type == "WorkflowRun":
        return core_db.get_workflow_runs(ticker=filters.get("ticker"), limit=limit)
    if object_type == "DocumentArtifact":
        document_type = filters.get("document_type")
        document_id = filters.get("document_id")
        if document_type and document_type != "news_digest":
            return []
        from portfolio import news_digests

        if document_id:
            try:
                return [_digest_artifact(news_digests.get_digest(str(document_id)))]
            except FileNotFoundError:
                return []
        return [_digest_artifact(item) for item in news_digests.list_digests().get("items", [])][:limit]
    if object_type == "InvestmentIdea":
        return core_db.list_investment_ideas(
            status=filters.get("status"),
            include_archived=True,
            limit=limit,
        )
    if object_type == "IdeaEvaluation":
        idea_id = _optional_int(filters.get("idea_id"))
        if idea_id is not None:
            return core_db.get_idea_evaluations(idea_id, limit=limit)
        idea_rows: list[dict[str, Any]] = []
        for idea in core_db.list_investment_ideas(include_archived=True, limit=500):
            idea_rows.extend(core_db.get_idea_evaluations(int(idea["id"]), limit=limit))
        return sorted(idea_rows, key=lambda row: str(row.get("evaluated_at") or ""), reverse=True)[:limit]
    if object_type == "IdeaComparisonRun":
        return core_db.list_idea_comparison_runs(limit=limit)
    if object_type == "OptimizationMission":
        status = filters.get("status")
        return core_db.get_optimization_missions(status=status)[:limit]
    if object_type == "OptimizationRun":
        return core_db.get_optimization_runs(mission_id=_optional_int(filters.get("mission_id")), limit=limit)
    if object_type == "OptimizationActionSnapshot":
        return core_db.get_optimization_snapshots(
            run_id=filters.get("run_id"),
            mission_id=_optional_int(filters.get("mission_id")),
            ticker=filters.get("ticker"),
        )[:limit]
    if object_type == "OptimizationAlert":
        rows = core_db.get_optimization_alerts(
            mission_id=_optional_int(filters.get("mission_id")),
            status=filters.get("status"),
            limit=limit,
        )
        ticker = filters.get("ticker")
        return [row for row in rows if not ticker or str(row.get("ticker") or "").upper() == str(ticker).upper()]
    return []


def _legacy_positions(portfolio_db: Any, *, include_hedges: bool) -> list[dict[str, Any]]:
    try:
        return cast(list[dict[str, Any]], portfolio_db.get_positions(include_hedges=include_hedges))
    except TypeError as exc:
        if "include_hedges" not in str(exc):
            raise
        return cast(list[dict[str, Any]], portfolio_db.get_positions())


def _find_by_id(rows: list[dict[str, Any]], raw_id: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("id")) == str(raw_id):
            return row
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _digest_artifact(digest: dict[str, Any]) -> dict[str, Any]:
    digest_id = str(digest.get("id") or digest.get("digest_id") or "")
    return {
        **digest,
        "id": f"document_artifact:news_digest:{digest_id}" if digest_id else "",
        "object_uid": f"document_artifact:news_digest:{digest_id}" if digest_id else "",
        "document_type": "news_digest",
        "document_id": digest_id,
        "status": "active",
    }


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
