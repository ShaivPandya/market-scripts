"""Ontology-backed runtime read helpers.

This module is the runtime replacement for the old domain SQLite readers. It
keeps API routers from depending on legacy table modules while read models are
still being filled out route by route.
"""

from __future__ import annotations

from typing import Any

from ontology.object_service import OntologyObjectService


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
    _ = fallback_to_csv
    return OntologyRuntimeReadService().positions_df(include_hedges=include_hedges)


def get_hedge_positions() -> list[dict[str, Any]]:
    rows = OntologyRuntimeReadService().positions(include_hedges=True)
    return [row for row in rows if str(row.get("role") or "").lower() == "hedge"]


class OntologyRuntimeReadService:
    def __init__(self, object_service: OntologyObjectService | None = None):
        self.objects = object_service or OntologyObjectService()

    def get(self, object_uid: str) -> dict[str, Any] | None:
        row = self.objects.get_object(object_uid)
        return object_props(row) if row else None

    def list_objects(
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

    def research_notes(self, *, ticker: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        filters = {"ticker": _ticker(ticker)} if ticker else None
        return self.list_objects("ResearchNote", filters=filters, limit=limit)

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
        filters = {"ticker": _ticker(ticker)} if ticker else None
        return self.list_objects("WorkflowRun", filters=filters, limit=limit)

    def report_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.list_objects("ReportRun", limit=limit)
        return sorted(rows, key=lambda row: str(row.get("as_of") or row.get("synced_at") or ""), reverse=True)


def _clean_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (filters or {}).items() if value is not None and value != ""}


def _ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _ticker_status_filter(ticker: str | None, status: str | None) -> dict[str, Any]:
    return _clean_filters({"ticker": _ticker(ticker) if ticker else None, "status": status})


def _first(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None
