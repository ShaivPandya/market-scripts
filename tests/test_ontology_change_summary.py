from __future__ import annotations

from ontology.change_summary import OntologyChangeSummaryService


def _row(
    object_type: str,
    object_uid: str,
    props: dict,
    *,
    tx_from: str,
    tx_to: str | None = None,
):
    temporal = {"tx_from": tx_from}
    if tx_to:
        temporal["tx_to"] = tx_to
    return {
        "object_type": object_type,
        "object_uid": object_uid,
        "properties": dict(props),
        "_meta": {"temporal": temporal},
    }


class _FakeObjects:
    def __init__(self, rows_by_type: dict[str, list[dict]]):
        self.rows_by_type = rows_by_type
        self.calls: list[dict] = []

    def query_objects(self, object_type, filters=None, include_history=False, limit=100, **_kwargs):
        self.calls.append(
            {
                "object_type": object_type,
                "filters": filters,
                "include_history": include_history,
                "limit": limit,
            }
        )
        rows = list(self.rows_by_type.get(object_type, []))
        ticker = (filters or {}).get("ticker")
        if ticker:
            rows = [row for row in rows if str(row.get("properties", {}).get("ticker") or "").upper() == ticker]
        return rows[:limit]


def test_workspace_baseline_prefers_latest_report_run_over_workflow():
    service = OntologyChangeSummaryService(object_service=_FakeObjects({}), now="2026-05-31T12:00:00Z")

    summary = service.workspace_summary(
        {
            "recent_report_runs": [{"object_uid": "report_run:1", "as_of": "2026-05-20"}],
            "recent_workflow_runs": [
                {
                    "object_uid": "workflow_run:1",
                    "status": "succeeded",
                    "completed_at": "2026-05-25T10:00:00Z",
                }
            ],
        }
    )

    assert summary["baseline"]["kind"] == "last_report_run"
    assert summary["baseline"]["source_id"] == "report_run:1"
    assert summary["baseline"]["at"] == "2026-05-20T00:00:00+00:00"


def test_workspace_baseline_falls_back_to_seven_day_lookback():
    service = OntologyChangeSummaryService(object_service=_FakeObjects({}), now="2026-05-31T12:00:00Z")

    summary = service.workspace_summary({})

    assert summary["baseline"]["kind"] == "lookback"
    assert summary["baseline"]["days"] == 7
    assert summary["baseline"]["at"] == "2026-05-24T12:00:00+00:00"


def test_change_summary_reports_created_and_updated_whitelisted_fields():
    rows = {
        "ActionItem": [
            _row(
                "ActionItem",
                "action_item:1",
                {
                    "ticker": "MU",
                    "description": "Research MU memory pricing",
                    "action_type": "research",
                    "urgency": "high",
                    "status": "open",
                    "created_at": "2026-05-14T10:00:00Z",
                    "internal_note": "do not expose",
                },
                tx_from="2026-05-14T10:00:00Z",
            )
        ],
        "Catalyst": [
            _row(
                "Catalyst",
                "catalyst:1",
                {
                    "ticker": "MU",
                    "description": "HBM pricing inflects",
                    "status": "played_out",
                    "updated_at": "2026-05-15T10:00:00Z",
                    "internal_note": "do not expose",
                },
                tx_from="2026-05-15T10:00:00Z",
            ),
            _row(
                "Catalyst",
                "catalyst:1",
                {
                    "ticker": "MU",
                    "description": "HBM pricing inflects",
                    "status": "pending",
                    "updated_at": "2026-05-01T10:00:00Z",
                },
                tx_from="2026-05-01T10:00:00Z",
                tx_to="2026-05-15T10:00:00Z",
            ),
        ],
    }
    service = OntologyChangeSummaryService(object_service=_FakeObjects(rows), now="2026-05-31T12:00:00Z")

    summary = service.workspace_summary({}, since="2026-05-10T00:00:00Z")

    assert summary["counts"]["total"] == 2
    created = next(item for item in summary["items"] if item["object_uid"] == "action_item:1")
    updated = next(item for item in summary["items"] if item["object_uid"] == "catalyst:1")
    assert created["change_kind"] == "created"
    assert created["severity"] == "warning"
    assert "internal_note" not in created["after"]
    assert updated["change_kind"] == "updated"
    assert updated["before"] == {"status": "pending"}
    assert updated["after"] == {"status": "played_out"}


def test_dossier_summary_filters_by_ticker_and_uses_workflow_baseline():
    fake = _FakeObjects(
        {
            "Evaluation": [
                _row(
                    "Evaluation",
                    "evaluation:mu",
                    {"ticker": "MU", "action": "trim", "updated_at": "2026-05-20T10:00:00Z"},
                    tx_from="2026-05-20T10:00:00Z",
                ),
                _row(
                    "Evaluation",
                    "evaluation:crwd",
                    {"ticker": "CRWD", "action": "hold", "updated_at": "2026-05-20T11:00:00Z"},
                    tx_from="2026-05-20T11:00:00Z",
                ),
            ],
            "ReportRun": [
                _row(
                    "ReportRun",
                    "report:mu",
                    {
                        "ticker": "MU",
                        "report_type": "daily",
                        "status": "completed",
                        "synced_at": "2026-05-20T09:00:00Z",
                    },
                    tx_from="2026-05-20T09:00:00Z",
                ),
                _row(
                    "ReportRun",
                    "report:crwd",
                    {
                        "ticker": "CRWD",
                        "report_type": "daily",
                        "status": "completed",
                        "synced_at": "2026-05-20T12:00:00Z",
                    },
                    tx_from="2026-05-20T12:00:00Z",
                ),
            ],
        }
    )
    service = OntologyChangeSummaryService(object_service=fake, now="2026-05-31T12:00:00Z")

    summary = service.dossier_summary(
        {
            "workflow_runs": [
                {
                    "object_uid": "workflow_run:crwd",
                    "ticker": "CRWD",
                    "status": "succeeded",
                    "completed_at": "2026-05-21T00:00:00Z",
                },
                {
                    "object_uid": "workflow_run:mu",
                    "ticker": "MU",
                    "status": "succeeded",
                    "completed_at": "2026-05-19T00:00:00Z",
                },
            ]
        },
        "MU",
    )

    assert summary["baseline"]["kind"] == "last_workflow_run"
    assert summary["baseline"]["source_id"] == "workflow_run:mu"
    assert [item["object_uid"] for item in summary["items"]] == ["evaluation:mu", "report:mu"]
    assert all(call["filters"] in ({"ticker": "MU"}, None) for call in fake.calls)
