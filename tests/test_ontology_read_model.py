from __future__ import annotations

from pathlib import Path

from ontology.read_model import OPERATIONAL_READ_MODEL_VIEW, TemporalReadModelRepository


class _FakeConnection:
    def __init__(self):
        self.execute_calls: list[tuple[str, object | None]] = []
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql: str, params: object | None = None):
        self.execute_calls.append((sql, params))
        return self

    def commit(self):
        self.commits += 1


def test_refresh_uses_security_definer_function():
    conn = _FakeConnection()
    repo = TemporalReadModelRepository(connection_factory=lambda: conn)

    repo.refresh()

    assert conn.execute_calls == [("SELECT refresh_ontology_temporal_read_models()", None)]
    assert conn.commits == 1


def test_temporal_read_model_migration_grants_runtime_roles():
    migration = Path("migrations/versions/20260505_0006_ontology_temporal_read_models.py").read_text(encoding="utf-8")

    for view_name in (
        "ontology_current_position_risk_read_model",
        "ontology_current_position_signal_evidence_read_model",
        "ontology_current_position_thesis_context_read_model",
        "ontology_current_decision_lineage_read_model",
        "ontology_current_source_status_read_model",
        "ontology_current_computed_snapshot_read_model",
    ):
        assert view_name in migration

    assert "CREATE OR REPLACE FUNCTION refresh_ontology_temporal_read_models()" in migration
    assert "SECURITY DEFINER" in migration
    assert "GRANT SELECT ON TABLE %I TO talisman_app" in migration
    assert "GRANT SELECT ON TABLE %I TO talisman_worker" in migration
    assert "GRANT MAINTAIN ON TABLE %I TO talisman_app" in migration
    assert "GRANT MAINTAIN ON TABLE %I TO talisman_worker" in migration
    assert "GRANT EXECUTE ON FUNCTION refresh_ontology_temporal_read_models() TO talisman_app" in migration
    assert "GRANT EXECUTE ON FUNCTION refresh_ontology_temporal_read_models() TO talisman_worker" in migration


def test_operational_read_model_migration_contract():
    migration = Path("migrations/versions/20260505_0009_ontology_operational_read_model.py").read_text(encoding="utf-8")

    assert OPERATIONAL_READ_MODEL_VIEW in migration
    assert 'down_revision: str | None = "20260505_0008"' in migration
    assert "CREATE MATERIALIZED VIEW {OPERATIONAL_READ_MODEL_VIEW}" in migration
    assert "REFRESH MATERIALIZED VIEW {view_name}" in migration
    for index_name in (
        "uq_ontology_operational_read_model_object_uid",
        "idx_ontology_operational_read_model_ticker_status",
        "idx_ontology_operational_read_model_status",
        "idx_ontology_operational_read_model_approval_status",
        "idx_ontology_operational_read_model_report_as_of",
        "idx_ontology_operational_read_model_parent",
        "idx_ontology_operational_read_model_assessment",
        "idx_ontology_operational_read_model_run",
        "idx_ontology_operational_read_model_updated",
    ):
        assert index_name in migration
    for column in (
        "properties_json->>'ticker'",
        "properties_json->>'status'",
        "properties_json->>'application_status'",
        "properties_json->>'approval_status'",
        "properties_json->>'report_type'",
        "properties_json->>'parent_uid'",
        "properties_json->>'assessment_id'",
        "properties_json->>'run_id'",
        "properties_json->>'as_of'",
        "properties_json->>'evaluated_at'",
    ):
        assert column in migration


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None


class _ScriptedConnection:
    def __init__(self, results):
        self.results = list(results)
        self.execute_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql: str, params=None):
        self.execute_calls.append((sql, params))
        if not self.results:
            raise AssertionError(f"Unexpected query: {sql}")
        return _Rows(self.results.pop(0))


def _op_row(object_type: str, object_uid: str, props: dict, **extra):
    ticker = props.get("ticker")
    return {
        "version_id": extra.get("version_id", f"{object_uid}:version"),
        "object_uid": object_uid,
        "object_type": object_type,
        "business_key": extra.get("business_key", object_uid),
        "properties_json": dict(props),
        "schema_name": extra.get("schema_name", object_type),
        "schema_version": extra.get("schema_version", 1),
        "source_record_id": None,
        "valid_from": extra.get("valid_from", "2026-05-05T00:00:00Z"),
        "valid_to": None,
        "tx_from": extra.get("tx_from", "2026-05-05T00:00:00Z"),
        "tx_to": None,
        "actor_id": None,
        "input_hash": None,
        "supersedes_version_id": None,
        "temporal_confidence": "native",
        "ticker": str(ticker).upper() if ticker else extra.get("ticker"),
        "status": str(props.get("status")).lower() if props.get("status") else extra.get("status"),
        "application_status": extra.get("application_status"),
        "approval_status": str(props.get("approval_status")).lower() if props.get("approval_status") else None,
        "outcome_status": None,
        "report_type": str(props.get("report_type")).lower() if props.get("report_type") else None,
        "parent_uid": props.get("parent_uid"),
        "assessment_id": props.get("assessment_id"),
        "run_id": props.get("run_id"),
        "current_snapshot_id": props.get("current_snapshot_id"),
        "previous_snapshot_id": props.get("previous_snapshot_id"),
        "as_of_sort": props.get("as_of", ""),
        "evaluated_at_sort": props.get("evaluated_at", ""),
        "created_at_sort": props.get("created_at", ""),
        "updated_sort": props.get("updated_at", "2026-05-05T00:00:00Z"),
    }


def test_workspace_bundle_uses_operational_read_model_and_groups_rows():
    alert = _op_row(
        "OptimizationAlert",
        "optimization_alert:1",
        {
            "id": "optimization_alert:1",
            "ticker": "MU",
            "status": "open",
            "current_snapshot_id": "optimization_snapshot:1",
        },
    )
    conn = _ScriptedConnection(
        [
            [
                _op_row(
                    "Evaluation", "evaluation:new", {"ticker": "MU", "evaluated_at": "2026-05-05", "action": "trim"}
                ),
                _op_row(
                    "Evaluation", "evaluation:old", {"ticker": "MU", "evaluated_at": "2026-05-01", "action": "hold"}
                ),
            ],
            [_op_row("Thesis", "thesis:MU", {"ticker": "MU", "status": "active"})],
            [_op_row("Approval", "approval:1", {"ticker": "MU", "status": "pending"})],
            [_op_row("Recommendation", "recommendation:daily", {"report_type": "daily", "as_of": "2026-05-05"})],
            [_op_row("Recommendation", "recommendation:weekly", {"report_type": "weekly", "as_of": "2026-05-04"})],
            [
                _op_row(
                    "Recommendation", "recommendation:pending", {"approval_status": "pending", "as_of": "2026-05-05"}
                )
            ],
            [
                _op_row(
                    "CourseOfAction",
                    "course_of_action:pending",
                    {"approval_status": "pending", "action": "add", "ticker": "MU", "as_of": "2026-05-05"},
                )
            ],
            [
                _op_row(
                    "CourseOfAction",
                    "course_of_action:recent",
                    {"approval_status": "none", "action": "watch", "ticker": "MU", "as_of": "2026-05-04"},
                )
            ],
            [
                _op_row(
                    "CourseOfActionComparison",
                    "course_of_action_comparison:1",
                    {"status": "open", "objective": "Compare MU actions", "as_of": "2026-05-05"},
                )
            ],
            [_op_row("ActionItem", "action_item:1", {"ticker": "MU", "status": "open"})],
            [alert],
            [_op_row("OptimizationActionSnapshot", "optimization_snapshot:1", {"ticker": "MU", "action": "Trim Long"})],
            [
                _op_row(
                    "SourceFreshness",
                    "source_freshness:1",
                    {"parent_uid": "optimization_alert:1", "source_name": "risk", "status": "ok"},
                )
            ],
            [_op_row("WatchTrigger", "watch_trigger:1", {"ticker": "MU", "status": "active"})],
            [_op_row("WorkflowRun", "workflow_run:1", {"ticker": "MU"})],
            [_op_row("ReportRun", "report_run:1", {"as_of": "2026-05-05"})],
            [_op_row("ThesisClaim", "claim:1", {"ticker": "MU", "status": "challenged"})],
            [_op_row("ThesisClaim", "claim:2", {"ticker": "MU", "status": "disconfirmed"})],
        ]
    )
    repo = TemporalReadModelRepository(connection_factory=lambda: conn)

    bundle = repo.fetch_workspace_bundle()

    assert all(OPERATIONAL_READ_MODEL_VIEW in sql for sql, _params in conn.execute_calls)
    assert [row["object_uid"] for row in bundle["latest_evaluations"]] == ["evaluation:new"]
    assert bundle["latest_daily_recommendation"]["object_uid"] == "recommendation:daily"
    assert bundle["pending_course_of_actions"][0]["object_uid"] == "course_of_action:pending"
    assert bundle["recent_course_of_actions"][0]["object_uid"] == "course_of_action:recent"
    assert bundle["open_course_of_action_comparisons"][0]["object_uid"] == "course_of_action_comparison:1"
    assert bundle["optimizer_alerts"][0]["current_snapshot"]["object_uid"] == "optimization_snapshot:1"
    assert bundle["optimizer_alerts"][0]["source_freshness"]["risk"]["status"] == "ok"
    assert bundle["challenged_claims"][0]["object_uid"] == "claim:1"
    assert bundle["disconfirmed_claims"][0]["object_uid"] == "claim:2"


def test_dossier_bundle_filters_by_ticker_and_attaches_management_quality_children():
    conn = _ScriptedConnection(
        [
            [_op_row("Position", "position:MU", {"ticker": "MU"})],
            [_op_row("Thesis", "thesis:MU", {"ticker": "MU", "status": "active"})],
            [
                _op_row(
                    "ManagementQualityAssessment",
                    "management_quality_assessment:MU",
                    {"ticker": "MU", "status": "active", "overall_rating": "Strong"},
                )
            ],
            [
                _op_row(
                    "ManagementQualityScorecardRow",
                    "management_quality_scorecard_row:1",
                    {
                        "assessment_id": "management_quality_assessment:MU",
                        "ordinal": 1,
                        "question": "Capital allocation",
                    },
                )
            ],
            [
                _op_row(
                    "ManagementQualityAccomplishment",
                    "management_quality_accomplishment:1",
                    {"assessment_id": "management_quality_assessment:MU", "ordinal": 1, "title": "Execution"},
                )
            ],
            [
                _op_row(
                    "ManagementQualitySetback",
                    "management_quality_setback:1",
                    {"assessment_id": "management_quality_assessment:MU", "ordinal": 1, "title": "Cycle"},
                )
            ],
            [_op_row("Evaluation", "evaluation:MU", {"ticker": "MU", "evaluated_at": "2026-05-05"})],
            [_op_row("Catalyst", "catalyst:MU", {"ticker": "MU"})],
            [_op_row("KillCondition", "kill_condition:MU", {"ticker": "MU"})],
            [_op_row("ThesisClaim", "thesis_claim:MU", {"ticker": "MU"})],
            [_op_row("WorkflowRun", "workflow_run:MU", {"ticker": "MU"})],
            [_op_row("ActionItem", "action_item:MU", {"ticker": "MU", "status": "open"})],
            [_op_row("WatchTrigger", "watch_trigger:MU", {"ticker": "MU"})],
            [_op_row("Approval", "approval:MU", {"ticker": "MU", "status": "pending"})],
        ]
    )
    repo = TemporalReadModelRepository(connection_factory=lambda: conn)

    bundle = repo.fetch_dossier_bundle("mu")

    first_sql, first_params = conn.execute_calls[0]
    assert OPERATIONAL_READ_MODEL_VIEW in first_sql
    assert first_params == ("Position", "MU", 1)
    assert bundle["position"]["object_uid"] == "position:MU"
    assert bundle["thesis_meta"]["object_uid"] == "thesis:MU"
    assessment = bundle["management_quality_assessment"]
    assert assessment["scorecard"][0]["object_uid"] == "management_quality_scorecard_row:1"
    assert assessment["accomplishments"][0]["object_uid"] == "management_quality_accomplishment:1"
    assert assessment["setbacks"][0]["object_uid"] == "management_quality_setback:1"
