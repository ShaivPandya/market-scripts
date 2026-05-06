from __future__ import annotations

import json
from pathlib import Path

import pytest

import portfolio.core_db as core_db
from api import governance
from api.audit import AuditWriteError, emit_audit_event


@pytest.fixture(autouse=True)
def _temp_core_db(tmp_path, monkeypatch):
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    monkeypatch.setenv("STATE_STORAGE_BACKEND", "local")
    yield
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)


def test_audit_helper_can_fail_closed(monkeypatch):
    def _raise(**_kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(core_db, "record_audit_event", _raise)

    assert emit_audit_event("test.best_effort", "test", "failed") is None
    with pytest.raises(AuditWriteError):
        emit_audit_event("test.critical", "test", "failed", fail_closed=True)


def test_record_now_normalizes_governance_link_verbs(monkeypatch):
    calls: list[dict] = []

    class _FakeObjects:
        pass

    def _capture_link(**kwargs):
        calls.append(kwargs)
        return {"relation_uid": "provenance:test"}

    monkeypatch.setattr(governance, "OntologyObjectService", _FakeObjects)
    monkeypatch.setattr(governance, "write_provenance_relation", _capture_link)

    root = governance.lineage_root(governance.REF_RECOMMENDATION, 42)
    bundle = governance.event_bundle(
        lineage_root_id=root,
        provenance_links=[
            governance.provenance_link(
                event_id="pv:gated",
                source_ref_type=governance.REF_POLICY_GATE_RESULT,
                source_ref_id=7,
                target_ref_type=governance.REF_RECOMMENDATION,
                target_ref_id=42,
                link_type=governance.LINK_GATED,
                lineage_root_id=root,
            ),
            governance.provenance_link(
                event_id="pv:applied",
                source_ref_type=governance.REF_APPROVAL,
                source_ref_id=9,
                target_ref_type=governance.REF_ACTION_RUN,
                target_ref_id=10,
                link_type=governance.LINK_APPLIED_BY,
                lineage_root_id=root,
                metadata={"note": "apply"},
            ),
            governance.provenance_link(
                event_id="pv:evaluated",
                source_ref_type=governance.REF_POLICY_GATE_RESULT,
                source_ref_id=7,
                target_ref_type=governance.REF_RECOMMENDATION,
                target_ref_id=42,
                link_type=governance.LINK_EVALUATED,
                lineage_root_id=root,
                metadata="legacy metadata",
            ),
        ],
    )

    result = governance.record_now_tx(None, bundle)

    assert result["provenance_links"] == 3
    assert [call["link_type"] for call in calls] == [
        governance.LINK_USED,
        governance.LINK_APPROVED_EXECUTION,
        governance.LINK_USED,
    ]
    assert all(call["fail_closed"] is True for call in calls)
    assert calls[0]["metadata"] == {"governance_link_type": governance.LINK_GATED}
    assert calls[1]["metadata"]["note"] == "apply"
    assert calls[1]["metadata"]["field_names"] == ["note"]
    assert calls[1]["metadata"]["governance_link_type"] == governance.LINK_APPLIED_BY
    assert calls[2]["metadata"] == {
        "metadata": "legacy metadata",
        "governance_link_type": governance.LINK_EVALUATED,
    }
    assert bundle["provenance_links"][1]["metadata"] == {"field_names": ["note"], "note": "apply"}


def test_governance_outbox_replays_idempotently_and_redacts():
    root = governance.lineage_root(governance.REF_RECOMMENDATION, 42)
    event_id = governance.deterministic_id("pv:recommendation", 42, "generated")
    bundle = {
        "lineage_root_id": root,
        "idempotency_key": "governance:test:recommendation:42",
        "provenance_events": [
            governance.provenance_event(
                event_id=event_id,
                event_type="recommendation",
                event_name=governance.EVENT_RECOMMENDATION_GENERATED,
                lineage_root_id=root,
                input_value={"prompt": "RAW PROMPT", "account_payload": {"secret": "VALUE"}},
                summary={"recommendation_id": 42, "prompt": "RAW PROMPT"},
            )
        ],
        "audit_events": [
            governance.audit_event(
                action_name=governance.EVENT_RECOMMENDATION_GENERATED,
                status="succeeded",
                lineage_root_id=root,
                object_refs=[{"type": governance.REF_RECOMMENDATION, "id": 42}],
                metadata={"prompt": "RAW PROMPT", "tool_output": "SECRET TOOL OUTPUT"},
            )
        ],
    }

    first = core_db.enqueue_governance_outbox(bundle)
    second = core_db.enqueue_governance_outbox(bundle)
    assert first["idempotency_key"] == second["idempotency_key"]

    result = core_db.drain_governance_outbox()
    assert result["completed"] == 1
    assert core_db.drain_governance_outbox()["claimed"] == 0

    trace = core_db.get_provenance_trace(ref_type=governance.REF_RECOMMENDATION, ref_id="42")
    audits = core_db.get_audit_events(lineage_root_id=root)
    serialized = json.dumps({"trace": trace, "audits": audits}, default=str)
    assert len(core_db.get_governance_outbox_items(status="completed")) == 1
    assert len(audits) == 1
    assert "RAW PROMPT" not in serialized
    assert "SECRET TOOL OUTPUT" not in serialized


def test_governance_lineage_migration_creates_policy_gate_schema():
    migration = Path("migrations/versions/20260503_0012_governance_outbox_lineage.py").read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS policy_gate_results" in migration
    assert "idx_policy_gate_results_decision" in migration
    assert "policy_gate_result_id integer" in migration
    assert "policy_gate_warnings_json text" in migration


def test_approval_creation_requires_and_records_financial_lineage():
    approval = core_db.create_pending_approval(
        "action_item",
        {"description": "Review sizing", "prompt": "RAW USER PROMPT", "token": "secret-token"},
        ticker="MU",
        source_type="agent",
        source_id="session-1",
        action_id="create_action_item",
        risk_class="financial",
    )

    assert approval["provenance_event_id"]
    trace = core_db.get_provenance_trace(approval_id=approval["id"])
    audits = core_db.get_audit_events(lineage_root_id=f"approval:{approval['id']}")
    serialized = json.dumps({"trace": trace, "audits": audits}, default=str)

    assert any(event["event_name"] == governance.EVENT_APPROVAL_CREATED for event in trace["events"])
    assert any(event["criticality"] == governance.CRITICAL_FINANCIAL for event in audits)
    assert "RAW USER PROMPT" not in serialized
    assert "secret-token" not in serialized


def test_policy_gate_and_recommendation_have_complete_lineage():
    gate = core_db.create_policy_gate_result(
        {
            "decision": "review_required",
            "review_required": True,
            "override_acknowledged": False,
            "account_id": "acct-1",
            "portfolio_id": "portfolio-1",
            "policy_id": "policy-1",
            "mandate_id": "mandate-1",
            "evaluated_at": "2026-05-04T12:00:00+00:00",
            "action_id": "create_recommendation",
        },
        action_id="create_recommendation",
        target_type="recommendation",
        target_id="rec-key-1",
        payload={"raw_account_payload": "SHOULD NOT APPEAR"},
    )
    rec = core_db.upsert_recommendation(
        {
            "report_type": "daily",
            "as_of": "2026-05-04",
            "stance": "risk_review",
            "action": "buy",
            "ticker": "MU",
            "rationale": "Sizing review",
            "idempotency_key": "rec-key-1",
            "policy_gate_result_id": gate["id"],
            "model": "gpt-test",
            "prompt_hash": "prompt-hash",
            "input_hash": "input-hash",
        }
    )

    assert gate["lineage_completeness"] == "complete"
    assert rec["lineage_completeness"] == "complete"
    report = core_db.get_decision_lineage_report(recommendation_id=rec["id"])
    assert report["lineage_root_id"] == f"recommendation:{rec['id']}"
    assert not report["completeness_warnings"]
    assert any(link["target_ref_type"] == governance.REF_RECOMMENDATION for link in report["provenance"]["links"])


def test_workflow_execution_does_not_fall_back_to_ephemeral_run(monkeypatch):
    from api import workflows

    def _raise(*_args, **_kwargs):
        raise RuntimeError("workflow run storage unavailable")

    monkeypatch.setattr(core_db, "create_workflow_run", _raise)

    with pytest.raises(RuntimeError, match="workflow run storage unavailable"):
        workflows.execute_workflow("morning_brief")
