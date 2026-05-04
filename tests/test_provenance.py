from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import portfolio.core_db as core_db
from api import agent_tools, provenance
from ontology.policy import admin_actor, agent_actor


@pytest.fixture(autouse=True)
def _use_temp_core_db(tmp_path, monkeypatch):
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    monkeypatch.setenv("STATE_STORAGE_BACKEND", "local")
    yield
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "_conn", None)


def test_workflow_run_provenance_uses_hashes_not_raw_synthesis():
    run = core_db.create_workflow_run("morning_brief", ticker="MU")
    core_db.complete_workflow_run(
        run["run_id"],
        synthesis="RAW SYNTHESIS WITH SECRET TOKEN",
        artifacts={"action_items": []},
        tool_sections=[{"tool": "get_portfolio", "data": {"ticker": "MU"}}],
    )

    trace = core_db.get_provenance_trace(workflow_run_id=run["run_id"])
    event_types = {event["event_type"] for event in trace["events"]}
    workflow_event = next(event for event in trace["events"] if event["event_type"] == "workflow_run")

    assert "workflow_run" in event_types
    assert workflow_event["output_hash"]
    assert workflow_event["summary"]["artifact_count"] == 1
    assert "RAW SYNTHESIS" not in json.dumps(trace, default=str)
    assert "SECRET TOKEN" not in json.dumps(trace, default=str)
    assert trace["timeline"]


def test_provenance_constants_and_redaction_are_stable():
    assert provenance.EVENT_WORKFLOW_RUN == "workflow_run"
    assert provenance.EVENT_TOOL_CALL == "tool_call"
    assert provenance.REF_SOURCE_RECORD == "source_record"
    assert provenance.LINK_APPROVED_EXECUTION == "approved_execution"
    assert provenance.LINK_UPDATED == "updated"

    summary = provenance.redacted_summary(
        {
            "instructions": "never store this prompt",
            "arguments": {"secret": "value"},
            "output": {"raw": "sensitive tool output"},
            "prompt_hash": "abc123",
        }
    )
    serialized = json.dumps(summary, default=str)
    assert "never store this prompt" not in serialized
    assert "sensitive tool output" not in serialized
    assert "abc123" in serialized


def test_approval_provenance_redacts_sensitive_change_content():
    approval = core_db.create_pending_approval(
        "action_item",
        {
            "description": "Review sizing",
            "prompt": "RAW USER PROMPT",
            "password": "secret-value",
        },
        ticker="MU",
        source_type="agent",
        source_id="agent-session-1",
        action_id="create_action_item",
    )

    fetched = core_db.get_pending_approval(approval["id"])
    trace = core_db.get_provenance_trace(approval_id=approval["id"])
    approval_event = next(event for event in trace["events"] if event["event_type"] == "approval")

    assert fetched["provenance_event_id"] == approval_event["id"]
    assert approval_event["input_hash"]
    trace_json = json.dumps(trace, default=str)
    assert "RAW USER PROMPT" not in trace_json
    assert "secret-value" not in trace_json


def test_workflow_artifact_records_link_to_pending_approvals():
    from api.workflow_artifacts import persist_artifacts

    count = persist_artifacts(
        "wf-run-1",
        "MU",
        {
            "action_items": [
                {
                    "description": "Review sizing after earnings",
                    "action_type": "review",
                    "urgency": "normal",
                }
            ]
        },
    )

    approvals = core_db.get_pending_approvals()
    trace = core_db.get_provenance_trace(workflow_run_id="wf-run-1")

    assert count == 1
    assert len(approvals) == 1
    assert approvals[0]["origin_artifact_id"]
    assert approvals[0]["lineage_completeness"] == "complete"
    assert trace["workflow_artifacts"][0]["approval_id"] == approvals[0]["id"]
    assert trace["workflow_artifacts"][0]["retention_class"] == "financial_lineage_7y"
    assert trace["workflow_artifacts"][0]["redaction_policy"] == "audit_summary_v1"
    assert any(event["event_type"] == "workflow_artifact" for event in trace["events"])
    assert any(link["link_type"] == "proposed" for link in trace["links"])


def test_agent_tool_proposal_links_tool_call_to_approval():
    actor = agent_actor(admin_actor("alice"))

    payload = json.loads(
        agent_tools.execute_tool(
            "propose_action_item",
            {
                "ticker": "mu",
                "description": "Review sizing",
                "action_type": "review",
                "reason": "Risk changed",
            },
            actor=actor,
            provenance_context={
                "agent_session_id": "session-1",
                "parent_event_id": "pv:agent_turn:session-1:test",
                "call_id": "call-1",
            },
        )
    )

    approval = core_db.get_pending_approval(int(payload["approval_id"]))
    trace = core_db.get_provenance_trace(agent_session_id="session-1")
    event_types = {event["event_type"] for event in trace["events"]}

    assert payload["status"] == "pending_approval_created"
    assert payload["_meta"]["provenance_event_id"]
    assert approval["provenance_event_id"]
    assert approval["origin_provenance_event_id"]
    assert {"tool_call", "action_run", "approval"}.issubset(event_types)
    assert any(
        link["source_ref_type"] == "action_run"
        and link["target_ref_type"] == "approval"
        and link["link_type"] == "proposed"
        for link in trace["links"]
    )


def test_agent_tool_proposal_fails_closed_without_start_provenance(monkeypatch):
    actor = agent_actor(admin_actor("alice"))

    def _fail_start_event(**_kwargs):
        raise RuntimeError("provenance store unavailable")

    monkeypatch.setattr(provenance, "start_event", _fail_start_event)

    payload = json.loads(
        agent_tools.execute_tool(
            "propose_action_item",
            {
                "ticker": "mu",
                "description": "Review sizing",
                "action_type": "review",
                "reason": "Risk changed",
            },
            actor=actor,
            provenance_context={"agent_session_id": "session-closed"},
        )
    )

    assert payload["_meta"]["status"] == "failed_closed"
    assert core_db.get_pending_approvals() == []


def test_workflow_tools_receive_real_provenance_context(monkeypatch):
    from api import workflows

    seen: list[dict] = []

    def _fake_execute_tool(name, arguments, *, actor=None, provenance_context=None):
        seen.append({"name": name, "arguments": arguments, "provenance_context": provenance_context})
        return json.dumps({"ok": True})

    monkeypatch.setattr(workflows, "execute_tool", _fake_execute_tool)
    monkeypatch.setattr(workflows, "get_tool_exposure", lambda _name: SimpleNamespace(access_mode="read"))

    results = workflows._exec_parallel(
        [("get_portfolio", {}), ("query_ontology", {"filters": {"tickers": ["MU"]}})],
        actor=admin_actor("admin"),
        workflow_run_id="wf-real-context",
        workflow_name="test_workflow",
    )

    assert [name for name, _parsed, _elapsed in results] == ["get_portfolio", "query_ontology"]
    seen_by_name = {item["name"]: item for item in seen}
    assert seen_by_name["get_portfolio"]["provenance_context"]["workflow_run_id"] == "wf-real-context"
    assert seen_by_name["get_portfolio"]["provenance_context"]["parent_event_id"] == "pv:workflow_run:wf-real-context"
    assert seen_by_name["query_ontology"]["provenance_context"]["call_id"] == "test_workflow:1:query_ontology"


def test_provenance_trace_api_accepts_entity_selector(auth_client):
    provenance.start_event(
        event_id="pv:test:workflow",
        event_type=provenance.EVENT_WORKFLOW_RUN,
        event_name="test_workflow",
        actor=admin_actor("admin"),
        workflow_run_id="wf-api-1",
        summary={"status": "started"},
    )
    provenance.link_refs(
        event_id="pv:test:workflow",
        source_ref_type=provenance.REF_WORKFLOW_RUN,
        source_ref_id="wf-api-1",
        target_ref_type=provenance.REF_TOOL_CALL,
        target_ref_id="tool-api-1",
        link_type=provenance.LINK_EXECUTED,
    )

    resp = auth_client.get(
        "/api/v1/provenance/trace",
        params={"ref_type": "workflow_run", "ref_id": "wf-api-1"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["seed"]["ref_type"] == "workflow_run"
    assert any(event["id"] == "pv:test:workflow" for event in body["events"])
    assert body["timeline"]


def test_provenance_storage_has_retention_columns_and_link_type_index():
    conn = core_db._get_conn()

    source_cols = {row[1] for row in conn.execute("PRAGMA table_info(source_record_refs)").fetchall()}
    artifact_cols = {row[1] for row in conn.execute("PRAGMA table_info(workflow_artifact_records)").fetchall()}
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(provenance_links)").fetchall()}

    assert {"redaction_policy", "retention_class"}.issubset(source_cols)
    assert {"redaction_policy", "retention_class"}.issubset(artifact_cols)
    assert "idx_provenance_links_type_time" in indexes
