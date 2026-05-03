from __future__ import annotations

import json

import pytest

import portfolio.core_db as core_db
from api import agent_tools
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
    assert trace["workflow_artifacts"][0]["approval_id"] == approvals[0]["id"]
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
    assert approval["provenance_event_id"]
    assert approval["origin_provenance_event_id"]
    assert {"tool_call", "action_run", "approval"}.issubset(event_types)
    assert any(
        link["source_ref_type"] == "action_run"
        and link["target_ref_type"] == "approval"
        and link["link_type"] == "proposed"
        for link in trace["links"]
    )
