from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import portfolio.core_db as core_db
from api import agent_tools, provenance
from ontology.policy import Actor, PolicyDenied, admin_actor, agent_actor
from ontology.schemas.identity import provenance_event_id


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


def test_primary_start_finish_event_enforces_idempotent_terminal_lifecycle(monkeypatch):
    stored: dict[str, dict] = {}
    writes: list[tuple[str, dict]] = []

    class _FakeObjects:
        def get_object(self, object_uid):
            return stored.get(object_uid)

        def write_object(self, object_type, business_key, properties, valid_from, **_kwargs):
            assert object_type == "ProvenanceEvent"
            uid = provenance_event_id(properties.get("event_id") or business_key)
            row = {
                "object_uid": uid,
                "object_type": object_type,
                "schema_name": object_type,
                "schema_version": 1,
                "properties_json": dict(properties),
            }
            stored[uid] = row
            writes.append((business_key, dict(properties)))
            return row

    monkeypatch.setattr(provenance, "_ontology_primary_writes_enabled", lambda: True)
    monkeypatch.setattr(provenance, "OntologyObjectService", _FakeObjects)

    event = provenance.start_event(
        event_id="pv:lifecycle",
        event_type=provenance.EVENT_WORKFLOW_RUN,
        event_name="workflow",
        actor=admin_actor("admin"),
        workflow_run_id="wf-1",
    )
    again = provenance.start_event(
        event_id="pv:lifecycle",
        event_type=provenance.EVENT_WORKFLOW_RUN,
        event_name="workflow",
        actor=admin_actor("admin"),
        workflow_run_id="wf-1",
    )
    finished = provenance.finish_event("pv:lifecycle", status="succeeded")
    same_terminal = provenance.finish_event("pv:lifecycle", status="succeeded")

    assert event["status"] == "started"
    assert again["status"] == "started"
    assert finished["status"] == "succeeded"
    assert same_terminal["status"] == "succeeded"
    assert len(writes) == 2
    with pytest.raises(provenance.ProvenanceWriteError, match="already terminal"):
        provenance.finish_event("pv:lifecycle", status="failed")


def test_primary_link_refs_materializes_reference_objects_and_typed_relation(monkeypatch):
    object_writes: list[dict] = []
    relation_writes: list[dict] = []

    class _FakeObjects:
        def write_object(self, object_type, business_key, properties, valid_from, **_kwargs):
            object_writes.append(
                {
                    "object_type": object_type,
                    "business_key": business_key,
                    "properties": dict(properties),
                    "valid_from": valid_from,
                }
            )
            return {"object_uid": business_key, "object_type": object_type, "properties_json": dict(properties)}

        def write_relation(self, source_uid, target_uid, relation_type, properties, valid_from, **_kwargs):
            relation_writes.append(
                {
                    "source_uid": source_uid,
                    "target_uid": target_uid,
                    "relation_type": relation_type,
                    "properties": dict(properties),
                    "valid_from": valid_from,
                }
            )
            return {
                "relation_uid": f"{relation_type}:unit",
                "relation_type": relation_type,
                "properties_json": dict(properties),
            }

    monkeypatch.setattr(provenance, "_ontology_primary_writes_enabled", lambda: True)
    monkeypatch.setattr(provenance, "OntologyObjectService", _FakeObjects)

    row = provenance.link_refs(
        event_id="pv:link",
        source_ref_type=provenance.REF_ONTOLOGY_RUN,
        source_ref_id="run-1",
        target_ref_type=provenance.REF_TOOL_CALL,
        target_ref_id="call-1",
        link_type=provenance.LINK_EXECUTED,
        target_ref_version="tool-v1",
    )

    assert row is not None
    assert [write["object_type"] for write in object_writes] == ["OntologyRunRef", "ToolCallRef"]
    assert "ProvenanceLink" not in {write["object_type"] for write in object_writes}
    assert relation_writes[0]["relation_type"] == "provenance_executed"
    assert relation_writes[0]["source_uid"] == "ontology_run_ref:run_1"
    assert relation_writes[0]["target_uid"] == "tool_call_ref:call_1"
    assert relation_writes[0]["properties"]["event_id"] == "pv:link"
    assert relation_writes[0]["properties"]["redaction_policy"] == "audit_summary_v1"
    assert relation_writes[0]["properties"]["retention_class"] == "provenance_365d"


def test_primary_link_refs_fails_closed_for_missing_event_and_unknown_verb(monkeypatch):
    monkeypatch.setattr(provenance, "_ontology_primary_writes_enabled", lambda: True)

    with pytest.raises(provenance.ProvenanceWriteError, match="requires event_id"):
        provenance.link_refs(
            event_id=None,
            source_ref_type=provenance.REF_ONTOLOGY_RUN,
            source_ref_id="run-1",
            target_ref_type=provenance.REF_TOOL_CALL,
            target_ref_id="call-1",
            link_type=provenance.LINK_EXECUTED,
        )

    with pytest.raises(provenance.ProvenanceWriteError, match="Unsupported provenance link type"):
        provenance.link_refs(
            event_id="pv:link",
            source_ref_type=provenance.REF_ONTOLOGY_RUN,
            source_ref_id="run-1",
            target_ref_type=provenance.REF_TOOL_CALL,
            target_ref_id="call-1",
            link_type="legacy_link_type",
        )


def test_primary_source_and_workflow_artifact_refs_require_event(monkeypatch):
    monkeypatch.setattr(provenance, "_ontology_primary_writes_enabled", lambda: True)

    with pytest.raises(provenance.ProvenanceWriteError, match="adapter_run_event_id"):
        provenance.record_source_ref(
            adapter_run_event_id=None,
            source_name="unit",
            record_kind="record",
            record_key={"id": 1},
            record_value={"payload": 1},
        )

    with pytest.raises(provenance.ProvenanceWriteError, match="provenance_event_id"):
        provenance.record_workflow_artifact(
            workflow_run_id="wf-1",
            artifact_key="actions",
            artifact_index=0,
            artifact_value={"value": 1},
        )


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


def test_provenance_trace_api_is_admin_only_and_ontology_shaped(monkeypatch):
    from api.routers import provenance as provenance_router

    class _FakeObjects:
        def query_relations(self, relation_type, include_history=False, limit=100, **_kwargs):
            if relation_type != "provenance_produced":
                return []
            return [
                {
                    "relation_uid": "provenance_produced:unit",
                    "relation_type": "provenance_produced",
                    "properties_json": {
                        "event_id": "pv:trace",
                        "source_ref_type": "producer_event",
                        "source_ref_id": "pv:trace",
                        "target_ref_type": "ontology_object_version",
                        "target_ref_id": "version:1",
                        "redaction_policy": "audit_summary_v1",
                        "retention_class": "provenance_365d",
                    },
                    "_meta": {"temporal": {"valid_from": "2026-05-01T00:00:00Z"}},
                }
            ]

    class _FakeReads:
        def __init__(self):
            self.objects = _FakeObjects()

        def list_objects(self, object_type, limit=100, **_kwargs):
            if object_type == "ProvenanceEvent":
                return [
                    {
                        "id": "provenance_event:pv_trace",
                        "event_id": "pv:trace",
                        "event_type": "workflow_run",
                        "event_name": "trace",
                        "status": "succeeded",
                        "started_at": "2026-05-01T00:00:00Z",
                    }
                ]
            if object_type == "ObjectVersionRef":
                return [
                    {
                        "id": "object_version_ref:version_1",
                        "ref_id": "version:1",
                        "object_uid": "position:MU",
                        "version_id": "version:1",
                        "event_id": "pv:trace",
                    }
                ]
            return []

    monkeypatch.setattr(provenance_router, "ontology_primary_writes_enabled", lambda: True)
    monkeypatch.setattr(provenance_router, "OntologyRuntimeReadService", _FakeReads)

    with pytest.raises(PolicyDenied):
        provenance_router.get_provenance_trace(
            actor=Actor(actor_id="analyst", actor_type="user", roles=()),
            event_id="pv:trace",
        )

    body = provenance_router.get_provenance_trace(actor=admin_actor("admin"), event_id="pv:trace")

    assert body["lineage_state"] == "ontology"
    assert body["selector"] == {"event_id": "pv:trace"}
    assert body["seed"] == {"selector_type": "event_id", "selector_id": "pv:trace"}
    assert body["events"][0]["event_id"] == "pv:trace"
    assert body["references"][0]["ref_id"] == "version:1"
    assert body["relations"][0]["relation_type"] == "provenance_produced"
    assert body["relations"][0]["properties"]["relation_uid"] == "provenance_produced:unit"
    assert body["timeline"]


def test_provenance_storage_has_retention_columns_and_link_type_index():
    conn = core_db._get_conn()

    source_cols = {row[1] for row in conn.execute("PRAGMA table_info(source_record_refs)").fetchall()}
    artifact_cols = {row[1] for row in conn.execute("PRAGMA table_info(workflow_artifact_records)").fetchall()}
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(provenance_links)").fetchall()}

    assert {"redaction_policy", "retention_class"}.issubset(source_cols)
    assert {"redaction_policy", "retention_class"}.issubset(artifact_cols)
    assert "idx_provenance_links_type_time" in indexes
