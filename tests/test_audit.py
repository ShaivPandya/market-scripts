from __future__ import annotations

import json
from pathlib import Path

import pytest

import portfolio.core_db as core_db
from api.audit import emit_audit_event


@pytest.fixture
def temp_core_db(tmp_path, monkeypatch):
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    yield
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)


def _fake_job_compute(_req):
    return {"ok": True, "final_count": 1}


def test_audit_event_helpers_insert_query_parse_and_prune(temp_core_db):
    event = core_db.record_audit_event(
        action_name="test.action",
        action_category="test",
        status="succeeded",
        request_id="req-1",
        actor_id="actor-1",
        actor_type="user",
        object_refs=[{"type": "thing", "id": "T1"}],
        before_summary={"status": "old"},
        after_summary={"status": "new"},
        source_lineage={"source_type": "unit"},
        metadata={"count": 1},
    )

    assert event["event_id"]
    assert event["object_type"] == "thing"
    assert event["object_id"] == "T1"

    rows = core_db.get_audit_events(request_id="req-1")
    assert len(rows) == 1
    assert rows[0]["object_refs"] == [{"type": "thing", "id": "T1"}]
    assert rows[0]["before_summary"]["status"] == "old"
    assert rows[0]["after_summary"]["status"] == "new"
    assert rows[0]["source_lineage"]["source_type"] == "unit"
    assert rows[0]["metadata"]["count"] == 1

    core_db.record_audit_event(
        action_name="test.old",
        action_category="test",
        status="succeeded",
        occurred_at="2000-01-01T00:00:00+00:00",
        received_at="2000-01-01T00:00:00+00:00",
    )
    assert core_db.prune_audit_events(retention_days=365, batch_size=1) == 1
    assert core_db.get_audit_events(action_name="test.old") == []
    assert core_db.get_audit_events(action_name="test.action")


def test_audit_writer_redacts_sensitive_payloads(temp_core_db):
    emit_audit_event(
        "test.redaction",
        "test",
        "succeeded",
        metadata={"password": "secret-pass", "prompt": "do not store this", "safe": "ok"},
        after_summary={"content": "full research note text", "count": 2},
    )

    event = core_db.get_audit_events(action_name="test.redaction")[0]
    serialized = json.dumps(event, sort_keys=True)
    assert "secret-pass" not in serialized
    assert "do not store this" not in serialized
    assert "full research note text" not in serialized
    assert event["metadata"]["password"]["redacted"] is True
    assert event["metadata"]["prompt"]["redacted"] is True
    assert event["metadata"]["safe"] == "ok"
    assert event["after_summary"]["content"]["redacted"] is True


def test_ontology_snapshot_emits_lineage_audit_event(temp_core_db, tmp_path):
    from ontology.models import OntologyEdge, OntologyNode
    from ontology.repository import OntologyRepository

    repo = OntologyRepository(db_path=tmp_path / "ontology.sqlite3")
    repo.save_snapshot(
        run_id="run-audit",
        as_of="2026-05-03T12:00:00+00:00",
        source_status={"portfolio": {"status": "ok"}, "macro": {"status": "error"}},
        required_modules=["portfolio"],
        optional_modules=["macro"],
        component_scores={"portfolio": 0.9},
        nodes=[
            OntologyNode(
                id="position:MU",
                type="Position",
                label="MU",
                properties={"ticker": "MU", "ontology_run_id": "run-audit"},
            ),
            OntologyNode(
                id="asset:MU",
                type="Asset",
                label="MU",
                properties={"ticker": "MU", "asset": "equity", "ontology_run_id": "run-audit"},
            ),
            OntologyNode(
                id="sector:information_technology",
                type="Sector",
                label="Information Technology",
                properties={"name": "Information Technology", "ontology_run_id": "run-audit"},
            ),
        ],
        edges=[
            OntologyEdge("position:MU", "asset:MU", "references_asset", {"ontology_run_id": "run-audit"}),
            OntologyEdge(
                "asset:MU",
                "sector:information_technology",
                "belongs_to_sector",
                {"ontology_run_id": "run-audit"},
            ),
        ],
    )

    event = core_db.get_audit_events(action_name="ontology.snapshot.saved")[0]
    assert event["status"] == "succeeded"
    assert event["object_refs"] == [{"type": "ontology_run", "id": "run-audit"}]
    assert event["after_summary"]["node_count"] == 3
    assert event["after_summary"]["edge_count"] == 2
    assert event["source_lineage"]["source_status_counts"]["ok"] == 1
    assert event["source_lineage"]["source_status_counts"]["error"] == 1
    assert event["source_lineage"]["component_scores_hash"]


def test_async_job_runner_emits_lifecycle_audit_events(temp_core_db, monkeypatch):
    from api import async_job_runner, cache
    from api.job_registry import JobSpec

    cache.invalidate_all()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ASYNC_JOB_BACKEND", "local")
    monkeypatch.setattr(async_job_runner, "_enqueue_local_job", lambda _job_id: None)
    monkeypatch.setattr(
        async_job_runner,
        "get_job_spec",
        lambda job_type: JobSpec(
            job_type=job_type,
            request_model=None,
            compute_func="tests.test_audit._fake_job_compute",
            cache_key_func=None,
        ),
    )

    row, disposition = async_job_runner.enqueue_registered_job("audit_test", {}, cache_key="audit-job")
    assert disposition == "created"
    async_job_runner.perform_job(row["job_id"])

    names = {event["action_name"] for event in core_db.get_audit_events(limit=20)}
    assert "async_job.enqueued" in names
    assert "async_job.running" in names
    assert "async_job.completed" in names


def test_workflow_artifact_parse_failure_is_audited(temp_core_db):
    from api.workflow_artifacts import extract_artifacts

    assert extract_artifacts("```artifacts\n{bad json\n```", "thesis_review") == {}
    event = core_db.get_audit_events(action_name="workflow.artifacts.parse")[0]
    assert event["status"] == "failed"
    assert event["object_refs"] == [{"type": "workflow", "id": "thesis_review"}]
    assert event["error"]


def test_audit_migration_contract_and_gcp_mapping():
    migration = Path("migrations/versions/20260503_0005_audit_events.py").read_text(encoding="utf-8")
    assert 'down_revision: str | None = "20260503_0004"' in migration
    assert "audit_events" in migration
    assert "event_id" in migration
    assert "idx_audit_events_object_time" in migration
    assert "idx_audit_events_status_time" in migration

    migration_tool = Path("api/gcp_state_migration.py").read_text(encoding="utf-8")
    assert '"audit_events": [' in migration_tool
    assert 'if table == "audit_events"' in migration_tool
    assert '["event_id"]' in migration_tool
