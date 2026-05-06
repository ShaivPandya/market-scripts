from __future__ import annotations

import sqlite3

import pytest

from ontology.legacy_backfill import LegacyBackfillUnmappedRefs, backfill_runtime_objects


def test_legacy_runtime_backfill_dry_run_covers_contract_domains(monkeypatch, tmp_path):
    monkeypatch.setenv("TALISMAN_ENABLE_LEGACY_BACKFILL", "true")
    core_db = tmp_path / "core.db"
    portfolio_db = tmp_path / "portfolio.db"
    thesis_db = tmp_path / "thesis.db"
    snapshot_db = tmp_path / "snapshots.db"

    with sqlite3.connect(core_db) as conn:
        conn.executescript(
            """
            CREATE TABLE investment_ideas (
                id INTEGER, ticker TEXT, status TEXT, tags_json TEXT, metadata_json TEXT, created_at TEXT, updated_at TEXT
            );
            INSERT INTO investment_ideas VALUES (1, 'MU', 'watching', '[]', '{}', '2026-05-01', '2026-05-01');
            CREATE TABLE optimization_missions (
                id INTEGER, name TEXT, status TEXT, scenario_json TEXT, source_config_json TEXT, thresholds_json TEXT,
                created_at TEXT, updated_at TEXT
            );
            INSERT INTO optimization_missions VALUES (1, 'default', 'active', '{}', '{}', '{}', '2026-05-01', '2026-05-01');
            CREATE TABLE provenance_events (
                id TEXT, event_type TEXT, event_name TEXT, status TEXT, started_at TEXT, retention_class TEXT
            );
            INSERT INTO provenance_events VALUES ('pv:1', 'unit', 'test', 'succeeded', '2026-05-01', 'provenance_365d');
            CREATE TABLE provenance_links (
                id TEXT, event_id TEXT, source_ref_type TEXT, source_ref_id TEXT, target_ref_type TEXT,
                target_ref_id TEXT, link_type TEXT, created_at TEXT, source_ref_version TEXT, target_ref_version TEXT,
                lineage_root_id TEXT, metadata_json TEXT
            );
            INSERT INTO provenance_links VALUES (
                'link:1', 'pv:1', 'source_record', 'src:1', 'ontology_object_version', 'version:1', 'produced',
                '2026-05-01', NULL, 'object-version-1', 'pv:1', '{}'
            );
            INSERT INTO provenance_links VALUES (
                'link:2', 'pv:1', 'relation_version', 'relation:1', 'schema_definition', 'Position', 'schema_bound',
                '2026-05-01', 'relation-version-1', '1', 'pv:1', '{}'
            );
            INSERT INTO provenance_links VALUES (
                'link:3', 'pv:1', 'ontology_run', 'run-1', 'workflow_run', 'wf-1', 'executed',
                '2026-05-01', NULL, NULL, 'pv:1', '{}'
            );
            INSERT INTO provenance_links VALUES (
                'link:4', 'pv:1', 'agent_session', 'session-1', 'model_call', 'model-call-1', 'executed',
                '2026-05-01', NULL, NULL, 'pv:1', '{}'
            );
            INSERT INTO provenance_links VALUES (
                'link:5', 'pv:1', 'model_call', 'model-call-1', 'tool_call', 'tool-call-1', 'executed',
                '2026-05-01', NULL, NULL, 'pv:1', '{}'
            );
            INSERT INTO provenance_links VALUES (
                'link:6', 'pv:1', 'tool_call', 'tool-call-1', 'computed_snapshot_version', 'snapshot:1', 'produced',
                '2026-05-01', NULL, 'snapshot-version-1', 'pv:1', '{}'
            );
            CREATE TABLE recommendations (
                id INTEGER, action TEXT, instrument TEXT, created_at TEXT
            );
            INSERT INTO recommendations VALUES (1, 'buy', 'MU', '2026-05-01');
            CREATE TABLE source_record_refs (
                record_ref_id TEXT, adapter_run_event_id TEXT, source_name TEXT, record_kind TEXT, record_key_hash TEXT,
                record_hash TEXT, created_at TEXT
            );
            INSERT INTO source_record_refs VALUES ('src:1', 'pv:1', 'unit', 'record', 'key-hash', 'payload-hash', '2026-05-01');
            CREATE TABLE workflow_runs (
                run_id TEXT, workflow_name TEXT, ticker TEXT, status TEXT, started_at TEXT, completed_at TEXT
            );
            INSERT INTO workflow_runs VALUES ('wf-1', 'unit_workflow', 'MU', 'completed', '2026-05-01', '2026-05-01');
            """
        )
    with sqlite3.connect(portfolio_db) as conn:
        conn.executescript(
            """
            CREATE TABLE positions (ticker TEXT, asset TEXT, direction TEXT, role TEXT);
            INSERT INTO positions VALUES ('MU', 'equity', 'long', 'position');
            """
        )
    with sqlite3.connect(thesis_db) as conn:
        conn.executescript(
            """
            CREATE TABLE thesis_meta (ticker TEXT, status TEXT, created_at TEXT, updated_at TEXT);
            INSERT INTO thesis_meta VALUES ('MU', 'active', '2026-05-01', '2026-05-01');
            CREATE TABLE thesis_evaluations (
                ticker TEXT, evaluated_at TEXT, thesis_status TEXT, technical_read TEXT, fundamental_read TEXT,
                action TEXT, confidence TEXT, key_developments TEXT
            );
            INSERT INTO thesis_evaluations VALUES ('MU', '2026-05-01', 'active', 'ok', 'ok', 'hold', 'medium', '[]');
            CREATE TABLE thesis_status_history (
                id INTEGER, ticker TEXT, old_status TEXT, new_status TEXT, reason TEXT, changed_at TEXT
            );
            INSERT INTO thesis_status_history VALUES (1, 'MU', 'under_review', 'active', 'resolved', '2026-05-01');
            """
        )
    with sqlite3.connect(snapshot_db) as conn:
        conn.executescript(
            """
            CREATE TABLE computed_snapshots (
                snapshot_key TEXT, payload_json TEXT, as_of_date TEXT, fetched_at TEXT, status TEXT, error TEXT,
                version INTEGER, artifact_uri TEXT
            );
            INSERT INTO computed_snapshots VALUES ('snapshot:1', '{"value": 1}', '2026-05-01', '2026-05-01', 'ok', NULL, 1, NULL);
            """
        )

    result = backfill_runtime_objects(
        core_db_path=core_db,
        portfolio_db_path=portfolio_db,
        thesis_db_path=thesis_db,
        snapshot_db_path=snapshot_db,
        dry_run=True,
    )

    assert result["objects"]["InvestmentIdea"] == 1
    assert result["objects"]["OptimizationMission"] == 1
    assert result["objects"]["ProvenanceEvent"] == 1
    assert "ProvenanceLink" not in result["objects"]
    assert result["objects"]["ObjectVersionRef"] == 1
    assert result["objects"]["RelationVersionRef"] == 1
    assert result["objects"]["SchemaDefinitionRef"] == 1
    assert result["objects"]["OntologyRunRef"] == 1
    assert result["objects"]["AgentSessionRef"] == 1
    assert result["objects"]["ModelCallRef"] == 1
    assert result["objects"]["ToolCallRef"] == 1
    assert result["objects"]["ComputedSnapshotRef"] == 1
    assert result["objects"]["Recommendation"] == 1
    assert result["objects"]["SourceRecord"] == 1
    assert result["objects"]["WorkflowRun"] == 1
    assert result["objects"]["Position"] == 1
    assert result["objects"]["Thesis"] == 1
    assert result["objects"]["Evaluation"] == 1
    assert result["objects"]["AuditEvent"] == 1
    assert result["relations"] == 6
    assert result["computed_snapshots"] == 1


def test_legacy_runtime_backfill_reports_unmapped_provenance_refs(monkeypatch, tmp_path):
    monkeypatch.setenv("TALISMAN_ENABLE_LEGACY_BACKFILL", "true")
    core_db = tmp_path / "core.db"

    with sqlite3.connect(core_db) as conn:
        conn.executescript(
            """
            CREATE TABLE provenance_links (
                id TEXT, event_id TEXT, source_ref_type TEXT, source_ref_id TEXT, target_ref_type TEXT,
                target_ref_id TEXT, link_type TEXT, created_at TEXT
            );
            INSERT INTO provenance_links VALUES (
                'link:bad', 'pv:1', 'legacy_shadow_ref', 'shadow:1', 'ontology_object_version', 'version:1',
                'produced', '2026-05-01'
            );
            """
        )

    with pytest.raises(LegacyBackfillUnmappedRefs) as exc_info:
        backfill_runtime_objects(core_db_path=core_db, dry_run=True)

    assert exc_info.value.unmapped_refs
    assert exc_info.value.unmapped_refs[0]["id"] == "link:bad"
