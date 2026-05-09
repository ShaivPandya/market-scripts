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


def test_legacy_runtime_backfill_promotes_domain_children_and_management_quality(monkeypatch, tmp_path):
    monkeypatch.setenv("TALISMAN_ENABLE_LEGACY_BACKFILL", "true")
    core_db = tmp_path / "core.db"
    mgmt_dir = tmp_path / "investment_management_quality"
    mgmt_dir.mkdir()
    (mgmt_dir / "MU.md").write_text(
        """# MU Management Quality

## Executive Summary
- **Overall Rating**: Strong

## Management Scorecard
| Question | Rating | Evidence |
|----------|--------|----------|
| Do managers think and act like owners? | Strong | Buybacks. |

## Most Impressive Accomplishments
- **HBM ramp (2025)**: Executed well.

## Biggest Setbacks and Responses
- **Inventory cycle (2023)**: Downturn. **Response**: Mixed - Costs reset.
""",
        encoding="utf-8",
    )
    (mgmt_dir / "RAW.md").write_text("unstructured source that still needs an assessment shell", encoding="utf-8")

    with sqlite3.connect(core_db) as conn:
        conn.executescript(
            """
            CREATE TABLE investment_ideas (
                id INTEGER, ticker TEXT, status TEXT, tags_json TEXT, metadata_json TEXT, created_at TEXT, updated_at TEXT,
                latest_evaluation_id INTEGER, accepted_recommendation_id INTEGER
            );
            INSERT INTO investment_ideas VALUES (1, 'MU', 'watching', '[]', '{}', '2026-05-01', '2026-05-01', 10, 7);
            CREATE TABLE idea_evaluations (
                id INTEGER, idea_id INTEGER, ticker TEXT, evaluated_at TEXT, action TEXT, recommendation_status TEXT,
                score REAL, confidence REAL, rationale TEXT, factor_scores_json TEXT, missing_information_json TEXT,
                data_quality_json TEXT, evidence_json TEXT, disconfirming_evidence_json TEXT, portfolio_fit_json TEXT,
                recommendation_record_json TEXT, recommendation_id INTEGER, approval_id INTEGER, action_approval_id INTEGER,
                created_at TEXT
            );
            INSERT INTO idea_evaluations VALUES (
                10, 1, 'MU', '2026-05-01', 'buy', 'clear', 82, 0.7, 'Good',
                '{"management_quality":{"score":82,"status":"strong","rationale":"Good"}}',
                '[{"field":"valuation","severity":"medium","reason":"Needs model."}]',
                '{}',
                '[{"source":"overview","summary":"Evidence","url":"https://example.test/mu"}]',
                '[]',
                '{}',
                '{}',
                7, 8, 9, '2026-05-01'
            );
            CREATE TABLE idea_comparison_runs (
                id INTEGER, run_id TEXT, job_id TEXT, scope_statuses_json TEXT, summary TEXT, ranking_count INTEGER,
                raw_result_json TEXT, created_at TEXT
            );
            INSERT INTO idea_comparison_runs VALUES (4, 'cmp-1', 'job-1', '["watching"]', 'Ranked', 1, '{}', '2026-05-01');
            CREATE TABLE idea_comparison_rankings (
                id INTEGER, run_id TEXT, idea_id INTEGER, evaluation_id INTEGER, ticker TEXT, rank INTEGER, action TEXT,
                score REAL, confidence REAL, confidence_level TEXT, rationale TEXT, created_at TEXT
            );
            INSERT INTO idea_comparison_rankings VALUES (5, 'cmp-1', 1, 10, 'MU', 1, 'buy', 82, 0.7, 'high', 'Best', '2026-05-01');
            CREATE TABLE optimization_missions (
                id INTEGER, name TEXT, status TEXT, scenario_json TEXT, source_config_json TEXT, thresholds_json TEXT,
                created_at TEXT, updated_at TEXT
            );
            INSERT INTO optimization_missions VALUES (1, 'Daily Command Center', 'active', '{}', '{}', '{}', '2026-05-01', '2026-05-01');
            CREATE TABLE optimization_runs (
                run_id TEXT, mission_id INTEGER, mission_name TEXT, status TEXT, started_at TEXT, completed_at TEXT,
                input_hash TEXT, output_hash TEXT, summary_json TEXT, source_freshness_json TEXT, error TEXT
            );
            INSERT INTO optimization_runs VALUES (
                'opt-run-1', 1, 'Daily Command Center', 'succeeded', '2026-05-01', '2026-05-01',
                'in', 'out', '{}', '{"reports":{"status":"ok","checked_at":"2026-05-01"}}', NULL
            );
            CREATE TABLE optimization_action_snapshots (
                id INTEGER, run_id TEXT, mission_id INTEGER, ticker TEXT, asset TEXT, direction TEXT, action TEXT,
                conviction_band TEXT, priority_score REAL, confidence REAL, gate_status TEXT, severity TEXT,
                state_hash TEXT, evidence_json TEXT, source_links_json TEXT, created_at TEXT
            );
            INSERT INTO optimization_action_snapshots VALUES (
                3, 'opt-run-1', 1, 'MU', 'equity', 'long', 'Trim Long', 'medium', 2, 0.7,
                'pass', 'high', 'state-1', '{}', '{}', '2026-05-01'
            );
            CREATE TABLE optimization_alerts (
                id INTEGER, mission_id INTEGER, run_id TEXT, ticker TEXT, alert_type TEXT, severity TEXT, status TEXT,
                previous_snapshot_id INTEGER, current_snapshot_id INTEGER, change_summary TEXT, evidence_json TEXT,
                approval_id INTEGER, recommendation_id INTEGER, action_item_approval_id INTEGER, created_at TEXT,
                dismissed_at TEXT, dismissed_note TEXT
            );
            INSERT INTO optimization_alerts VALUES (
                2, 1, 'opt-run-1', 'MU', 'action_changed', 'high', 'open', NULL, 3, 'Changed', '{}',
                8, NULL, 9, '2026-05-01', NULL, NULL
            );
            """
        )

    result = backfill_runtime_objects(core_db_path=core_db, management_quality_dir=mgmt_dir, dry_run=True)

    assert result["objects"]["InvestmentIdea"] == 1
    assert result["objects"]["IdeaEvaluation"] == 1
    assert result["objects"]["FactorScore"] == 1
    assert result["objects"]["MissingInformationRequirement"] == 1
    assert result["objects"]["Evidence"] == 1
    assert result["objects"]["Citation"] == 1
    assert result["objects"]["IdeaComparisonRanking"] == 1
    assert result["objects"]["SourceFreshness"] == 1
    assert result["objects"]["OptimizationMission"] == 1
    assert result["objects"]["OptimizationRun"] == 1
    assert result["objects"]["OptimizationActionSnapshot"] == 1
    assert result["objects"]["OptimizationAlert"] == 1
    assert result["objects"]["ManagementQualityAssessment"] == 2
    assert result["objects"]["DocumentArtifact"] == 2
    assert result["objects"]["Issuer"] == 2
    assert result["objects"]["ManagementQualityScorecardRow"] == 1
    assert result["objects"]["ManagementQualityAccomplishment"] == 1
    assert result["objects"]["ManagementQualitySetback"] == 1
    assert result["relations"] >= 20


def test_legacy_runtime_backfill_materializes_overview_and_thesis_markdown(monkeypatch, tmp_path):
    monkeypatch.setenv("TALISMAN_ENABLE_LEGACY_BACKFILL", "true")
    core_db = tmp_path / "core.db"
    overview_dir = tmp_path / "investment_overviews"
    thesis_dir = tmp_path / "investment_theses"
    overview_dir.mkdir()
    thesis_dir.mkdir()
    core_db.touch()
    (overview_dir / "MU.md").write_text(
        """# MU Overview

## Financials
- **3-Year Avg. YoY Revenue Growth**: +12% driven by memory recovery.
- **3-Year Avg. YoY EPS Growth**: +9% through cycle.
- **Debt**: manageable ladder.
| Tranche | Rate | Maturity |
|---------|------|----------|
| 2030 notes | 5.0% | 2030 |
- **Reinvestment Costs**: elevated HBM capex.

## Sensitivity to Extrinsic Factors
| Factor | Sensitivity | Capacity |
|--------|-------------|----------|
| Memory pricing | High | Medium |

## Porter's Five Forces
- **Supplier Power - Medium**: Equipment suppliers remain important.

## Supply Outlook
- **HBM capacity**: Supply remains constrained.

## Demand Outlook
- **AI servers**: Strong demand is visible.

### Supply Chain

#### Key Suppliers
| Entity | Relationship | Exposure | Notes |
|--------|--------------|----------|-------|
| ASML | Lithography equipment | Material capex supplier | EUV tools. |

#### Key Customers
| Entity | Relationship | Exposure | Notes |
|--------|--------------|----------|-------|
| Nvidia | HBM customer | Significant | AI accelerator demand. |
""",
        encoding="utf-8",
    )
    (thesis_dir / "MU.md").write_text(
        """# MU Thesis

## Core Thesis
HBM demand should support earnings.

## Invalidation
Memory pricing weakens.
""",
        encoding="utf-8",
    )

    result = backfill_runtime_objects(
        core_db_path=core_db,
        overview_dir=overview_dir,
        thesis_content_dir=thesis_dir,
        dry_run=True,
    )

    assert result["objects"]["DocumentArtifact"] == 2
    assert result["objects"]["EquityOverview"] == 1
    assert result["objects"]["CompanyFinancialProfile"] == 1
    assert result["objects"]["ExtrinsicSensitivity"] == 1
    assert result["objects"]["IndustryForceAssessment"] == 1
    assert result["objects"]["SupplyDemandOutlook"] == 2
    assert result["objects"]["SupplyChainRelationship"] == 2
    assert result["objects"]["Thesis"] == 1
    assert result["objects"]["ThesisDocument"] == 1
    assert result["objects"]["ThesisSection"] == 3
    assert result["objects"]["Instrument"] == 2
    assert result["relations"] >= 15
