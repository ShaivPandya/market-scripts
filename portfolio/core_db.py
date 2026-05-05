"""
core_db.py -- SQLite-backed canonical process model for the investing OS.

Houses catalysts, kill conditions, workflow runs, action items, watch triggers,
research notes, and pending approvals. Follows the same connection pattern as
thesis_db.py (WAL mode, thread-safe singleton, _get_conn/_init_db).

Public API:
  Catalysts:
    create_catalyst(ticker, description, ...)        -> dict
    get_catalysts(ticker)                            -> list[dict]
    update_catalyst_status(id, status, evidence)     -> dict

  Kill Conditions:
    create_kill_condition(ticker, condition, ...)     -> dict
    get_kill_conditions(ticker)                      -> list[dict]
    update_kill_condition_status(id, status)          -> dict

  Workflow Runs:
    create_workflow_run(run_id, workflow_name, ticker) -> dict
    complete_workflow_run(run_id, synthesis, artifacts, tool_sections) -> dict
    fail_workflow_run(run_id, error)                  -> dict
    get_workflow_runs(workflow_name, ticker, limit)    -> list[dict]
    get_workflow_run(run_id)                          -> dict | None

  Action Items:
    create_action_item(ticker, action_type, ...)      -> dict
    get_action_items(status, ticker)                  -> list[dict]
    complete_action_item(id, resolution_note)          -> dict
    dismiss_action_item(id)                           -> dict

  Watch Triggers:
    create_watch_trigger(ticker, trigger_type, ...)    -> dict
    get_watch_triggers(status, ticker)                -> list[dict]
    fire_watch_trigger(id)                            -> dict
    cancel_watch_trigger(id)                          -> dict

  Research Notes:
    create_research_note(ticker, title, content, ...) -> dict
    get_research_notes(ticker, limit)                 -> list[dict]

  Pending Approvals:
    create_pending_approval(entity_type, ...)          -> dict
    get_pending_approvals(status, ticker)              -> list[dict]
    get_pending_approval(id)                           -> dict | None
    resolve_approval(id, status, resolved_note)        -> dict

  Report Runs:
    upsert_report_run(record)                          -> dict
    get_report_runs(report_type, limit)                -> list[dict]

  Thesis Claims:
    create_thesis_claim(record)                        -> dict
    get_thesis_claims(ticker, status)                  -> list[dict]
    update_thesis_claim(id, updates)                   -> dict

  Recommendations:
    create_recommendation(record)                      -> dict
    upsert_recommendation(record)                      -> dict
    get_recommendations(report_type, status, ticker)   -> list[dict]
    get_recommendation(id)                             -> dict | None
    get_latest_recommendation(report_type)             -> dict | None
    update_recommendation_approval(id, approval_id, status) -> dict
    update_recommendation_outcome(id, status, outcome) -> dict
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "core.db"
APPROVAL_APPLICATION_STATUSES = ("pending", "applying", "applied", "failed", "not_applicable")
APPROVAL_APPLICATION_LEASE = timedelta(minutes=15)
GOVERNANCE_SCHEMA_VERSION = 1
GOVERNANCE_CRITICAL_FINANCIAL = "financial_critical"
GOVERNANCE_OPERATIONAL = "operational"
GOVERNANCE_REDACTION_POLICY = "audit_summary_v1"
GOVERNANCE_FINANCIAL_RETENTION_CLASS = "financial_lineage_7y"
GOVERNANCE_OUTBOX_STATUSES = ("pending", "processing", "completed", "failed", "dead_letter")
GOVERNANCE_LINEAGE_COMPLETENESS_STATES = (
    "complete",
    "retry_pending",
    "dead_letter",
    "legacy_partial",
    "failed_closed",
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

WATCH_TRIGGER_TYPES = (
    "price_level",
    "technical",
    "fundamental",
    "fundamental_news",
    "event",
    "news_event",
    "macro",
    "custom",
)
INVESTMENT_IDEA_STATUSES = (
    "watching",
    "researching",
    "ready_for_review",
    "accepted",
    "rejected",
    "archived",
)
IDEA_RECOMMENDATION_ACTIONS = ("buy", "watch", "avoid", "do_nothing")
IDEA_RECOMMENDATION_STATUSES = ("clear", "review_required", "blocked", "error")

_CREATE_CATALYSTS = """
CREATE TABLE IF NOT EXISTS catalysts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    description TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'fundamental'
                CHECK (category IN ('fundamental', 'technical', 'macro', 'event', 'regulatory')),
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'played_out', 'failed', 'superseded')),
    target_date TEXT,
    evidence    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    created_by  TEXT NOT NULL DEFAULT 'user'
                CHECK (created_by IN ('backfill', 'user', 'agent', 'workflow'))
)
"""

_CREATE_KILL_CONDITIONS = """
CREATE TABLE IF NOT EXISTS kill_conditions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    condition   TEXT NOT NULL,
    metric      TEXT,
    threshold   TEXT,
    status      TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'triggered', 'retired')),
    triggered_at TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    created_by  TEXT NOT NULL DEFAULT 'user'
                CHECK (created_by IN ('backfill', 'user', 'agent', 'workflow'))
)
"""

_CREATE_WORKFLOW_RUNS = """
CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id        TEXT PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    ticker        TEXT,
    status        TEXT NOT NULL DEFAULT 'running'
                  CHECK (status IN ('running', 'completed', 'failed')),
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    tool_sections TEXT,
    synthesis     TEXT,
    artifacts     TEXT,
    provenance_event_id TEXT,
    lineage_completeness TEXT NOT NULL DEFAULT 'retry_pending',
    error         TEXT
)
"""

_CREATE_REPORT_RUNS = """
CREATE TABLE IF NOT EXISTS report_runs (
    report_id           TEXT PRIMARY KEY,
    report_type         TEXT NOT NULL
                        CHECK (report_type IN ('daily', 'weekly')),
    as_of               TEXT NOT NULL,
    source              TEXT NOT NULL DEFAULT 'github_actions',
    source_run_id       TEXT,
    source_url          TEXT,
    status              TEXT NOT NULL DEFAULT 'completed'
                        CHECK (status IN ('completed', 'failed')),
    report_hash         TEXT,
    input_hash          TEXT,
    summary_json        TEXT,
    artifact_paths_json TEXT,
    issue_url           TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    synced_at           TEXT NOT NULL,
    error               TEXT
)
"""

_CREATE_OPTIMIZATION_MISSIONS = """
CREATE TABLE IF NOT EXISTS optimization_missions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL UNIQUE,
    status             TEXT NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active', 'paused', 'retired')),
    schedule_label     TEXT,
    scenario_json      TEXT,
    source_config_json TEXT,
    thresholds_json    TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
)
"""

_CREATE_OPTIMIZATION_RUNS = """
CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id                TEXT PRIMARY KEY,
    mission_id            INTEGER NOT NULL,
    mission_name          TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'running'
                          CHECK (status IN ('running', 'completed', 'failed')),
    started_at            TEXT NOT NULL,
    completed_at          TEXT,
    input_hash            TEXT,
    output_hash           TEXT,
    summary_json          TEXT,
    source_freshness_json TEXT,
    error                 TEXT
)
"""

_CREATE_OPTIMIZATION_ACTION_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS optimization_action_snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    mission_id        INTEGER NOT NULL,
    ticker            TEXT,
    asset             TEXT,
    direction         TEXT,
    action            TEXT NOT NULL,
    conviction_band   TEXT,
    priority_score    REAL,
    confidence        REAL,
    gate_status       TEXT,
    severity          TEXT,
    state_hash        TEXT NOT NULL,
    evidence_json     TEXT,
    source_links_json TEXT,
    created_at        TEXT NOT NULL
)
"""

_CREATE_OPTIMIZATION_ALERTS = """
CREATE TABLE IF NOT EXISTS optimization_alerts (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id               INTEGER NOT NULL,
    run_id                   TEXT NOT NULL,
    ticker                   TEXT,
    alert_type               TEXT NOT NULL,
    severity                 TEXT NOT NULL
                             CHECK (severity IN ('low', 'normal', 'high', 'urgent')),
    status                   TEXT NOT NULL DEFAULT 'open'
                             CHECK (status IN ('open', 'dismissed', 'superseded')),
    previous_snapshot_id     INTEGER,
    current_snapshot_id      INTEGER,
    change_summary           TEXT NOT NULL,
    evidence_json            TEXT,
    approval_id              INTEGER,
    recommendation_id        INTEGER,
    action_item_approval_id  INTEGER,
    created_at               TEXT NOT NULL,
    dismissed_at             TEXT,
    dismissed_note           TEXT
)
"""

_CREATE_INVESTMENT_IDEAS = """
CREATE TABLE IF NOT EXISTS investment_ideas (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                     TEXT NOT NULL,
    company_name               TEXT,
    status                     TEXT NOT NULL DEFAULT 'watching'
                               CHECK (status IN ('watching', 'researching', 'ready_for_review', 'accepted', 'rejected', 'archived')),
    user_notes                 TEXT NOT NULL DEFAULT '',
    tags_json                  TEXT NOT NULL DEFAULT '[]',
    created_at                 TEXT NOT NULL,
    updated_at                 TEXT NOT NULL,
    source_type                TEXT NOT NULL DEFAULT 'user'
                               CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id                  TEXT,
    latest_evaluation_id       INTEGER,
    latest_job_id              TEXT,
    accepted_recommendation_id INTEGER,
    metadata_json              TEXT NOT NULL DEFAULT '{}'
)
"""

_CREATE_IDEA_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS idea_evaluations (
    id                           INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id                      INTEGER NOT NULL,
    ticker                       TEXT NOT NULL,
    job_id                       TEXT,
    evaluated_at                 TEXT NOT NULL,
    action                       TEXT NOT NULL
                                 CHECK (action IN ('buy', 'watch', 'avoid', 'do_nothing')),
    recommendation_status        TEXT NOT NULL DEFAULT 'clear'
                                 CHECK (recommendation_status IN ('clear', 'review_required', 'blocked', 'error')),
    score                        REAL,
    confidence                   REAL,
    thesis_statement             TEXT,
    rationale                    TEXT NOT NULL DEFAULT '',
    factor_scores_json           TEXT NOT NULL DEFAULT '{}',
    missing_information_json     TEXT NOT NULL DEFAULT '[]',
    data_quality_json            TEXT NOT NULL DEFAULT '{}',
    evidence_json                TEXT NOT NULL DEFAULT '[]',
    disconfirming_evidence_json  TEXT NOT NULL DEFAULT '[]',
    catalyst                     TEXT,
    invalidation                 TEXT,
    portfolio_fit_json           TEXT NOT NULL DEFAULT '{}',
    recommendation_record_json   TEXT NOT NULL DEFAULT '{}',
    recommendation_id            INTEGER,
    approval_id                  INTEGER,
    action_approval_id           INTEGER,
    accepted_at                  TEXT,
    accepted_by                  TEXT,
    raw_result_json              TEXT NOT NULL DEFAULT '{}',
    created_at                   TEXT NOT NULL
)
"""

_CREATE_ACTION_ITEMS = """
CREATE TABLE IF NOT EXISTS action_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT,
    action_type     TEXT NOT NULL
                    CHECK (action_type IN ('review', 'resize', 'research', 'exit', 'enter', 'hedge', 'other')),
    description     TEXT NOT NULL,
    urgency         TEXT NOT NULL DEFAULT 'normal'
                    CHECK (urgency IN ('low', 'normal', 'high', 'urgent')),
    status          TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open', 'completed', 'dismissed', 'superseded')),
    source_type     TEXT NOT NULL DEFAULT 'user'
                    CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id       TEXT,
    created_at      TEXT NOT NULL,
    completed_at    TEXT,
    resolution_note TEXT
)
"""

_CREATE_WATCH_TRIGGERS = """
CREATE TABLE IF NOT EXISTS watch_triggers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT,
    trigger_type TEXT NOT NULL
                 CHECK (trigger_type IN ('price_level', 'technical', 'fundamental', 'fundamental_news', 'event', 'news_event', 'macro', 'custom')),
    condition    TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active'
                 CHECK (status IN ('active', 'fired', 'expired', 'cancelled')),
    source_type  TEXT NOT NULL DEFAULT 'user'
                 CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id    TEXT,
    created_at   TEXT NOT NULL,
    fired_at     TEXT,
    expires_at   TEXT,
    definition_json TEXT,
    last_checked_at TEXT,
    last_result_json TEXT,
    last_evidence TEXT
)
"""

_CREATE_THESIS_CLAIMS = """
CREATE TABLE IF NOT EXISTS thesis_claims (
    id                           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                       TEXT NOT NULL,
    claim                        TEXT NOT NULL,
    expected_evidence            TEXT,
    disconfirming_evidence       TEXT,
    source_requirements_json     TEXT,
    cadence                      TEXT,
    confidence                   REAL,
    status                       TEXT NOT NULL DEFAULT 'active'
                                 CHECK (status IN ('active', 'supported', 'challenged', 'disconfirmed', 'retired')),
    linked_catalyst_ids_json     TEXT,
    linked_kill_condition_ids_json TEXT,
    source_type                  TEXT NOT NULL DEFAULT 'user'
                                 CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id                    TEXT,
    created_at                   TEXT NOT NULL,
    updated_at                   TEXT NOT NULL
)
"""

_CREATE_RESEARCH_NOTES = """
CREATE TABLE IF NOT EXISTS research_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    note_type   TEXT NOT NULL DEFAULT 'general'
                CHECK (note_type IN ('general', 'earnings', 'catalyst_update', 'risk_assessment', 'workflow_output')),
    source_type TEXT NOT NULL DEFAULT 'user'
                CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id   TEXT,
    created_at  TEXT NOT NULL
)
"""

_CREATE_PENDING_APPROVALS = """
CREATE TABLE IF NOT EXISTS pending_approvals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT NOT NULL,
    entity_id       INTEGER,
    ticker          TEXT,
    action_id       TEXT,
    action_schema_name TEXT,
    action_schema_version INTEGER,
    action_input_hash TEXT,
    request_schema_name TEXT,
    request_schema_version INTEGER,
    proposed_change TEXT NOT NULL,
    reason          TEXT,
    source_type     TEXT NOT NULL DEFAULT 'workflow'
                    CHECK (source_type IN ('workflow', 'agent', 'user')),
    source_id       TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    resolved_note   TEXT,
    application_status       TEXT NOT NULL DEFAULT 'pending'
                             CHECK (application_status IN ('pending', 'applying', 'applied', 'failed', 'not_applicable')),
    application_attempts     INTEGER NOT NULL DEFAULT 0,
    application_started_at   TEXT,
    application_completed_at TEXT,
    application_error        TEXT,
    provenance_event_id       TEXT,
    origin_provenance_event_id TEXT,
    origin_artifact_id        TEXT,
    lineage_completeness      TEXT NOT NULL DEFAULT 'retry_pending',
    risk_class                TEXT,
    approval_mode             TEXT,
    base_state_hash           TEXT,
    requested_by_actor_id     TEXT,
    resolved_by_actor_id      TEXT,
    approval_note_required    INTEGER NOT NULL DEFAULT 0,
    reason_code               TEXT,
    supersedes_approval_id    INTEGER
)
"""

_CREATE_ACTION_RUNS = """
CREATE TABLE IF NOT EXISTS action_runs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id            TEXT NOT NULL,
    action_schema_name   TEXT,
    action_schema_version INTEGER NOT NULL DEFAULT 1,
    request_schema_name  TEXT,
    request_schema_version INTEGER,
    actor_type           TEXT NOT NULL,
    actor_id             TEXT,
    source_type          TEXT,
    source_id            TEXT,
    approval_id          INTEGER,
    parent_action_run_id INTEGER,
    input_hash           TEXT,
    input_json           TEXT,
    output_json          TEXT,
    status               TEXT NOT NULL DEFAULT 'running'
                         CHECK (status IN ('running', 'succeeded', 'failed', 'rolled_back')),
    error                TEXT,
    started_at           TEXT NOT NULL,
    completed_at         TEXT,
    provenance_event_id  TEXT,
    lineage_completeness TEXT NOT NULL DEFAULT 'retry_pending'
)
"""

_CREATE_ACTION_EVENTS = """
CREATE TABLE IF NOT EXISTS action_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    action_run_id INTEGER NOT NULL,
    event_type    TEXT NOT NULL,
    message       TEXT,
    payload_json  TEXT,
    created_at    TEXT NOT NULL
)
"""

_CREATE_RECOMMENDATIONS = """
CREATE TABLE IF NOT EXISTS recommendations (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type                 TEXT NOT NULL
                                CHECK (report_type IN ('daily', 'weekly')),
    as_of                       TEXT NOT NULL,
    created_at                  TEXT NOT NULL,
    source_report_path          TEXT,
    source_json_path            TEXT,
    stance                      TEXT NOT NULL,
    recommendation_status       TEXT NOT NULL
                                CHECK (recommendation_status IN ('clear', 'review_required', 'blocked', 'error')),
    critical_data_quality       TEXT NOT NULL
                                CHECK (critical_data_quality IN ('ok', 'degraded', 'stale', 'failed')),
    blocked_reasons_json        TEXT,
    what_changed_json           TEXT,
    do_nothing_rationale        TEXT,
    action                      TEXT NOT NULL
                                CHECK (action IN ('buy', 'sell', 'hold', 'watch', 'avoid', 'reduce', 'exit', 'rebalance', 'hedge', 'do_nothing')),
    ticker                      TEXT,
    instrument                  TEXT NOT NULL,
    horizon                     TEXT,
    target_change               TEXT,
    rationale                   TEXT NOT NULL,
    confidence                  REAL,
    source_quality              TEXT NOT NULL
                                CHECK (source_quality IN ('ok', 'degraded', 'stale', 'failed')),
    status                      TEXT NOT NULL DEFAULT 'open'
                                CHECK (status IN ('open', 'blocked', 'error', 'superseded', 'closed')),
    evidence_json               TEXT,
    disconfirming_evidence_json TEXT,
    catalyst                    TEXT,
    invalidation                TEXT,
    expected_onset_window       TEXT,
    alternatives_json           TEXT,
    opportunity_cost_json       TEXT,
    approval_id                 INTEGER,
    approval_status             TEXT NOT NULL DEFAULT 'none'
                                CHECK (approval_status IN ('none', 'pending', 'approved', 'rejected')),
    outcome_status              TEXT NOT NULL DEFAULT 'pending'
                                CHECK (outcome_status IN ('pending', 'evaluated', 'unavailable')),
    outcome_json                TEXT,
    model                       TEXT,
    prompt_hash                 TEXT,
    input_hash                  TEXT,
    validation_status           TEXT,
    source_quality_summary_json TEXT,
    report_id                   TEXT,
    idempotency_key             TEXT,
    provenance_event_id         TEXT,
    lineage_root_id             TEXT,
    lineage_completeness        TEXT NOT NULL DEFAULT 'retry_pending',
    policy_gate_result_id       INTEGER,
    policy_gate_status          TEXT,
    policy_gate_decision        TEXT,
    policy_gate_review_required INTEGER NOT NULL DEFAULT 0,
    policy_gate_failures_json   TEXT,
    policy_gate_warnings_json   TEXT,
    policy_gate_disclosures_json TEXT,
    account_id                  TEXT,
    portfolio_id                TEXT,
    policy_id                   TEXT,
    trade_proposal_json         TEXT,
    risk_snapshot_id            TEXT,
    portfolio_risk_snapshot_id  TEXT,
    risk_quality                TEXT,
    risk_confidence             REAL,
    risk_score                  REAL,
    risk_level                  TEXT,
    risk_source_status_json     TEXT,
    risk_bindings_json          TEXT
)
"""

_CREATE_RECOMMENDATION_RISK_BINDINGS = """
CREATE TABLE IF NOT EXISTS recommendation_risk_bindings (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id           INTEGER NOT NULL,
    created_at                  TEXT NOT NULL,
    ticker                      TEXT,
    risk_snapshot_id            TEXT,
    portfolio_risk_snapshot_id  TEXT,
    risk_quality                TEXT,
    risk_confidence             REAL,
    risk_score                  REAL,
    risk_level                  TEXT,
    source_status_json          TEXT,
    binding_json                TEXT NOT NULL
)
"""

_CREATE_POLICY_GATE_RESULTS = """
CREATE TABLE IF NOT EXISTS policy_gate_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at              TEXT NOT NULL,
    decision                TEXT NOT NULL
                            CHECK (decision IN ('pass', 'warn', 'review_required', 'blocked', 'error')),
    review_required         INTEGER NOT NULL DEFAULT 0,
    override_acknowledged   INTEGER NOT NULL DEFAULT 0,
    account_id              TEXT,
    portfolio_id            TEXT,
    policy_id               TEXT,
    mandate_id              TEXT,
    action_id               TEXT,
    source_type             TEXT,
    source_id               TEXT,
    target_type             TEXT,
    target_id               TEXT,
    payload_hash            TEXT,
    provenance_event_id     TEXT,
    lineage_root_id         TEXT,
    lineage_completeness    TEXT NOT NULL DEFAULT 'retry_pending',
    result_json             TEXT NOT NULL
)
"""

_CREATE_AUDIT_EVENTS = """
CREATE TABLE IF NOT EXISTS audit_events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id             TEXT NOT NULL UNIQUE,
    occurred_at          TEXT NOT NULL,
    received_at          TEXT NOT NULL,
    request_id           TEXT,
    actor_id             TEXT,
    actor_type           TEXT NOT NULL DEFAULT 'system',
    parent_actor_id      TEXT,
    action_name          TEXT NOT NULL,
    action_category      TEXT NOT NULL,
    status               TEXT NOT NULL,
    object_type          TEXT,
    object_id            TEXT,
    object_refs_json     TEXT NOT NULL DEFAULT '[]',
    before_summary_json  TEXT,
    after_summary_json   TEXT,
    source_lineage_json  TEXT,
    metadata_json        TEXT,
    error                TEXT,
    schema_version       INTEGER NOT NULL DEFAULT 1,
    criticality          TEXT NOT NULL DEFAULT 'operational',
    lineage_root_id      TEXT,
    idempotency_key      TEXT,
    producer_name        TEXT,
    producer_version     TEXT,
    redaction_policy     TEXT NOT NULL DEFAULT 'audit_summary_v1',
    retention_class      TEXT NOT NULL DEFAULT 'audit_365d'
)
"""

_CREATE_PROVENANCE_EVENTS = """
CREATE TABLE IF NOT EXISTS provenance_events (
    id                   TEXT PRIMARY KEY,
    event_type           TEXT NOT NULL,
    event_name           TEXT NOT NULL,
    status               TEXT NOT NULL,
    started_at           TEXT NOT NULL,
    completed_at         TEXT,
    actor_type           TEXT,
    actor_id             TEXT,
    parent_actor_id      TEXT,
    request_id           TEXT,
    parent_event_id      TEXT,
    workflow_run_id      TEXT,
    ontology_run_id      TEXT,
    agent_session_id     TEXT,
    action_run_id        INTEGER,
    approval_id          INTEGER,
    audit_event_id       TEXT,
    input_hash           TEXT,
    output_hash          TEXT,
    summary_json         TEXT,
    metadata_json        TEXT,
    schema_version       INTEGER NOT NULL DEFAULT 1,
    criticality          TEXT NOT NULL DEFAULT 'operational',
    lineage_root_id      TEXT,
    idempotency_key      TEXT,
    producer_name        TEXT,
    producer_version     TEXT,
    redaction_policy     TEXT NOT NULL DEFAULT 'audit_summary_v1',
    retention_class      TEXT NOT NULL DEFAULT 'provenance_365d',
    error                TEXT
)
"""

_CREATE_PROVENANCE_LINKS = """
CREATE TABLE IF NOT EXISTS provenance_links (
    id                   TEXT PRIMARY KEY,
    event_id             TEXT NOT NULL,
    source_ref_type      TEXT NOT NULL,
    source_ref_id        TEXT NOT NULL,
    source_ref_version   TEXT,
    target_ref_type      TEXT NOT NULL,
    target_ref_id        TEXT NOT NULL,
    target_ref_version   TEXT,
    link_type            TEXT NOT NULL,
    metadata_json        TEXT,
    lineage_root_id      TEXT,
    created_at           TEXT NOT NULL
)
"""

_CREATE_SOURCE_RECORD_REFS = """
CREATE TABLE IF NOT EXISTS source_record_refs (
    record_ref_id        TEXT PRIMARY KEY,
    adapter_run_event_id TEXT NOT NULL,
    source_name          TEXT NOT NULL,
    record_kind          TEXT NOT NULL,
    record_key_hash      TEXT NOT NULL,
    record_hash          TEXT NOT NULL,
    as_of                TEXT,
    summary_json         TEXT,
    redaction_policy     TEXT NOT NULL DEFAULT 'audit_summary_v1',
    retention_class      TEXT NOT NULL DEFAULT 'source_ref_90d',
    created_at           TEXT NOT NULL
)
"""

_CREATE_GOVERNANCE_OUTBOX = """
CREATE TABLE IF NOT EXISTS governance_outbox (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key      TEXT NOT NULL UNIQUE,
    event_bundle_json    TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'dead_letter')),
    attempt_count        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at      TEXT NOT NULL,
    locked_at            TEXT,
    last_error           TEXT,
    dead_lettered_at     TEXT,
    lineage_root_id      TEXT,
    retention_class      TEXT NOT NULL DEFAULT 'financial_lineage_7y',
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
)
"""

_CREATE_WORKFLOW_ARTIFACT_RECORDS = """
CREATE TABLE IF NOT EXISTS workflow_artifact_records (
    artifact_id          TEXT PRIMARY KEY,
    workflow_run_id      TEXT NOT NULL,
    artifact_key         TEXT NOT NULL,
    artifact_index       INTEGER NOT NULL DEFAULT 0,
    artifact_hash        TEXT NOT NULL,
    summary_json         TEXT,
    approval_id          INTEGER,
    provenance_event_id  TEXT,
    redaction_policy     TEXT NOT NULL DEFAULT 'audit_summary_v1',
    retention_class      TEXT NOT NULL DEFAULT 'workflow_artifact_365d',
    created_at           TEXT NOT NULL
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_catalysts_ticker ON catalysts(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_kill_conditions_ticker ON kill_conditions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_name ON workflow_runs(workflow_name)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_ticker ON workflow_runs(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_started ON workflow_runs(started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_report_runs_type_asof ON report_runs(report_type, as_of DESC)",
    "CREATE INDEX IF NOT EXISTS idx_report_runs_source ON report_runs(source_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_optimization_missions_status ON optimization_missions(status)",
    "CREATE INDEX IF NOT EXISTS idx_optimization_runs_mission_started ON optimization_runs(mission_id, started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_optimization_runs_status ON optimization_runs(status)",
    "CREATE INDEX IF NOT EXISTS idx_optimization_snapshots_run ON optimization_action_snapshots(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_optimization_snapshots_mission_ticker ON optimization_action_snapshots(mission_id, ticker)",
    "CREATE INDEX IF NOT EXISTS idx_optimization_snapshots_hash ON optimization_action_snapshots(state_hash)",
    "CREATE INDEX IF NOT EXISTS idx_optimization_alerts_status ON optimization_alerts(status, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_optimization_alerts_mission ON optimization_alerts(mission_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_optimization_alerts_ticker ON optimization_alerts(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_investment_ideas_ticker ON investment_ideas(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_investment_ideas_status ON investment_ideas(status)",
    "CREATE INDEX IF NOT EXISTS idx_investment_ideas_latest_eval ON investment_ideas(latest_evaluation_id)",
    "CREATE INDEX IF NOT EXISTS idx_idea_evaluations_idea_created ON idea_evaluations(idea_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_idea_evaluations_ticker_created ON idea_evaluations(ticker, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_idea_evaluations_job ON idea_evaluations(job_id)",
    "CREATE INDEX IF NOT EXISTS idx_action_items_status ON action_items(status)",
    "CREATE INDEX IF NOT EXISTS idx_action_items_ticker ON action_items(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_watch_triggers_status ON watch_triggers(status)",
    "CREATE INDEX IF NOT EXISTS idx_watch_triggers_ticker ON watch_triggers(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_thesis_claims_ticker ON thesis_claims(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_thesis_claims_status ON thesis_claims(status)",
    "CREATE INDEX IF NOT EXISTS idx_research_notes_ticker ON research_notes(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_status ON pending_approvals(status)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_ticker ON pending_approvals(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_application_status ON pending_approvals(application_status)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_action_id ON pending_approvals(action_id)",
    "CREATE INDEX IF NOT EXISTS idx_pending_approvals_lineage_completeness ON pending_approvals(lineage_completeness)",
    "CREATE INDEX IF NOT EXISTS idx_action_runs_action_id ON action_runs(action_id)",
    "CREATE INDEX IF NOT EXISTS idx_action_runs_status ON action_runs(status)",
    "CREATE INDEX IF NOT EXISTS idx_action_runs_approval ON action_runs(approval_id)",
    "CREATE INDEX IF NOT EXISTS idx_action_runs_lineage_completeness ON action_runs(lineage_completeness)",
    "CREATE INDEX IF NOT EXISTS idx_action_events_run ON action_events(action_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_lineage_completeness ON workflow_runs(lineage_completeness)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_report ON recommendations(report_type, as_of DESC)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_report_id ON recommendations(report_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_lineage_root ON recommendations(lineage_root_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_lineage_completeness ON recommendations(lineage_completeness)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_recommendations_idempotency ON recommendations(idempotency_key)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_ticker ON recommendations(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_approval ON recommendations(approval_status)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_outcome ON recommendations(outcome_status)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_policy_gate ON recommendations(policy_gate_decision)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_risk_snapshot ON recommendations(risk_snapshot_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_portfolio_risk_snapshot ON recommendations(portfolio_risk_snapshot_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendation_risk_bindings_recommendation ON recommendation_risk_bindings(recommendation_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendation_risk_bindings_position ON recommendation_risk_bindings(risk_snapshot_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendation_risk_bindings_portfolio ON recommendation_risk_bindings(portfolio_risk_snapshot_id)",
    "CREATE INDEX IF NOT EXISTS idx_policy_gate_results_decision ON policy_gate_results(decision, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_policy_gate_results_target ON policy_gate_results(target_type, target_id)",
    "CREATE INDEX IF NOT EXISTS idx_policy_gate_results_action ON policy_gate_results(action_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_policy_gate_results_lineage_root ON policy_gate_results(lineage_root_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_occurred_at ON audit_events(occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_request ON audit_events(request_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_actor_time ON audit_events(actor_id, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_action_time ON audit_events(action_name, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_object_time ON audit_events(object_type, object_id, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_status_time ON audit_events(status, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_lineage_root ON audit_events(lineage_root_id, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_criticality_time ON audit_events(criticality, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_idempotency ON audit_events(idempotency_key)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_events_type_time ON provenance_events(event_type, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_events_workflow ON provenance_events(workflow_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_events_ontology ON provenance_events(ontology_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_events_agent_session ON provenance_events(agent_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_events_action_run ON provenance_events(action_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_events_approval ON provenance_events(approval_id)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_events_parent ON provenance_events(parent_event_id)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_events_lineage_root ON provenance_events(lineage_root_id, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_events_idempotency ON provenance_events(idempotency_key)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_links_event ON provenance_links(event_id)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_links_source ON provenance_links(source_ref_type, source_ref_id)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_links_target ON provenance_links(target_ref_type, target_ref_id)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_links_type_time ON provenance_links(link_type, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_provenance_links_lineage_root ON provenance_links(lineage_root_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_source_record_refs_adapter ON source_record_refs(adapter_run_event_id)",
    "CREATE INDEX IF NOT EXISTS idx_source_record_refs_source ON source_record_refs(source_name, record_kind)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_artifact_records_run ON workflow_artifact_records(workflow_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_workflow_artifact_records_approval ON workflow_artifact_records(approval_id)",
    "CREATE INDEX IF NOT EXISTS idx_governance_outbox_status_next ON governance_outbox(status, next_attempt_at)",
    "CREATE INDEX IF NOT EXISTS idx_governance_outbox_lineage_root ON governance_outbox(lineage_root_id)",
    "CREATE INDEX IF NOT EXISTS idx_governance_outbox_dead_letter ON governance_outbox(dead_lettered_at)",
]

# ---------------------------------------------------------------------------
# Connection management (same pattern as thesis_db.py)
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_conn: sqlite3.Connection | PostgresCompatConnection | None = None


def _get_conn() -> sqlite3.Connection | PostgresCompatConnection:
    global _conn
    if _conn is not None:
        try:
            _conn.execute("SELECT 1")
        except Exception:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
    if _conn is None:
        with _lock:
            if _conn is None:
                if use_postgres_state():
                    _conn = PostgresCompatConnection(
                        identity_tables={
                            "catalysts",
                            "kill_conditions",
                            "thesis_claims",
                            "action_items",
                            "watch_triggers",
                            "research_notes",
                            "pending_approvals",
                            "action_runs",
                            "action_events",
                            "recommendations",
                            "policy_gate_results",
                            "audit_events",
                            "governance_outbox",
                            "optimization_missions",
                            "optimization_action_snapshots",
                            "optimization_alerts",
                            "investment_ideas",
                            "idea_evaluations",
                        }
                    )
                else:
                    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                    _conn.execute("PRAGMA journal_mode=WAL")
                    _conn.row_factory = sqlite3.Row
                    _conn.execute("PRAGMA foreign_keys = ON")
                    _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection) -> None:
    for stmt in [
        _CREATE_CATALYSTS,
        _CREATE_KILL_CONDITIONS,
        _CREATE_WORKFLOW_RUNS,
        _CREATE_REPORT_RUNS,
        _CREATE_OPTIMIZATION_MISSIONS,
        _CREATE_OPTIMIZATION_RUNS,
        _CREATE_OPTIMIZATION_ACTION_SNAPSHOTS,
        _CREATE_OPTIMIZATION_ALERTS,
        _CREATE_INVESTMENT_IDEAS,
        _CREATE_IDEA_EVALUATIONS,
        _CREATE_ACTION_ITEMS,
        _CREATE_WATCH_TRIGGERS,
        _CREATE_THESIS_CLAIMS,
        _CREATE_RESEARCH_NOTES,
        _CREATE_PENDING_APPROVALS,
        _CREATE_ACTION_RUNS,
        _CREATE_ACTION_EVENTS,
        _CREATE_RECOMMENDATIONS,
        _CREATE_RECOMMENDATION_RISK_BINDINGS,
        _CREATE_POLICY_GATE_RESULTS,
        _CREATE_AUDIT_EVENTS,
        _CREATE_PROVENANCE_EVENTS,
        _CREATE_PROVENANCE_LINKS,
        _CREATE_SOURCE_RECORD_REFS,
        _CREATE_WORKFLOW_ARTIFACT_RECORDS,
        _CREATE_GOVERNANCE_OUTBOX,
    ]:
        conn.execute(stmt)
    _ensure_sqlite_columns(conn)
    for idx in _INDEXES:
        conn.execute(idx)
    conn.commit()


def _ensure_sqlite_columns(conn: sqlite3.Connection) -> None:
    """Apply small additive schema upgrades for local SQLite databases."""

    def _columns(table: str) -> set[str]:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _add_missing(table: str, columns: dict[str, str]) -> None:
        existing = _columns(table)
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    _add_missing(
        "watch_triggers",
        {
            "definition_json": "TEXT",
            "last_checked_at": "TEXT",
            "last_result_json": "TEXT",
            "last_evidence": "TEXT",
        },
    )
    _add_missing(
        "recommendations",
        {
            "report_id": "TEXT",
            "idempotency_key": "TEXT",
            "provenance_event_id": "TEXT",
            "lineage_root_id": "TEXT",
            "lineage_completeness": "TEXT NOT NULL DEFAULT 'legacy_partial'",
            "policy_gate_result_id": "INTEGER",
            "policy_gate_status": "TEXT",
            "policy_gate_decision": "TEXT",
            "policy_gate_review_required": "INTEGER NOT NULL DEFAULT 0",
            "policy_gate_failures_json": "TEXT",
            "policy_gate_warnings_json": "TEXT",
            "policy_gate_disclosures_json": "TEXT",
            "account_id": "TEXT",
            "portfolio_id": "TEXT",
            "policy_id": "TEXT",
            "trade_proposal_json": "TEXT",
            "risk_snapshot_id": "TEXT",
            "portfolio_risk_snapshot_id": "TEXT",
            "risk_quality": "TEXT",
            "risk_confidence": "REAL",
            "risk_score": "REAL",
            "risk_level": "TEXT",
            "risk_source_status_json": "TEXT",
            "risk_bindings_json": "TEXT",
        },
    )
    _add_missing(
        "policy_gate_results",
        {
            "provenance_event_id": "TEXT",
            "lineage_root_id": "TEXT",
            "lineage_completeness": "TEXT NOT NULL DEFAULT 'legacy_partial'",
        },
    )
    _add_missing(
        "audit_events",
        {
            "schema_version": "INTEGER NOT NULL DEFAULT 1",
            "criticality": "TEXT NOT NULL DEFAULT 'operational'",
            "lineage_root_id": "TEXT",
            "idempotency_key": "TEXT",
            "producer_name": "TEXT",
            "producer_version": "TEXT",
            "redaction_policy": "TEXT NOT NULL DEFAULT 'audit_summary_v1'",
            "retention_class": "TEXT NOT NULL DEFAULT 'audit_365d'",
        },
    )
    _add_missing(
        "provenance_events",
        {
            "schema_version": "INTEGER NOT NULL DEFAULT 1",
            "criticality": "TEXT NOT NULL DEFAULT 'operational'",
            "lineage_root_id": "TEXT",
            "idempotency_key": "TEXT",
            "producer_name": "TEXT",
            "producer_version": "TEXT",
        },
    )
    _add_missing(
        "provenance_links",
        {
            "lineage_root_id": "TEXT",
        },
    )
    _add_missing(
        "pending_approvals",
        {
            "application_status": (
                "TEXT NOT NULL DEFAULT 'pending' "
                "CHECK (application_status IN ('pending', 'applying', 'applied', 'failed', 'not_applicable'))"
            ),
            "application_attempts": "INTEGER NOT NULL DEFAULT 0",
            "application_started_at": "TEXT",
            "application_completed_at": "TEXT",
            "application_error": "TEXT",
            "action_id": "TEXT",
            "action_schema_name": "TEXT",
            "action_schema_version": "INTEGER",
            "action_input_hash": "TEXT",
            "request_schema_name": "TEXT",
            "request_schema_version": "INTEGER",
            "provenance_event_id": "TEXT",
            "origin_provenance_event_id": "TEXT",
            "origin_artifact_id": "TEXT",
            "lineage_completeness": "TEXT NOT NULL DEFAULT 'legacy_partial'",
            "risk_class": "TEXT",
            "approval_mode": "TEXT",
            "base_state_hash": "TEXT",
            "requested_by_actor_id": "TEXT",
            "resolved_by_actor_id": "TEXT",
            "approval_note_required": "INTEGER NOT NULL DEFAULT 0",
            "reason_code": "TEXT",
            "supersedes_approval_id": "INTEGER",
        },
    )
    _add_missing(
        "action_runs",
        {
            "action_schema_name": "TEXT",
            "request_schema_name": "TEXT",
            "request_schema_version": "INTEGER",
            "provenance_event_id": "TEXT",
            "lineage_completeness": "TEXT NOT NULL DEFAULT 'legacy_partial'",
        },
    )
    _add_missing(
        "workflow_runs",
        {
            "provenance_event_id": "TEXT",
            "lineage_completeness": "TEXT NOT NULL DEFAULT 'legacy_partial'",
        },
    )
    _add_missing(
        "source_record_refs",
        {
            "redaction_policy": "TEXT NOT NULL DEFAULT 'audit_summary_v1'",
            "retention_class": "TEXT NOT NULL DEFAULT 'source_ref_90d'",
        },
    )
    _add_missing(
        "workflow_artifact_records",
        {
            "redaction_policy": "TEXT NOT NULL DEFAULT 'audit_summary_v1'",
            "retention_class": "TEXT NOT NULL DEFAULT 'workflow_artifact_365d'",
        },
    )
    conn.execute(
        "UPDATE pending_approvals "
        "SET application_status = 'applied', "
        "application_completed_at = COALESCE(application_completed_at, resolved_at, created_at) "
        "WHERE status = 'approved' AND application_status = 'pending'"
    )
    conn.execute(
        "UPDATE pending_approvals "
        "SET application_status = 'not_applicable', "
        "application_completed_at = COALESCE(application_completed_at, resolved_at, created_at) "
        "WHERE status IN ('rejected', 'expired') AND application_status = 'pending'"
    )
    _ensure_sqlite_watch_trigger_types(conn)
    _ensure_sqlite_recommendation_status_types(conn)


def _ensure_sqlite_watch_trigger_types(conn: sqlite3.Connection) -> None:
    """Rebuild legacy SQLite watch_triggers tables whose CHECK enum is stale."""

    row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'watch_triggers'").fetchone()
    create_sql = str(row[0] if row else "")
    if not create_sql or all(trigger_type in create_sql for trigger_type in ("news_event", "fundamental_news")):
        return
    if "CHECK" not in create_sql or "trigger_type" not in create_sql:
        return

    legacy_table = "watch_triggers_legacy_trigger_type_upgrade"
    conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
    conn.execute(f"ALTER TABLE watch_triggers RENAME TO {legacy_table}")
    conn.execute(_CREATE_WATCH_TRIGGERS)

    legacy_cols = {str(col[1]) for col in conn.execute(f"PRAGMA table_info({legacy_table})").fetchall()}
    target_cols = {str(col[1]) for col in conn.execute("PRAGMA table_info(watch_triggers)").fetchall()}
    copy_cols = [
        col
        for col in (
            "id",
            "ticker",
            "trigger_type",
            "condition",
            "status",
            "source_type",
            "source_id",
            "created_at",
            "fired_at",
            "expires_at",
            "definition_json",
            "last_checked_at",
            "last_result_json",
            "last_evidence",
        )
        if col in legacy_cols and col in target_cols
    ]
    cols_sql = ", ".join(copy_cols)
    conn.execute(f"INSERT INTO watch_triggers ({cols_sql}) SELECT {cols_sql} FROM {legacy_table}")
    conn.execute(f"DROP TABLE {legacy_table}")


def _ensure_sqlite_recommendation_status_types(conn: sqlite3.Connection) -> None:
    """Rebuild legacy recommendations tables whose status CHECK enum is stale."""

    row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'recommendations'").fetchone()
    create_sql = str(row[0] if row else "")
    if not create_sql or "review_required" in create_sql:
        return
    if "CHECK" not in create_sql or "recommendation_status" not in create_sql:
        return

    legacy_table = "recommendations_legacy_status_upgrade"
    conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
    conn.execute(f"ALTER TABLE recommendations RENAME TO {legacy_table}")
    conn.execute(_CREATE_RECOMMENDATIONS)

    legacy_cols = {str(col[1]) for col in conn.execute(f"PRAGMA table_info({legacy_table})").fetchall()}
    target_cols = {str(col[1]) for col in conn.execute("PRAGMA table_info(recommendations)").fetchall()}
    copy_cols = [col for col in target_cols if col in legacy_cols]
    cols_sql = ", ".join(copy_cols)
    if cols_sql:
        conn.execute(f"INSERT INTO recommendations ({cols_sql}) SELECT {cols_sql} FROM {legacy_table}")
    conn.execute(f"DROP TABLE {legacy_table}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(row: Any | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return cast(dict[str, Any], dict(row))


def _require_row_dict(row: Any | None) -> dict[str, Any]:
    if row is None:
        raise RuntimeError("Expected database row.")
    return cast(dict[str, Any], dict(row))


def _rows_to_list(rows: Iterable[Any]) -> list[dict[str, Any]]:
    return [_require_row_dict(r) for r in rows]


def _parse_json_field(d: dict, field: str) -> dict:
    """Parse a JSON string field in a dict, returning the dict with the field parsed."""
    val = d.get(field)
    if isinstance(val, str):
        try:
            d[field] = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def _json_hash(value: Any) -> str:
    import hashlib

    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def create_action_run(
    *,
    action_id: str,
    action_schema_version: int,
    action_schema_name: str | None = None,
    actor_type: str,
    actor_id: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    approval_id: int | None = None,
    parent_action_run_id: int | None = None,
    input_hash: str | None = None,
    input_payload: Any | None = None,
    request_schema_name: str | None = None,
    request_schema_version: int | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    input_json = json.dumps(input_payload, default=str) if input_payload is not None else None
    with _lock:
        cur = conn.execute(
            "INSERT INTO action_runs (action_id, action_schema_name, action_schema_version, request_schema_name, "
            "request_schema_version, actor_type, actor_id, source_type, source_id, approval_id, parent_action_run_id, "
            "input_hash, input_json, status, started_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                action_id,
                action_schema_name,
                action_schema_version,
                request_schema_name,
                request_schema_version,
                actor_type,
                actor_id,
                source_type,
                source_id,
                approval_id,
                parent_action_run_id,
                input_hash,
                input_json,
                "running",
                now,
            ),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "action_id": action_id,
        "action_schema_name": action_schema_name,
        "action_schema_version": action_schema_version,
        "request_schema_name": request_schema_name,
        "request_schema_version": request_schema_version,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "source_type": source_type,
        "source_id": source_id,
        "approval_id": approval_id,
        "parent_action_run_id": parent_action_run_id,
        "input_hash": input_hash,
        "input_json": input_json,
        "status": "running",
        "started_at": now,
        "completed_at": None,
        "error": None,
        "provenance_event_id": None,
        "lineage_completeness": "retry_pending",
    }


def record_action_event(
    action_run_id: int,
    event_type: str,
    *,
    message: str | None = None,
    payload: Any | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    payload_json = json.dumps(payload, default=str) if payload is not None else None
    with _lock:
        cur = conn.execute(
            "INSERT INTO action_events (action_run_id, event_type, message, payload_json, created_at) VALUES (?,?,?,?,?)",
            (action_run_id, event_type, message, payload_json, now),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "action_run_id": action_run_id,
        "event_type": event_type,
        "message": message,
        "payload": payload,
        "created_at": now,
    }


def complete_action_run(
    action_run_id: int,
    *,
    status: str,
    output_payload: Any | None = None,
    error: str | None = None,
) -> dict:
    if status not in {"succeeded", "failed", "rolled_back"}:
        raise ValueError(f"Invalid action run status: {status}")
    conn = _get_conn()
    now = _now()
    output_json = json.dumps(output_payload, default=str) if output_payload is not None else None
    with _lock:
        conn.execute(
            "UPDATE action_runs SET status = ?, output_json = ?, error = ?, completed_at = ? WHERE id = ?",
            (status, output_json, error, now, action_run_id),
        )
        updated = conn.execute("SELECT * FROM action_runs WHERE id = ?", (action_run_id,)).fetchone()
        conn.commit()
    return _require_row_dict(updated)


def get_action_run(action_run_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM action_runs WHERE id = ?", (action_run_id,)).fetchone()
    return _row_to_dict(row)


def get_action_runs(action_id: str | None = None, approval_id: int | None = None) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if action_id:
        clauses.append("action_id = ?")
        params.append(action_id)
    if approval_id is not None:
        clauses.append("approval_id = ?")
        params.append(approval_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(f"SELECT * FROM action_runs{where} ORDER BY id", params).fetchall()
    return _rows_to_list(rows)


def get_action_events(action_run_id: int) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM action_events WHERE action_run_id = ? ORDER BY id",
            (action_run_id,),
        ).fetchall()
    out = _rows_to_list(rows)
    for row in out:
        _parse_json_field(row, "payload_json")
        row["payload"] = row.get("payload_json")
    return out


# ---------------------------------------------------------------------------
# Audit Events
# ---------------------------------------------------------------------------

_AUDIT_JSON_FIELDS = (
    "object_refs_json",
    "before_summary_json",
    "after_summary_json",
    "source_lineage_json",
    "metadata_json",
)


def _json_or_none(value: Any | None) -> str | None:
    return json.dumps(value, default=str) if value is not None else None


def _parse_audit_event_row(row: Any) -> dict:
    d = _require_row_dict(row)
    for field in _AUDIT_JSON_FIELDS:
        _parse_json_field(d, field)
    d["object_refs"] = d.get("object_refs_json") if isinstance(d.get("object_refs_json"), list) else []
    d["before_summary"] = d.get("before_summary_json")
    d["after_summary"] = d.get("after_summary_json")
    d["source_lineage"] = d.get("source_lineage_json")
    d["metadata"] = d.get("metadata_json")
    return d


def _record_audit_event_conn(
    conn: sqlite3.Connection | PostgresCompatConnection,
    *,
    action_name: str,
    action_category: str,
    status: str,
    event_id: str | None = None,
    occurred_at: str | None = None,
    received_at: str | None = None,
    request_id: str | None = None,
    actor_id: str | None = None,
    actor_type: str = "system",
    parent_actor_id: str | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    object_refs: list[dict[str, Any]] | None = None,
    before_summary: Any | None = None,
    after_summary: Any | None = None,
    source_lineage: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
    schema_version: int = GOVERNANCE_SCHEMA_VERSION,
    criticality: str = GOVERNANCE_OPERATIONAL,
    lineage_root_id: str | None = None,
    idempotency_key: str | None = None,
    producer_name: str | None = None,
    producer_version: str | None = None,
    redaction_policy: str = GOVERNANCE_REDACTION_POLICY,
    retention_class: str = "audit_365d",
) -> dict:
    now = _now()
    refs = object_refs or []
    first_ref = refs[0] if refs and isinstance(refs[0], dict) else {}
    resolved_object_type = object_type or first_ref.get("type") or first_ref.get("object_type")
    resolved_object_id = object_id or first_ref.get("id") or first_ref.get("object_id")
    params = (
        event_id or uuid.uuid4().hex,
        occurred_at or now,
        received_at or now,
        request_id,
        actor_id,
        actor_type or "system",
        parent_actor_id,
        action_name,
        action_category,
        status,
        str(resolved_object_type) if resolved_object_type is not None else None,
        str(resolved_object_id) if resolved_object_id is not None else None,
        json.dumps(refs, default=str),
        _json_or_none(before_summary),
        _json_or_none(after_summary),
        _json_or_none(source_lineage),
        _json_or_none(metadata),
        str(error)[:1000] if error is not None else None,
        int(schema_version or GOVERNANCE_SCHEMA_VERSION),
        criticality or GOVERNANCE_OPERATIONAL,
        lineage_root_id,
        idempotency_key,
        producer_name,
        producer_version,
        redaction_policy or GOVERNANCE_REDACTION_POLICY,
        retention_class or "audit_365d",
    )
    cur = conn.execute(
        """
        INSERT INTO audit_events (
            event_id,
            occurred_at,
            received_at,
            request_id,
            actor_id,
            actor_type,
            parent_actor_id,
            action_name,
            action_category,
            status,
            object_type,
            object_id,
            object_refs_json,
            before_summary_json,
            after_summary_json,
            source_lineage_json,
            metadata_json,
            error,
            schema_version,
            criticality,
            lineage_root_id,
            idempotency_key,
            producer_name,
            producer_version,
            redaction_policy,
            retention_class
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        params,
    )
    row = conn.execute("SELECT * FROM audit_events WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _parse_audit_event_row(row)


def record_audit_event(
    *,
    action_name: str,
    action_category: str,
    status: str,
    event_id: str | None = None,
    occurred_at: str | None = None,
    received_at: str | None = None,
    request_id: str | None = None,
    actor_id: str | None = None,
    actor_type: str = "system",
    parent_actor_id: str | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    object_refs: list[dict[str, Any]] | None = None,
    before_summary: Any | None = None,
    after_summary: Any | None = None,
    source_lineage: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
    schema_version: int = GOVERNANCE_SCHEMA_VERSION,
    criticality: str = GOVERNANCE_OPERATIONAL,
    lineage_root_id: str | None = None,
    idempotency_key: str | None = None,
    producer_name: str | None = None,
    producer_version: str | None = None,
    redaction_policy: str = GOVERNANCE_REDACTION_POLICY,
    retention_class: str = "audit_365d",
) -> dict:
    """Append one structured audit event."""

    conn = _get_conn()
    with _lock:
        event = _record_audit_event_conn(
            conn,
            action_name=action_name,
            action_category=action_category,
            status=status,
            event_id=event_id,
            occurred_at=occurred_at,
            received_at=received_at,
            request_id=request_id,
            actor_id=actor_id,
            actor_type=actor_type,
            parent_actor_id=parent_actor_id,
            object_type=object_type,
            object_id=object_id,
            object_refs=object_refs,
            before_summary=before_summary,
            after_summary=after_summary,
            source_lineage=source_lineage,
            metadata=metadata,
            error=error,
            schema_version=schema_version,
            criticality=criticality,
            lineage_root_id=lineage_root_id,
            idempotency_key=idempotency_key,
            producer_name=producer_name,
            producer_version=producer_version,
            redaction_policy=redaction_policy,
            retention_class=retention_class,
        )
        conn.commit()
    return event


def get_audit_events(
    *,
    request_id: str | None = None,
    actor_id: str | None = None,
    action_name: str | None = None,
    action_category: str | None = None,
    status: str | None = None,
    criticality: str | None = None,
    lineage_root_id: str | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("request_id", request_id),
        ("actor_id", actor_id),
        ("action_name", action_name),
        ("action_category", action_category),
        ("status", status),
        ("criticality", criticality),
        ("lineage_root_id", lineage_root_id),
        ("object_type", object_type),
        ("object_id", object_id),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    if since is not None:
        clauses.append("occurred_at >= ?")
        params.append(since)
    if until is not None:
        clauses.append("occurred_at <= ?")
        params.append(until)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    safe_limit = max(1, min(int(limit), 1000))
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM audit_events{where} ORDER BY occurred_at DESC, id DESC LIMIT ?",
            (*params, safe_limit),
        ).fetchall()
    return [_parse_audit_event_row(row) for row in rows]


def prune_audit_events(*, retention_days: int = 2555, batch_size: int = 5000) -> int:
    if retention_days <= 0:
        return 0
    conn = _get_conn()
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
    safe_batch = max(1, min(int(batch_size), 10000))
    deleted = 0
    while True:
        with _lock:
            rows = conn.execute(
                "SELECT id FROM audit_events WHERE occurred_at < ? ORDER BY occurred_at ASC, id ASC LIMIT ?",
                (cutoff, safe_batch),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            if not ids:
                conn.commit()
                break
            placeholders = ", ".join("?" for _ in ids)
            cur = conn.execute(f"DELETE FROM audit_events WHERE id IN ({placeholders})", tuple(ids))
            conn.commit()
        deleted += int(cur.rowcount or len(ids))
        if len(ids) < safe_batch:
            break
    return deleted


# ---------------------------------------------------------------------------
# Provenance Events
# ---------------------------------------------------------------------------

_PROVENANCE_EVENT_JSON_FIELDS = ("summary_json", "metadata_json")
_PROVENANCE_LINK_JSON_FIELDS = ("metadata_json",)
_SOURCE_RECORD_REF_JSON_FIELDS = ("summary_json",)
_WORKFLOW_ARTIFACT_JSON_FIELDS = ("summary_json",)
_GOVERNANCE_OUTBOX_JSON_FIELDS = ("event_bundle_json",)


def _parse_provenance_event_row(row: Any) -> dict:
    d = _require_row_dict(row)
    for field in _PROVENANCE_EVENT_JSON_FIELDS:
        _parse_json_field(d, field)
    d["summary"] = d.get("summary_json")
    d["metadata"] = d.get("metadata_json")
    return d


def _parse_provenance_link_row(row: Any) -> dict:
    d = _require_row_dict(row)
    for field in _PROVENANCE_LINK_JSON_FIELDS:
        _parse_json_field(d, field)
    d["metadata"] = d.get("metadata_json")
    return d


def _parse_source_record_ref_row(row: Any) -> dict:
    d = _require_row_dict(row)
    for field in _SOURCE_RECORD_REF_JSON_FIELDS:
        _parse_json_field(d, field)
    d["summary"] = d.get("summary_json")
    return d


def _parse_workflow_artifact_row(row: Any) -> dict:
    d = _require_row_dict(row)
    for field in _WORKFLOW_ARTIFACT_JSON_FIELDS:
        _parse_json_field(d, field)
    d["summary"] = d.get("summary_json")
    return d


def _parse_governance_outbox_row(row: Any) -> dict:
    d = _require_row_dict(row)
    for field in _GOVERNANCE_OUTBOX_JSON_FIELDS:
        _parse_json_field(d, field)
    d["event_bundle"] = d.get("event_bundle_json")
    return d


def _provenance_timeline(
    *,
    events: list[dict],
    links: list[dict],
    source_records: list[dict],
    workflow_artifacts: list[dict],
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for event in events:
        timestamp = str(event.get("started_at") or event.get("completed_at") or "")
        timeline.append(
            {
                "kind": "event",
                "timestamp": timestamp,
                "id": event.get("id"),
                "label": event.get("event_name"),
                "event_type": event.get("event_type"),
                "status": event.get("status"),
                "summary": event.get("summary"),
            }
        )
    for link in links:
        timeline.append(
            {
                "kind": "link",
                "timestamp": str(link.get("created_at") or ""),
                "id": link.get("id"),
                "label": link.get("link_type"),
                "source_ref_type": link.get("source_ref_type"),
                "source_ref_id": link.get("source_ref_id"),
                "target_ref_type": link.get("target_ref_type"),
                "target_ref_id": link.get("target_ref_id"),
            }
        )
    for record in source_records:
        timeline.append(
            {
                "kind": "source_record",
                "timestamp": str(record.get("created_at") or record.get("as_of") or ""),
                "id": record.get("record_ref_id"),
                "label": record.get("record_kind"),
                "source_name": record.get("source_name"),
                "summary": record.get("summary"),
            }
        )
    for artifact in workflow_artifacts:
        timeline.append(
            {
                "kind": "workflow_artifact",
                "timestamp": str(artifact.get("created_at") or ""),
                "id": artifact.get("artifact_id"),
                "label": artifact.get("artifact_key"),
                "workflow_run_id": artifact.get("workflow_run_id"),
                "approval_id": artifact.get("approval_id"),
                "summary": artifact.get("summary"),
            }
        )
    timeline.sort(
        key=lambda row: (str(row.get("timestamp") or ""), str(row.get("kind") or ""), str(row.get("id") or ""))
    )
    return timeline


def _upsert_provenance_event_conn(
    conn: sqlite3.Connection | PostgresCompatConnection,
    *,
    event_id: str | None = None,
    event_type: str,
    event_name: str,
    status: str = "started",
    started_at: str | None = None,
    completed_at: str | None = None,
    actor_type: str | None = None,
    actor_id: str | None = None,
    parent_actor_id: str | None = None,
    request_id: str | None = None,
    parent_event_id: str | None = None,
    workflow_run_id: str | None = None,
    ontology_run_id: str | None = None,
    agent_session_id: str | None = None,
    action_run_id: int | None = None,
    approval_id: int | None = None,
    audit_event_id: str | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    schema_version: int = GOVERNANCE_SCHEMA_VERSION,
    criticality: str = GOVERNANCE_OPERATIONAL,
    lineage_root_id: str | None = None,
    idempotency_key: str | None = None,
    producer_name: str | None = None,
    producer_version: str | None = None,
    redaction_policy: str = "audit_summary_v1",
    retention_class: str = "provenance_365d",
    error: str | None = None,
) -> dict:
    eid = event_id or f"pv:{uuid.uuid4().hex}"
    now = _now()
    params = {
        "id": eid,
        "event_type": event_type,
        "event_name": event_name,
        "status": status,
        "started_at": started_at or now,
        "completed_at": completed_at,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "parent_actor_id": parent_actor_id,
        "request_id": request_id,
        "parent_event_id": parent_event_id,
        "workflow_run_id": workflow_run_id,
        "ontology_run_id": ontology_run_id,
        "agent_session_id": agent_session_id,
        "action_run_id": action_run_id,
        "approval_id": approval_id,
        "audit_event_id": audit_event_id,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "summary_json": _json_or_none(summary),
        "metadata_json": _json_or_none(metadata),
        "schema_version": int(schema_version or GOVERNANCE_SCHEMA_VERSION),
        "criticality": criticality or GOVERNANCE_OPERATIONAL,
        "lineage_root_id": lineage_root_id,
        "idempotency_key": idempotency_key,
        "producer_name": producer_name,
        "producer_version": producer_version,
        "redaction_policy": redaction_policy,
        "retention_class": retention_class,
        "error": str(error)[:1000] if error is not None else None,
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    updates = ", ".join(f"{column} = excluded.{column}" for column in params if column != "id")
    conn.execute(
        f"INSERT INTO provenance_events ({columns}) VALUES ({placeholders}) ON CONFLICT(id) DO UPDATE SET {updates}",
        tuple(params.values()),
    )
    row = conn.execute("SELECT * FROM provenance_events WHERE id = ?", (eid,)).fetchone()
    return _parse_provenance_event_row(row)


def upsert_provenance_event(
    *,
    event_id: str | None = None,
    event_type: str,
    event_name: str,
    status: str = "started",
    started_at: str | None = None,
    completed_at: str | None = None,
    actor_type: str | None = None,
    actor_id: str | None = None,
    parent_actor_id: str | None = None,
    request_id: str | None = None,
    parent_event_id: str | None = None,
    workflow_run_id: str | None = None,
    ontology_run_id: str | None = None,
    agent_session_id: str | None = None,
    action_run_id: int | None = None,
    approval_id: int | None = None,
    audit_event_id: str | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    schema_version: int = GOVERNANCE_SCHEMA_VERSION,
    criticality: str = GOVERNANCE_OPERATIONAL,
    lineage_root_id: str | None = None,
    idempotency_key: str | None = None,
    producer_name: str | None = None,
    producer_version: str | None = None,
    redaction_policy: str = "audit_summary_v1",
    retention_class: str = "provenance_365d",
    error: str | None = None,
) -> dict:
    """Insert or update one provenance event.

    This is intentionally plain storage. Redaction and hashing are handled by
    api.provenance before values reach this function.
    """

    conn = _get_conn()
    with _lock:
        event = _upsert_provenance_event_conn(
            conn,
            event_id=event_id,
            event_type=event_type,
            event_name=event_name,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            actor_type=actor_type,
            actor_id=actor_id,
            parent_actor_id=parent_actor_id,
            request_id=request_id,
            parent_event_id=parent_event_id,
            workflow_run_id=workflow_run_id,
            ontology_run_id=ontology_run_id,
            agent_session_id=agent_session_id,
            action_run_id=action_run_id,
            approval_id=approval_id,
            audit_event_id=audit_event_id,
            input_hash=input_hash,
            output_hash=output_hash,
            summary=summary,
            metadata=metadata,
            schema_version=schema_version,
            criticality=criticality,
            lineage_root_id=lineage_root_id,
            idempotency_key=idempotency_key,
            producer_name=producer_name,
            producer_version=producer_version,
            redaction_policy=redaction_policy,
            retention_class=retention_class,
            error=error,
        )
        conn.commit()
    return event


def finish_provenance_event(
    event_id: str,
    *,
    status: str,
    completed_at: str | None = None,
    output_hash: str | None = None,
    summary: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
) -> dict | None:
    conn = _get_conn()
    now = completed_at or _now()
    with _lock:
        conn.execute(
            """
            UPDATE provenance_events
            SET status = ?,
                completed_at = ?,
                output_hash = COALESCE(?, output_hash),
                summary_json = COALESCE(?, summary_json),
                metadata_json = COALESCE(?, metadata_json),
                error = ?
            WHERE id = ?
            """,
            (
                status,
                now,
                output_hash,
                _json_or_none(summary),
                _json_or_none(metadata),
                str(error)[:1000] if error is not None else None,
                event_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM provenance_events WHERE id = ?", (event_id,)).fetchone()
    return _parse_provenance_event_row(row) if row else None


def _upsert_provenance_link_conn(
    conn: sqlite3.Connection | PostgresCompatConnection,
    *,
    link_id: str | None = None,
    event_id: str,
    source_ref_type: str,
    source_ref_id: str,
    target_ref_type: str,
    target_ref_id: str,
    link_type: str,
    source_ref_version: str | None = None,
    target_ref_version: str | None = None,
    metadata: Any | None = None,
    lineage_root_id: str | None = None,
    created_at: str | None = None,
) -> dict:
    lid = (
        link_id
        or f"pvlink:{_json_hash([event_id, source_ref_type, source_ref_id, target_ref_type, target_ref_id, link_type, source_ref_version, target_ref_version])}"
    )
    now = created_at or _now()
    params = {
        "id": lid,
        "event_id": event_id,
        "source_ref_type": source_ref_type,
        "source_ref_id": source_ref_id,
        "source_ref_version": source_ref_version,
        "target_ref_type": target_ref_type,
        "target_ref_id": target_ref_id,
        "target_ref_version": target_ref_version,
        "link_type": link_type,
        "metadata_json": _json_or_none(metadata),
        "lineage_root_id": lineage_root_id,
        "created_at": now,
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    updates = ", ".join(f"{column} = excluded.{column}" for column in params if column != "id")
    conn.execute(
        f"INSERT INTO provenance_links ({columns}) VALUES ({placeholders}) ON CONFLICT(id) DO UPDATE SET {updates}",
        tuple(params.values()),
    )
    row = conn.execute("SELECT * FROM provenance_links WHERE id = ?", (lid,)).fetchone()
    return _parse_provenance_link_row(row)


def upsert_provenance_link(
    *,
    link_id: str | None = None,
    event_id: str,
    source_ref_type: str,
    source_ref_id: str,
    target_ref_type: str,
    target_ref_id: str,
    link_type: str,
    source_ref_version: str | None = None,
    target_ref_version: str | None = None,
    metadata: Any | None = None,
    lineage_root_id: str | None = None,
    created_at: str | None = None,
) -> dict:
    conn = _get_conn()
    with _lock:
        row = _upsert_provenance_link_conn(
            conn,
            link_id=link_id,
            event_id=event_id,
            source_ref_type=source_ref_type,
            source_ref_id=source_ref_id,
            source_ref_version=source_ref_version,
            target_ref_type=target_ref_type,
            target_ref_id=target_ref_id,
            target_ref_version=target_ref_version,
            link_type=link_type,
            metadata=metadata,
            lineage_root_id=lineage_root_id,
            created_at=created_at,
        )
        conn.commit()
    return _parse_provenance_link_row(row)


def upsert_source_record_ref(
    *,
    record_ref_id: str,
    adapter_run_event_id: str,
    source_name: str,
    record_kind: str,
    record_key_hash: str,
    record_hash: str,
    as_of: str | None = None,
    summary: Any | None = None,
    redaction_policy: str = "audit_summary_v1",
    retention_class: str = "source_ref_90d",
    created_at: str | None = None,
) -> dict:
    conn = _get_conn()
    now = created_at or _now()
    params = {
        "record_ref_id": record_ref_id,
        "adapter_run_event_id": adapter_run_event_id,
        "source_name": source_name,
        "record_kind": record_kind,
        "record_key_hash": record_key_hash,
        "record_hash": record_hash,
        "as_of": as_of,
        "summary_json": _json_or_none(summary),
        "redaction_policy": redaction_policy,
        "retention_class": retention_class,
        "created_at": now,
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    updates = ", ".join(f"{column} = excluded.{column}" for column in params if column != "record_ref_id")
    with _lock:
        conn.execute(
            f"INSERT INTO source_record_refs ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(record_ref_id) DO UPDATE SET {updates}",
            tuple(params.values()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM source_record_refs WHERE record_ref_id = ?", (record_ref_id,)).fetchone()
    return _parse_source_record_ref_row(row)


def upsert_workflow_artifact_record(
    *,
    artifact_id: str,
    workflow_run_id: str,
    artifact_key: str,
    artifact_index: int = 0,
    artifact_hash: str,
    summary: Any | None = None,
    approval_id: int | None = None,
    provenance_event_id: str | None = None,
    redaction_policy: str = "audit_summary_v1",
    retention_class: str = "workflow_artifact_365d",
    created_at: str | None = None,
) -> dict:
    conn = _get_conn()
    now = created_at or _now()
    params = {
        "artifact_id": artifact_id,
        "workflow_run_id": workflow_run_id,
        "artifact_key": artifact_key,
        "artifact_index": int(artifact_index),
        "artifact_hash": artifact_hash,
        "summary_json": _json_or_none(summary),
        "approval_id": approval_id,
        "provenance_event_id": provenance_event_id,
        "redaction_policy": redaction_policy,
        "retention_class": retention_class,
        "created_at": now,
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    updates = ", ".join(f"{column} = excluded.{column}" for column in params if column != "artifact_id")
    with _lock:
        conn.execute(
            f"INSERT INTO workflow_artifact_records ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(artifact_id) DO UPDATE SET {updates}",
            tuple(params.values()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM workflow_artifact_records WHERE artifact_id = ?", (artifact_id,)).fetchone()
    return _parse_workflow_artifact_row(row)


# ---------------------------------------------------------------------------
# Governance outbox
# ---------------------------------------------------------------------------


def _governance_idempotency_key(event_bundle: dict[str, Any], prefix: str = "governance") -> str:
    explicit = event_bundle.get("idempotency_key")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return f"{prefix}:{_json_hash(event_bundle)}"


def _governance_retry_jitter_seconds(seed: Any, attempt_count: int, jitter_max_seconds: int) -> int:
    safe_jitter = max(0, int(jitter_max_seconds or 0))
    if safe_jitter <= 0:
        return 0
    raw = f"{seed}:{attempt_count}"
    return int(_json_hash(raw), 16) % (safe_jitter + 1)


def _lineage_root_from_bundle(event_bundle: dict[str, Any]) -> str | None:
    root = event_bundle.get("lineage_root_id")
    if root is not None and str(root).strip():
        return str(root).strip()
    for key in ("audit_events", "provenance_events", "provenance_links"):
        rows = event_bundle.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("lineage_root_id"):
                    return str(row["lineage_root_id"])
    return None


def _enqueue_governance_outbox_tx(
    conn: sqlite3.Connection | PostgresCompatConnection,
    event_bundle: dict[str, Any],
    *,
    idempotency_key: str | None = None,
    lineage_root_id: str | None = None,
    retention_class: str = GOVERNANCE_FINANCIAL_RETENTION_CLASS,
    next_attempt_at: str | None = None,
) -> dict:
    if not isinstance(event_bundle, dict) or not event_bundle:
        raise ValueError("Governance outbox bundle must be a non-empty object")
    now = _now()
    key = idempotency_key or _governance_idempotency_key(event_bundle)
    root = lineage_root_id or _lineage_root_from_bundle(event_bundle)
    params = {
        "idempotency_key": key,
        "event_bundle_json": json.dumps(event_bundle, sort_keys=True, default=str),
        "status": "pending",
        "attempt_count": 0,
        "next_attempt_at": next_attempt_at or now,
        "locked_at": None,
        "last_error": None,
        "dead_lettered_at": None,
        "lineage_root_id": root,
        "retention_class": retention_class,
        "created_at": now,
        "updated_at": now,
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    updates = """
        event_bundle_json = excluded.event_bundle_json,
        next_attempt_at = CASE
            WHEN governance_outbox.status IN ('completed', 'processing') THEN governance_outbox.next_attempt_at
            ELSE excluded.next_attempt_at
        END,
        status = CASE
            WHEN governance_outbox.status = 'completed' THEN governance_outbox.status
            ELSE excluded.status
        END,
        lineage_root_id = COALESCE(excluded.lineage_root_id, governance_outbox.lineage_root_id),
        retention_class = excluded.retention_class,
        updated_at = excluded.updated_at
    """
    conn.execute(
        f"INSERT INTO governance_outbox ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(idempotency_key) DO UPDATE SET {updates}",
        tuple(params.values()),
    )
    row = conn.execute(
        "SELECT * FROM governance_outbox WHERE idempotency_key = ?",
        (key,),
    ).fetchone()
    return _parse_governance_outbox_row(row)


def enqueue_governance_outbox(
    event_bundle: dict[str, Any],
    *,
    idempotency_key: str | None = None,
    lineage_root_id: str | None = None,
    retention_class: str = GOVERNANCE_FINANCIAL_RETENTION_CLASS,
    next_attempt_at: str | None = None,
) -> dict:
    conn = _get_conn()
    with _lock:
        row = _enqueue_governance_outbox_tx(
            conn,
            event_bundle,
            idempotency_key=idempotency_key,
            lineage_root_id=lineage_root_id,
            retention_class=retention_class,
            next_attempt_at=next_attempt_at,
        )
        conn.commit()
    return row


def _materialize_governance_bundle_tx(
    conn: sqlite3.Connection | PostgresCompatConnection,
    event_bundle: dict[str, Any],
) -> dict[str, int]:
    if not isinstance(event_bundle, dict):
        raise ValueError("Governance event bundle must be a JSON object")
    counts = {
        "audit_events": 0,
        "provenance_events": 0,
        "provenance_links": 0,
        "source_record_refs": 0,
        "workflow_artifact_records": 0,
    }
    now = _now()
    root = _lineage_root_from_bundle(event_bundle)
    for event in event_bundle.get("audit_events") or []:
        if not isinstance(event, dict):
            continue
        event_id = str(
            event.get("event_id") or event.get("idempotency_key") or _governance_idempotency_key(event, "audit")
        )
        existing = conn.execute("SELECT id FROM audit_events WHERE event_id = ?", (event_id,)).fetchone()
        if existing:
            continue
        _record_audit_event_conn(
            conn,
            event_id=event_id,
            action_name=str(event["action_name"]),
            action_category=str(event.get("action_category") or "governance"),
            status=str(event.get("status") or "succeeded"),
            occurred_at=event.get("occurred_at"),
            received_at=event.get("received_at"),
            request_id=event.get("request_id"),
            actor_id=event.get("actor_id"),
            actor_type=str(event.get("actor_type") or "system"),
            parent_actor_id=event.get("parent_actor_id"),
            object_type=event.get("object_type"),
            object_id=event.get("object_id"),
            object_refs=event.get("object_refs"),
            before_summary=event.get("before_summary"),
            after_summary=event.get("after_summary"),
            source_lineage=event.get("source_lineage"),
            metadata=event.get("metadata"),
            error=event.get("error"),
            schema_version=int(event.get("schema_version") or GOVERNANCE_SCHEMA_VERSION),
            criticality=str(event.get("criticality") or GOVERNANCE_CRITICAL_FINANCIAL),
            lineage_root_id=str(event.get("lineage_root_id") or root) if event.get("lineage_root_id") or root else None,
            idempotency_key=event.get("idempotency_key"),
            producer_name=event.get("producer_name"),
            producer_version=event.get("producer_version"),
            redaction_policy=str(event.get("redaction_policy") or GOVERNANCE_REDACTION_POLICY),
            retention_class=str(event.get("retention_class") or GOVERNANCE_FINANCIAL_RETENTION_CLASS),
        )
        counts["audit_events"] += 1

    for event in event_bundle.get("provenance_events") or []:
        if not isinstance(event, dict):
            continue
        _upsert_provenance_event_conn(
            conn,
            event_id=event.get("id") or event.get("event_id"),
            event_type=str(event["event_type"]),
            event_name=str(event["event_name"]),
            status=str(event.get("status") or "succeeded"),
            started_at=event.get("started_at") or now,
            completed_at=event.get("completed_at") or now
            if str(event.get("status") or "succeeded") != "started"
            else event.get("completed_at"),
            actor_type=event.get("actor_type"),
            actor_id=event.get("actor_id"),
            parent_actor_id=event.get("parent_actor_id"),
            request_id=event.get("request_id"),
            parent_event_id=event.get("parent_event_id"),
            workflow_run_id=event.get("workflow_run_id"),
            ontology_run_id=event.get("ontology_run_id"),
            agent_session_id=event.get("agent_session_id"),
            action_run_id=event.get("action_run_id"),
            approval_id=event.get("approval_id"),
            audit_event_id=event.get("audit_event_id"),
            input_hash=event.get("input_hash"),
            output_hash=event.get("output_hash"),
            summary=event.get("summary"),
            metadata=event.get("metadata"),
            schema_version=int(event.get("schema_version") or GOVERNANCE_SCHEMA_VERSION),
            criticality=str(event.get("criticality") or GOVERNANCE_CRITICAL_FINANCIAL),
            lineage_root_id=str(event.get("lineage_root_id") or root) if event.get("lineage_root_id") or root else None,
            idempotency_key=event.get("idempotency_key"),
            producer_name=event.get("producer_name"),
            producer_version=event.get("producer_version"),
            redaction_policy=str(event.get("redaction_policy") or GOVERNANCE_REDACTION_POLICY),
            retention_class=str(event.get("retention_class") or GOVERNANCE_FINANCIAL_RETENTION_CLASS),
            error=event.get("error"),
        )
        counts["provenance_events"] += 1

    for link in event_bundle.get("provenance_links") or []:
        if not isinstance(link, dict):
            continue
        _upsert_provenance_link_conn(
            conn,
            link_id=link.get("id") or link.get("link_id"),
            event_id=str(link["event_id"]),
            source_ref_type=str(link["source_ref_type"]),
            source_ref_id=str(link["source_ref_id"]),
            source_ref_version=link.get("source_ref_version"),
            target_ref_type=str(link["target_ref_type"]),
            target_ref_id=str(link["target_ref_id"]),
            target_ref_version=link.get("target_ref_version"),
            link_type=str(link["link_type"]),
            metadata=link.get("metadata"),
            lineage_root_id=str(link.get("lineage_root_id") or root) if link.get("lineage_root_id") or root else None,
            created_at=link.get("created_at") or now,
        )
        counts["provenance_links"] += 1

    for update in event_bundle.get("recommendation_updates") or []:
        if isinstance(update, dict) and update.get("recommendation_id"):
            conn.execute(
                "UPDATE recommendations SET provenance_event_id = COALESCE(?, provenance_event_id), "
                "lineage_root_id = COALESCE(?, lineage_root_id), lineage_completeness = COALESCE(?, lineage_completeness) "
                "WHERE id = ?",
                (
                    update.get("provenance_event_id"),
                    update.get("lineage_root_id") or root,
                    update.get("lineage_completeness"),
                    int(update["recommendation_id"]),
                ),
            )
    for update in event_bundle.get("policy_gate_result_updates") or []:
        if isinstance(update, dict) and update.get("policy_gate_result_id"):
            conn.execute(
                "UPDATE policy_gate_results SET provenance_event_id = COALESCE(?, provenance_event_id), "
                "lineage_root_id = COALESCE(?, lineage_root_id), lineage_completeness = COALESCE(?, lineage_completeness) "
                "WHERE id = ?",
                (
                    update.get("provenance_event_id"),
                    update.get("lineage_root_id") or root,
                    update.get("lineage_completeness"),
                    int(update["policy_gate_result_id"]),
                ),
            )
    for update in event_bundle.get("action_run_updates") or []:
        if isinstance(update, dict) and update.get("action_run_id"):
            conn.execute(
                "UPDATE action_runs SET provenance_event_id = COALESCE(?, provenance_event_id), "
                "lineage_completeness = COALESCE(?, lineage_completeness) WHERE id = ?",
                (
                    update.get("provenance_event_id"),
                    update.get("lineage_completeness"),
                    int(update["action_run_id"]),
                ),
            )
    for update in event_bundle.get("approval_updates") or []:
        if isinstance(update, dict) and update.get("approval_id"):
            conn.execute(
                "UPDATE pending_approvals SET provenance_event_id = COALESCE(?, provenance_event_id), "
                "origin_provenance_event_id = COALESCE(?, origin_provenance_event_id), "
                "origin_artifact_id = COALESCE(?, origin_artifact_id), "
                "lineage_completeness = COALESCE(?, lineage_completeness) WHERE id = ?",
                (
                    update.get("provenance_event_id"),
                    update.get("origin_provenance_event_id"),
                    update.get("origin_artifact_id"),
                    update.get("lineage_completeness"),
                    int(update["approval_id"]),
                ),
            )
    for update in event_bundle.get("workflow_run_updates") or []:
        if isinstance(update, dict) and update.get("workflow_run_id"):
            conn.execute(
                "UPDATE workflow_runs SET provenance_event_id = COALESCE(?, provenance_event_id), "
                "lineage_completeness = COALESCE(?, lineage_completeness) WHERE run_id = ?",
                (
                    update.get("provenance_event_id"),
                    update.get("lineage_completeness"),
                    str(update["workflow_run_id"]),
                ),
            )
    return counts


def materialize_governance_bundle(event_bundle: dict[str, Any]) -> dict[str, int]:
    conn = _get_conn()
    with _lock:
        counts = _materialize_governance_bundle_tx(conn, event_bundle)
        conn.commit()
    return counts


def get_governance_outbox_items(
    *,
    status: str | None = None,
    lineage_root_id: str | None = None,
    limit: int = 100,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if lineage_root_id:
        clauses.append("lineage_root_id = ?")
        params.append(lineage_root_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    safe_limit = max(1, min(int(limit), 1000))
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM governance_outbox{where} ORDER BY created_at ASC, id ASC LIMIT ?",
            (*params, safe_limit),
        ).fetchall()
    return [_parse_governance_outbox_row(row) for row in rows]


def requeue_governance_outbox_item(
    *,
    outbox_id: int | None = None,
    idempotency_key: str | None = None,
    next_attempt_at: str | None = None,
) -> dict:
    if outbox_id is None and not idempotency_key:
        raise ValueError("Provide outbox_id or idempotency_key")
    conn = _get_conn()
    now = _now()
    with _lock:
        if outbox_id is not None:
            conn.execute(
                """
                UPDATE governance_outbox
                SET status = 'pending',
                    locked_at = NULL,
                    last_error = NULL,
                    dead_lettered_at = NULL,
                    next_attempt_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (next_attempt_at or now, now, int(outbox_id)),
            )
            row = conn.execute("SELECT * FROM governance_outbox WHERE id = ?", (int(outbox_id),)).fetchone()
        else:
            conn.execute(
                """
                UPDATE governance_outbox
                SET status = 'pending',
                    locked_at = NULL,
                    last_error = NULL,
                    dead_lettered_at = NULL,
                    next_attempt_at = ?,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (next_attempt_at or now, now, idempotency_key),
            )
            row = conn.execute(
                "SELECT * FROM governance_outbox WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            conn.rollback()
            raise ValueError("Governance outbox item not found")
        conn.commit()
    return _parse_governance_outbox_row(row)


def claim_governance_outbox_batch(*, limit: int = 50, lease_seconds: int = 300) -> list[dict]:
    conn = _get_conn()
    now_dt = datetime.now(UTC)
    now = now_dt.isoformat()
    stale_cutoff = (now_dt - timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
    safe_limit = max(1, min(int(limit), 500))
    with _lock:
        rows = conn.execute(
            """
            SELECT id FROM governance_outbox
            WHERE (
                    status IN ('pending', 'failed')
                    AND next_attempt_at <= ?
                  )
               OR (
                    status = 'processing'
                    AND locked_at IS NOT NULL
                    AND locked_at <= ?
                  )
            ORDER BY next_attempt_at ASC, id ASC
            LIMIT ?
            """,
            (now, stale_cutoff, safe_limit),
        ).fetchall()
        ids = [int(row["id"]) for row in rows]
        claimed: list[dict] = []
        for oid in ids:
            conn.execute(
                "UPDATE governance_outbox SET status = 'processing', attempt_count = attempt_count + 1, "
                "locked_at = ?, updated_at = ? WHERE id = ?",
                (now, now, oid),
            )
            row = conn.execute("SELECT * FROM governance_outbox WHERE id = ?", (oid,)).fetchone()
            claimed.append(_parse_governance_outbox_row(row))
        conn.commit()
    return claimed


def drain_governance_outbox(
    *,
    limit: int = 50,
    lease_seconds: int = 300,
    max_attempts: int = 8,
    retry_base_seconds: int = 30,
    retry_max_seconds: int = 3600,
    retry_jitter_seconds: int = 30,
) -> dict[str, Any]:
    items = claim_governance_outbox_batch(limit=limit, lease_seconds=lease_seconds)
    conn = _get_conn()
    completed = 0
    failed = 0
    dead_lettered = 0
    errors: list[dict[str, Any]] = []
    for item in items:
        oid = int(item["id"])
        bundle = item.get("event_bundle")
        try:
            if not isinstance(bundle, dict):
                raise ValueError("Outbox bundle is not a JSON object")
            with _lock:
                _materialize_governance_bundle_tx(conn, bundle)
                now = _now()
                conn.execute(
                    "UPDATE governance_outbox SET status = 'completed', locked_at = NULL, last_error = NULL, "
                    "updated_at = ? WHERE id = ?",
                    (now, oid),
                )
                conn.commit()
            completed += 1
        except Exception as exc:
            conn.rollback()
            failed += 1
            error = _approval_error_message(exc)
            attempt_count = int(item.get("attempt_count") or 1)
            now_dt = datetime.now(UTC)
            if attempt_count >= max(1, int(max_attempts)):
                status = "dead_letter"
                dead_lettered_at = now_dt.isoformat()
                next_attempt = item.get("next_attempt_at") or now_dt.isoformat()
                dead_lettered += 1
            else:
                status = "failed"
                dead_lettered_at = None
                delay = min(
                    max(1, int(retry_max_seconds)), max(1, int(retry_base_seconds)) * (2 ** (attempt_count - 1))
                )
                jitter = _governance_retry_jitter_seconds(
                    item.get("idempotency_key") or oid, attempt_count, retry_jitter_seconds
                )
                next_attempt = (now_dt + timedelta(seconds=delay + jitter)).isoformat()
            with _lock:
                conn.execute(
                    "UPDATE governance_outbox SET status = ?, locked_at = NULL, last_error = ?, "
                    "next_attempt_at = ?, dead_lettered_at = COALESCE(?, dead_lettered_at), updated_at = ? "
                    "WHERE id = ?",
                    (status, error, next_attempt, dead_lettered_at, now_dt.isoformat(), oid),
                )
                conn.commit()
            errors.append({"id": oid, "idempotency_key": item.get("idempotency_key"), "error": error, "status": status})
    return {
        "claimed": len(items),
        "completed": completed,
        "failed": failed,
        "dead_lettered": dead_lettered,
        "errors": errors[:20],
    }


def get_governance_outbox_metrics() -> dict[str, Any]:
    conn = _get_conn()
    now_dt = datetime.now(UTC)
    with _lock:
        rows = conn.execute("SELECT status, COUNT(*) AS count FROM governance_outbox GROUP BY status").fetchall()
        oldest = conn.execute(
            "SELECT created_at FROM governance_outbox WHERE status IN ('pending', 'failed') ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
    counts = {str(row["status"]): int(row["count"] or 0) for row in rows}
    oldest_age_seconds = None
    if oldest and oldest["created_at"]:
        try:
            created = datetime.fromisoformat(str(oldest["created_at"]))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            oldest_age_seconds = max(0.0, (now_dt - created).total_seconds())
        except ValueError:
            oldest_age_seconds = None
    return {
        "counts": counts,
        "pending": counts.get("pending", 0),
        "failed": counts.get("failed", 0),
        "processing": counts.get("processing", 0),
        "dead_letter": counts.get("dead_letter", 0),
        "oldest_pending_age_seconds": oldest_age_seconds,
    }


def set_workflow_run_provenance_event(run_id: str, provenance_event_id: str | None) -> None:
    conn = _get_conn()
    with _lock:
        if provenance_event_id is not None:
            conn.execute(
                "UPDATE workflow_runs SET provenance_event_id = ?, lineage_completeness = 'complete' WHERE run_id = ?",
                (provenance_event_id, run_id),
            )
        else:
            conn.execute(
                "UPDATE workflow_runs SET provenance_event_id = ? WHERE run_id = ?",
                (provenance_event_id, run_id),
            )
        conn.commit()


def set_action_run_provenance_event(action_run_id: int, provenance_event_id: str | None) -> None:
    conn = _get_conn()
    with _lock:
        if provenance_event_id is not None:
            conn.execute(
                "UPDATE action_runs SET provenance_event_id = ?, lineage_completeness = 'complete' WHERE id = ?",
                (provenance_event_id, action_run_id),
            )
        else:
            conn.execute(
                "UPDATE action_runs SET provenance_event_id = ? WHERE id = ?",
                (provenance_event_id, action_run_id),
            )
        conn.commit()


def set_action_run_lineage_completeness(action_run_id: int, lineage_completeness: str) -> None:
    if lineage_completeness not in GOVERNANCE_LINEAGE_COMPLETENESS_STATES:
        raise ValueError(f"Invalid lineage completeness: {lineage_completeness}")
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE action_runs SET lineage_completeness = ? WHERE id = ?", (lineage_completeness, action_run_id)
        )
        conn.commit()


def set_workflow_run_lineage_completeness(run_id: str, lineage_completeness: str) -> None:
    if lineage_completeness not in GOVERNANCE_LINEAGE_COMPLETENESS_STATES:
        raise ValueError(f"Invalid lineage completeness: {lineage_completeness}")
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE workflow_runs SET lineage_completeness = ? WHERE run_id = ?",
            (lineage_completeness, run_id),
        )
        conn.commit()


def set_pending_approval_lineage_completeness(approval_id: int, lineage_completeness: str) -> None:
    if lineage_completeness not in GOVERNANCE_LINEAGE_COMPLETENESS_STATES:
        raise ValueError(f"Invalid lineage completeness: {lineage_completeness}")
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE pending_approvals SET lineage_completeness = ? WHERE id = ?",
            (lineage_completeness, approval_id),
        )
        conn.commit()


def set_pending_approval_provenance(
    approval_id: int,
    *,
    provenance_event_id: str | None = None,
    origin_provenance_event_id: str | None = None,
    origin_artifact_id: str | None = None,
) -> None:
    conn = _get_conn()
    with _lock:
        if provenance_event_id is not None:
            conn.execute(
                """
                UPDATE pending_approvals
                SET provenance_event_id = COALESCE(?, provenance_event_id),
                    origin_provenance_event_id = COALESCE(?, origin_provenance_event_id),
                    origin_artifact_id = COALESCE(?, origin_artifact_id),
                    lineage_completeness = 'complete'
                WHERE id = ?
                """,
                (provenance_event_id, origin_provenance_event_id, origin_artifact_id, approval_id),
            )
        else:
            conn.execute(
                """
                UPDATE pending_approvals
                SET provenance_event_id = COALESCE(?, provenance_event_id),
                    origin_provenance_event_id = COALESCE(?, origin_provenance_event_id),
                    origin_artifact_id = COALESCE(?, origin_artifact_id)
                WHERE id = ?
                """,
                (provenance_event_id, origin_provenance_event_id, origin_artifact_id, approval_id),
            )
        conn.commit()


def get_provenance_event(event_id: str) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM provenance_events WHERE id = ?", (event_id,)).fetchone()
    return _parse_provenance_event_row(row) if row else None


def get_provenance_trace(
    *,
    workflow_run_id: str | None = None,
    ontology_run_id: str | None = None,
    approval_id: int | None = None,
    action_run_id: int | None = None,
    agent_session_id: str | None = None,
    event_id: str | None = None,
    ref_type: str | None = None,
    ref_id: str | None = None,
    max_depth: int = 4,
) -> dict:
    conn = _get_conn()
    safe_depth = max(1, min(int(max_depth), 8))
    event_ids: set[str] = set()
    refs: set[tuple[str, str]] = set()
    if ref_type and ref_id:
        refs.add((str(ref_type), str(ref_id)))
    if workflow_run_id:
        refs.add(("workflow_run", str(workflow_run_id)))
    if ontology_run_id:
        refs.add(("ontology_run", str(ontology_run_id)))
    if approval_id is not None:
        refs.add(("approval", str(approval_id)))
    if action_run_id is not None:
        refs.add(("action_run", str(action_run_id)))
    with _lock:
        if event_id:
            event_ids.add(event_id)
        for column, value in (
            ("workflow_run_id", workflow_run_id),
            ("ontology_run_id", ontology_run_id),
            ("approval_id", approval_id),
            ("action_run_id", action_run_id),
            ("agent_session_id", agent_session_id),
        ):
            if value is None:
                continue
            rows = conn.execute(f"SELECT id FROM provenance_events WHERE {column} = ?", (value,)).fetchall()
            event_ids.update(str(row["id"]) for row in rows)

        links_by_id: dict[str, dict] = {}
        for _ in range(safe_depth):
            next_event_ids = set(event_ids)
            next_refs = set(refs)
            if event_ids:
                placeholders = ", ".join("?" for _ in event_ids)
                rows = conn.execute(
                    f"""
                    SELECT id, parent_event_id
                    FROM provenance_events
                    WHERE parent_event_id IN ({placeholders})
                       OR id IN (
                            SELECT parent_event_id
                            FROM provenance_events
                            WHERE id IN ({placeholders})
                              AND parent_event_id IS NOT NULL
                       )
                    """,
                    tuple(event_ids) + tuple(event_ids),
                ).fetchall()
                for row in rows:
                    next_event_ids.add(str(row["id"]))
                    if row["parent_event_id"]:
                        next_event_ids.add(str(row["parent_event_id"]))
                for row in conn.execute(
                    f"SELECT * FROM provenance_links WHERE event_id IN ({placeholders})",
                    tuple(event_ids),
                ).fetchall():
                    link = _parse_provenance_link_row(row)
                    links_by_id[str(link["id"])] = link
                    next_refs.add((str(link["source_ref_type"]), str(link["source_ref_id"])))
                    next_refs.add((str(link["target_ref_type"]), str(link["target_ref_id"])))
            for rtype, rid in refs:
                rows = conn.execute(
                    """
                    SELECT * FROM provenance_links
                    WHERE (source_ref_type = ? AND source_ref_id = ?)
                       OR (target_ref_type = ? AND target_ref_id = ?)
                    """,
                    (rtype, rid, rtype, rid),
                ).fetchall()
                for row in rows:
                    link = _parse_provenance_link_row(row)
                    links_by_id[str(link["id"])] = link
                    next_event_ids.add(str(link["event_id"]))
                    next_refs.add((str(link["source_ref_type"]), str(link["source_ref_id"])))
                    next_refs.add((str(link["target_ref_type"]), str(link["target_ref_id"])))
            if next_event_ids == event_ids and next_refs == refs:
                break
            event_ids, refs = next_event_ids, next_refs

        events: list[dict] = []
        if event_ids:
            placeholders = ", ".join("?" for _ in event_ids)
            rows = conn.execute(
                f"SELECT * FROM provenance_events WHERE id IN ({placeholders}) ORDER BY started_at ASC, id ASC",
                tuple(event_ids),
            ).fetchall()
            events = [_parse_provenance_event_row(row) for row in rows]

        source_records: list[dict] = []
        if event_ids:
            placeholders = ", ".join("?" for _ in event_ids)
            rows = conn.execute(
                f"SELECT * FROM source_record_refs WHERE adapter_run_event_id IN ({placeholders}) ORDER BY created_at ASC",
                tuple(event_ids),
            ).fetchall()
            source_records = [_parse_source_record_ref_row(row) for row in rows]

        artifact_clauses: list[str] = []
        artifact_params: list[Any] = []
        if workflow_run_id:
            artifact_clauses.append("workflow_run_id = ?")
            artifact_params.append(workflow_run_id)
        if approval_id is not None:
            artifact_clauses.append("approval_id = ?")
            artifact_params.append(approval_id)
        if event_ids:
            placeholders = ", ".join("?" for _ in event_ids)
            artifact_clauses.append(f"provenance_event_id IN ({placeholders})")
            artifact_params.extend(event_ids)
        artifacts: list[dict] = []
        if artifact_clauses:
            rows = conn.execute(
                f"SELECT * FROM workflow_artifact_records WHERE {' OR '.join(artifact_clauses)} ORDER BY created_at ASC",
                tuple(artifact_params),
            ).fetchall()
            artifacts = [_parse_workflow_artifact_row(row) for row in rows]

    links = sorted(links_by_id.values(), key=lambda row: (str(row.get("created_at") or ""), str(row.get("id") or "")))
    return {
        "events": events,
        "links": links,
        "source_records": source_records,
        "workflow_artifacts": artifacts,
        "timeline": _provenance_timeline(
            events=events,
            links=links,
            source_records=source_records,
            workflow_artifacts=artifacts,
        ),
        "seed": {
            "workflow_run_id": workflow_run_id,
            "ontology_run_id": ontology_run_id,
            "approval_id": approval_id,
            "action_run_id": action_run_id,
            "agent_session_id": agent_session_id,
            "event_id": event_id,
            "ref_type": ref_type,
            "ref_id": ref_id,
        },
    }


def provenance_summary(
    *,
    workflow_run_id: str | None = None,
    ontology_run_id: str | None = None,
    approval_id: int | None = None,
    action_run_id: int | None = None,
    agent_session_id: str | None = None,
    event_id: str | None = None,
    ref_type: str | None = None,
    ref_id: str | None = None,
) -> dict:
    trace = get_provenance_trace(
        workflow_run_id=workflow_run_id,
        ontology_run_id=ontology_run_id,
        approval_id=approval_id,
        action_run_id=action_run_id,
        agent_session_id=agent_session_id,
        event_id=event_id,
        ref_type=ref_type,
        ref_id=ref_id,
        max_depth=2,
    )
    events = trace["events"]
    return {
        "event_count": len(events),
        "link_count": len(trace["links"]),
        "source_record_count": len(trace["source_records"]),
        "workflow_artifact_count": len(trace["workflow_artifacts"]),
        "events": [
            {
                "id": row.get("id"),
                "event_type": row.get("event_type"),
                "event_name": row.get("event_name"),
                "status": row.get("status"),
                "started_at": row.get("started_at"),
                "completed_at": row.get("completed_at"),
            }
            for row in events[:20]
        ],
    }


def get_decision_lineage_report(
    *,
    recommendation_id: int | None = None,
    approval_id: int | None = None,
    action_run_id: int | None = None,
    workflow_run_id: str | None = None,
    object_version_id: str | None = None,
    relation_version_id: str | None = None,
    max_depth: int = 5,
) -> dict[str, Any]:
    selectors = [
        recommendation_id,
        approval_id,
        action_run_id,
        workflow_run_id,
        object_version_id,
        relation_version_id,
    ]
    if sum(value is not None for value in selectors) != 1:
        raise ValueError("Provide exactly one lineage selector")

    ref_type: str | None = None
    ref_id: str | None = None
    lineage_root_id: str | None = None
    completeness: str | None = None
    object_snapshot: dict[str, Any] | None = None
    if recommendation_id is not None:
        rec = get_recommendation(int(recommendation_id))
        if not rec:
            raise ValueError(f"No recommendation with id {recommendation_id}")
        ref_type, ref_id = "recommendation", str(recommendation_id)
        lineage_root_id = rec.get("lineage_root_id") or f"{ref_type}:{ref_id}"
        completeness = rec.get("lineage_completeness")
        object_snapshot = {
            "recommendation_id": recommendation_id,
            "ticker": rec.get("ticker"),
            "action": rec.get("action"),
            "status": rec.get("status"),
            "approval_id": rec.get("approval_id"),
            "policy_gate_result_id": rec.get("policy_gate_result_id"),
            "model": rec.get("model"),
            "prompt_hash": rec.get("prompt_hash"),
            "input_hash": rec.get("input_hash"),
        }
    elif approval_id is not None:
        approval = get_pending_approval(int(approval_id))
        if not approval:
            raise ValueError(f"No approval with id {approval_id}")
        ref_type, ref_id = "approval", str(approval_id)
        lineage_root_id = f"{ref_type}:{ref_id}"
        completeness = approval.get("lineage_completeness") or (
            "complete" if approval.get("provenance_event_id") else "retry_pending"
        )
        object_snapshot = {
            "approval_id": approval_id,
            "entity_type": approval.get("entity_type"),
            "entity_id": approval.get("entity_id"),
            "action_id": approval.get("action_id"),
            "status": approval.get("status"),
            "application_status": approval.get("application_status"),
            "provenance_event_id": approval.get("provenance_event_id"),
            "origin_provenance_event_id": approval.get("origin_provenance_event_id"),
        }
    elif action_run_id is not None:
        action_run = get_action_run(int(action_run_id))
        if not action_run:
            raise ValueError(f"No action run with id {action_run_id}")
        ref_type, ref_id = "action_run", str(action_run_id)
        lineage_root_id = f"{ref_type}:{ref_id}"
        completeness = action_run.get("lineage_completeness") or (
            "complete" if action_run.get("provenance_event_id") else "retry_pending"
        )
        object_snapshot = {
            "action_run_id": action_run_id,
            "action_id": action_run.get("action_id"),
            "status": action_run.get("status"),
            "approval_id": action_run.get("approval_id"),
            "input_hash": action_run.get("input_hash"),
            "provenance_event_id": action_run.get("provenance_event_id"),
        }
    elif workflow_run_id is not None:
        workflow = get_workflow_run(workflow_run_id)
        if not workflow:
            raise ValueError(f"No workflow run with id {workflow_run_id}")
        ref_type, ref_id = "workflow_run", workflow_run_id
        lineage_root_id = f"{ref_type}:{ref_id}"
        completeness = workflow.get("lineage_completeness") or (
            "complete" if workflow.get("provenance_event_id") else "retry_pending"
        )
        object_snapshot = {
            "workflow_run_id": workflow_run_id,
            "workflow_name": workflow.get("workflow_name"),
            "ticker": workflow.get("ticker"),
            "status": workflow.get("status"),
            "provenance_event_id": workflow.get("provenance_event_id"),
        }
    elif object_version_id is not None:
        ref_type, ref_id = "ontology_object_version", str(object_version_id)
        lineage_root_id = f"{ref_type}:{ref_id}"
        completeness = "unknown"
    elif relation_version_id is not None:
        ref_type, ref_id = "relation_version", str(relation_version_id)
        lineage_root_id = f"{ref_type}:{ref_id}"
        completeness = "unknown"

    trace = get_provenance_trace(
        workflow_run_id=workflow_run_id,
        approval_id=approval_id,
        action_run_id=action_run_id,
        ref_type=ref_type
        if recommendation_id is not None or object_version_id is not None or relation_version_id is not None
        else None,
        ref_id=ref_id
        if recommendation_id is not None or object_version_id is not None or relation_version_id is not None
        else None,
        max_depth=max_depth,
    )
    audit_events = get_audit_events(lineage_root_id=lineage_root_id, limit=500) if lineage_root_id else []
    if not audit_events and ref_type and ref_id:
        audit_events = get_audit_events(object_type=ref_type, object_id=ref_id, limit=500)
    outbox = get_governance_outbox_items(lineage_root_id=lineage_root_id, limit=100) if lineage_root_id else []
    warnings: list[str] = []
    if not trace["events"]:
        warnings.append("No provenance events found for selector.")
    if not audit_events:
        warnings.append("No audit events found for lineage root.")
    if any(row.get("status") in {"pending", "failed", "processing"} for row in outbox):
        warnings.append("Lineage materialization has retry-pending outbox work.")
    if any(row.get("status") == "dead_letter" for row in outbox):
        warnings.append("Lineage materialization has dead-lettered outbox work.")
    if completeness and completeness != "complete":
        warnings.append(f"Lineage completeness is {completeness}.")
    return {
        "selector": {
            "recommendation_id": recommendation_id,
            "approval_id": approval_id,
            "action_run_id": action_run_id,
            "workflow_run_id": workflow_run_id,
            "object_version_id": object_version_id,
            "relation_version_id": relation_version_id,
        },
        "lineage_root_id": lineage_root_id,
        "ref": {"type": ref_type, "id": ref_id},
        "lineage_completeness": completeness,
        "object": object_snapshot,
        "provenance": trace,
        "audit_events": audit_events,
        "outbox": outbox,
        "completeness_warnings": warnings,
    }


def _emit_core_audit(
    action_name: str,
    *,
    status: str,
    object_refs: list[dict[str, Any]] | None = None,
    before_summary: Any | None = None,
    after_summary: Any | None = None,
    source_lineage: Any | None = None,
    metadata: Any | None = None,
    error: str | None = None,
    fail_closed: bool = False,
    criticality: str = GOVERNANCE_OPERATIONAL,
    lineage_root_id: str | None = None,
    idempotency_key: str | None = None,
    retention_class: str = "audit_365d",
) -> None:
    try:
        from api.audit import emit_audit_event

        emit_audit_event(
            action_name,
            "core_db",
            status,
            object_refs=object_refs,
            before_summary=before_summary,
            after_summary=after_summary,
            source_lineage=source_lineage,
            metadata=metadata,
            error=error,
            fail_closed=fail_closed,
            criticality=criticality,
            lineage_root_id=lineage_root_id,
            idempotency_key=idempotency_key,
            retention_class=retention_class,
        )
    except Exception:
        logger.debug("Failed to emit core audit event action=%s", action_name, exc_info=True)
        if fail_closed:
            raise


def _guard_legacy_domain_write(surface: str) -> None:
    from ontology.domain_write_service import assert_legacy_domain_write_allowed

    assert_legacy_domain_write_allowed(surface)


# ---------------------------------------------------------------------------
# Catalysts
# ---------------------------------------------------------------------------


def create_catalyst(
    ticker: str,
    description: str,
    category: str = "fundamental",
    *,
    target_date: str | None = None,
    evidence: str | None = None,
    created_by: str = "user",
) -> dict:
    _guard_legacy_domain_write("core_db.create_catalyst")
    conn = _get_conn()
    now = _now()
    with _lock:
        cur = conn.execute(
            "INSERT INTO catalysts (ticker, description, category, target_date, evidence, created_at, updated_at, created_by) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ticker.upper(), description, category, target_date, evidence, now, now, created_by),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper(),
        "description": description,
        "category": category,
        "status": "pending",
        "target_date": target_date,
        "evidence": evidence,
        "created_at": now,
        "updated_at": now,
        "created_by": created_by,
    }


def get_catalysts(ticker: str) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM catalysts WHERE ticker = ? ORDER BY created_at DESC",
            (ticker.upper(),),
        ).fetchall()
    return _rows_to_list(rows)


def update_catalyst_status(catalyst_id: int, status: str, evidence: str | None = None) -> dict:
    _guard_legacy_domain_write("core_db.update_catalyst_status")
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM catalysts WHERE id = ?", (catalyst_id,)).fetchone()
        if not row:
            raise ValueError(f"No catalyst with id {catalyst_id}")
        updates = {"status": status, "updated_at": now}
        if evidence is not None:
            updates["evidence"] = evidence
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"UPDATE catalysts SET {set_clause} WHERE id = ?",
            (*updates.values(), catalyst_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM catalysts WHERE id = ?", (catalyst_id,)).fetchone()
    return _require_row_dict(updated)


# ---------------------------------------------------------------------------
# Kill Conditions
# ---------------------------------------------------------------------------


def create_kill_condition(
    ticker: str,
    condition: str,
    *,
    metric: str | None = None,
    threshold: str | None = None,
    created_by: str = "user",
) -> dict:
    _guard_legacy_domain_write("core_db.create_kill_condition")
    conn = _get_conn()
    now = _now()
    with _lock:
        cur = conn.execute(
            "INSERT INTO kill_conditions (ticker, condition, metric, threshold, created_at, updated_at, created_by) "
            "VALUES (?,?,?,?,?,?,?)",
            (ticker.upper(), condition, metric, threshold, now, now, created_by),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper(),
        "condition": condition,
        "metric": metric,
        "threshold": threshold,
        "status": "active",
        "triggered_at": None,
        "created_at": now,
        "updated_at": now,
        "created_by": created_by,
    }


def get_kill_conditions(ticker: str) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM kill_conditions WHERE ticker = ? ORDER BY created_at DESC",
            (ticker.upper(),),
        ).fetchall()
    return _rows_to_list(rows)


def update_kill_condition_status(kc_id: int, status: str) -> dict:
    _guard_legacy_domain_write("core_db.update_kill_condition_status")
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM kill_conditions WHERE id = ?", (kc_id,)).fetchone()
        if not row:
            raise ValueError(f"No kill condition with id {kc_id}")
        triggered_at = now if status == "triggered" else _require_row_dict(row).get("triggered_at")
        conn.execute(
            "UPDATE kill_conditions SET status = ?, triggered_at = ?, updated_at = ? WHERE id = ?",
            (status, triggered_at, now, kc_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM kill_conditions WHERE id = ?", (kc_id,)).fetchone()
    return _require_row_dict(updated)


def delete_catalysts_by_ticker(ticker: str, *, created_by: str | None = None) -> int:
    """Delete catalysts for a ticker. Optionally filter by created_by."""
    conn = _get_conn()
    with _lock:
        if created_by:
            cur = conn.execute(
                "DELETE FROM catalysts WHERE ticker = ? AND created_by = ?",
                (ticker.upper(), created_by),
            )
        else:
            cur = conn.execute("DELETE FROM catalysts WHERE ticker = ?", (ticker.upper(),))
        conn.commit()
    return cur.rowcount


def delete_kill_conditions_by_ticker(ticker: str, *, created_by: str | None = None) -> int:
    """Delete kill conditions for a ticker. Optionally filter by created_by."""
    conn = _get_conn()
    with _lock:
        if created_by:
            cur = conn.execute(
                "DELETE FROM kill_conditions WHERE ticker = ? AND created_by = ?",
                (ticker.upper(), created_by),
            )
        else:
            cur = conn.execute("DELETE FROM kill_conditions WHERE ticker = ?", (ticker.upper(),))
        conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# Workflow Runs
# ---------------------------------------------------------------------------


def create_workflow_run(
    workflow_name: str,
    ticker: str | None = None,
    run_id: str | None = None,
) -> dict:
    from api import governance

    conn = _get_conn()
    now = _now()
    rid = run_id or uuid.uuid4().hex
    with _lock:
        try:
            conn.execute(
                "INSERT INTO workflow_runs (run_id, workflow_name, ticker, status, started_at) VALUES (?,?,?,?,?)",
                (rid, workflow_name, ticker.upper() if ticker else None, "running", now),
            )
            lineage_root_id = governance.lineage_root("workflow_run", rid)
            provenance_event_id = governance.deterministic_id("pv:workflow_run", rid)
            _materialize_governance_bundle_tx(
                conn,
                {
                    "lineage_root_id": lineage_root_id,
                    "provenance_events": [
                        governance.provenance_event(
                            event_id=provenance_event_id,
                            event_type="workflow_run",
                            event_name=workflow_name,
                            status="started",
                            lineage_root_id=lineage_root_id,
                            workflow_run_id=rid,
                            summary={
                                "workflow_name": workflow_name,
                                "ticker": ticker.upper() if ticker else None,
                                "status": "running",
                            },
                        )
                    ],
                    "provenance_links": [
                        governance.provenance_link(
                            event_id=provenance_event_id,
                            source_ref_type="workflow",
                            source_ref_id=workflow_name,
                            target_ref_type="workflow_run",
                            target_ref_id=rid,
                            link_type="executed_as",
                            lineage_root_id=lineage_root_id,
                        )
                    ],
                    "audit_events": [
                        governance.audit_event(
                            action_name="workflow.run.started",
                            status="started",
                            lineage_root_id=lineage_root_id,
                            object_refs=[
                                {"type": "workflow_run", "id": rid},
                                {"type": "workflow", "id": workflow_name},
                            ],
                            after_summary={
                                "workflow_name": workflow_name,
                                "ticker": ticker.upper() if ticker else None,
                                "status": "running",
                            },
                            source_lineage={"run_id": rid},
                        )
                    ],
                },
            )
            conn.execute(
                "UPDATE workflow_runs SET provenance_event_id = ?, lineage_completeness = 'complete' WHERE run_id = ?",
                (provenance_event_id, rid),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return {
        "run_id": rid,
        "workflow_name": workflow_name,
        "ticker": ticker,
        "status": "running",
        "started_at": now,
        "provenance_event_id": provenance_event_id,
        "lineage_completeness": "complete",
    }


def complete_workflow_run(
    run_id: str,
    synthesis: str,
    artifacts: dict | list | None = None,
    tool_sections: list[dict] | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    artifacts_json = json.dumps(artifacts, default=str) if artifacts else None
    sections_json = json.dumps(tool_sections, default=str) if tool_sections else None
    with _lock:
        conn.execute(
            "UPDATE workflow_runs SET status = 'completed', completed_at = ?, synthesis = ?, artifacts = ?, tool_sections = ? WHERE run_id = ?",
            (now, synthesis, artifacts_json, sections_json, run_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        raise ValueError(f"No workflow run with id {run_id}")
    d = _require_row_dict(row)
    _parse_json_field(d, "artifacts")
    _parse_json_field(d, "tool_sections")
    raw_sections = d.get("tool_sections")
    sections: list[Any] = raw_sections if isinstance(raw_sections, list) else []
    raw_artifacts = d.get("artifacts")
    artifact_payload: dict[Any, Any] | list[Any] = raw_artifacts if isinstance(raw_artifacts, (dict, list)) else {}
    artifact_count = len(artifact_payload) if isinstance(artifact_payload, dict) else len(artifact_payload)
    _emit_core_audit(
        "workflow.run.completed",
        status="succeeded",
        object_refs=[{"type": "workflow_run", "id": run_id}, {"type": "workflow", "id": d.get("workflow_name")}],
        after_summary={
            "workflow_name": d.get("workflow_name"),
            "ticker": d.get("ticker"),
            "status": "completed",
            "tool_section_count": len(sections),
            "artifact_count": artifact_count,
            "synthesis_hash": _json_hash(synthesis),
        },
        source_lineage={"run_id": run_id},
    )
    try:
        from api import provenance

        event_id = str(d.get("provenance_event_id") or provenance.deterministic_id("pv:workflow_run", run_id))
        provenance.finish_event(
            event_id,
            status="succeeded",
            output_value={
                "run_id": run_id,
                "artifact_count": artifact_count,
                "tool_section_count": len(sections),
                "synthesis_hash": _json_hash(synthesis),
            },
            summary={
                "workflow_name": d.get("workflow_name"),
                "ticker": d.get("ticker"),
                "status": "completed",
                "tool_section_count": len(sections),
                "artifact_count": artifact_count,
            },
            metadata={"synthesis_hash": _json_hash(synthesis)},
        )
    except Exception:
        pass
    return d


def fail_workflow_run(run_id: str, error: str) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        conn.execute(
            "UPDATE workflow_runs SET status = 'failed', completed_at = ?, error = ? WHERE run_id = ?",
            (now, error, run_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        raise ValueError(f"No workflow run with id {run_id}")
    d = _require_row_dict(row)
    _emit_core_audit(
        "workflow.run.failed",
        status="failed",
        object_refs=[{"type": "workflow_run", "id": run_id}, {"type": "workflow", "id": d.get("workflow_name")}],
        after_summary={"workflow_name": d.get("workflow_name"), "ticker": d.get("ticker"), "status": "failed"},
        source_lineage={"run_id": run_id},
        error=error,
    )
    try:
        from api import provenance

        event_id = str(d.get("provenance_event_id") or provenance.deterministic_id("pv:workflow_run", run_id))
        provenance.finish_event(
            event_id,
            status="failed",
            summary={"workflow_name": d.get("workflow_name"), "ticker": d.get("ticker"), "status": "failed"},
            error=error,
        )
    except Exception:
        pass
    return d


def get_workflow_runs(
    workflow_name: str | None = None,
    ticker: str | None = None,
    limit: int = 20,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list = []
    if workflow_name:
        clauses.append("workflow_name = ?")
        params.append(workflow_name)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM workflow_runs{where} ORDER BY started_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    results = _rows_to_list(rows)
    for d in results:
        _parse_json_field(d, "artifacts")
        _parse_json_field(d, "tool_sections")
    _emit_core_audit(
        "workflow.runs.read",
        status="succeeded",
        after_summary={
            "workflow_name": workflow_name,
            "ticker": ticker.upper() if ticker else None,
            "result_count": len(results),
            "limit": limit,
        },
    )
    return results


def get_workflow_run(run_id: str) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return None
    d = _require_row_dict(row)
    _parse_json_field(d, "artifacts")
    _parse_json_field(d, "tool_sections")
    _emit_core_audit(
        "workflow.run.read",
        status="succeeded",
        object_refs=[{"type": "workflow_run", "id": run_id}, {"type": "workflow", "id": d.get("workflow_name")}],
        after_summary={"run_id": run_id, "workflow_name": d.get("workflow_name"), "status": d.get("status")},
    )
    return d


# ---------------------------------------------------------------------------
# Report Runs
# ---------------------------------------------------------------------------

_REPORT_RUN_JSON_FIELDS = ("summary_json", "artifact_paths_json")


def _parse_report_run_json_fields(d: dict) -> dict:
    for field in _REPORT_RUN_JSON_FIELDS:
        _parse_json_field(d, field)
    return d


def upsert_report_run(record: dict) -> dict:
    _guard_legacy_domain_write("core_db.upsert_report_run")
    conn = _get_conn()
    now = _now()
    report_type = str(record["report_type"])
    as_of = str(record["as_of"])
    report_id = str(record.get("report_id") or f"{report_type}:{as_of}")
    params = {
        "report_id": report_id,
        "report_type": report_type,
        "as_of": as_of,
        "source": record.get("source") or "github_actions",
        "source_run_id": record.get("source_run_id"),
        "source_url": record.get("source_url"),
        "status": record.get("status") or "completed",
        "report_hash": record.get("report_hash"),
        "input_hash": record.get("input_hash"),
        "summary_json": json.dumps(record.get("summary", {}), default=str),
        "artifact_paths_json": json.dumps(record.get("artifact_paths", {}), default=str),
        "issue_url": record.get("issue_url"),
        "created_at": record.get("created_at") or now,
        "updated_at": now,
        "synced_at": record.get("synced_at") or now,
        "error": record.get("error"),
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    updates = ", ".join(
        f"{column} = excluded.{column}" for column in params if column not in {"report_id", "created_at"}
    )
    with _lock:
        conn.execute(
            f"INSERT INTO report_runs ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(report_id) DO UPDATE SET {updates}",
            tuple(params.values()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM report_runs WHERE report_id = ?", (report_id,)).fetchone()
    result = _parse_report_run_json_fields(_require_row_dict(row))
    _emit_core_audit(
        "report.run.upserted",
        status=str(result.get("status") or "completed"),
        object_refs=[{"type": "report_run", "id": report_id}, {"type": "report_type", "id": report_type}],
        after_summary={
            "report_id": report_id,
            "report_type": report_type,
            "as_of": as_of,
            "status": result.get("status"),
            "report_hash": result.get("report_hash"),
            "input_hash": result.get("input_hash"),
        },
        source_lineage={
            "source": result.get("source"),
            "source_run_id": result.get("source_run_id"),
            "source_url": result.get("source_url"),
        },
        error=str(result.get("error")) if result.get("error") else None,
    )
    return result


def get_report_runs(report_type: str | None = None, limit: int = 20) -> list[dict]:
    conn = _get_conn()
    safe_limit = max(1, min(int(limit), 100))
    if report_type:
        with _lock:
            rows = conn.execute(
                "SELECT * FROM report_runs WHERE report_type = ? ORDER BY as_of DESC, synced_at DESC LIMIT ?",
                (report_type, safe_limit),
            ).fetchall()
    else:
        with _lock:
            rows = conn.execute(
                "SELECT * FROM report_runs ORDER BY as_of DESC, synced_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
    return [_parse_report_run_json_fields(d) for d in _rows_to_list(rows)]


# ---------------------------------------------------------------------------
# Continuous Optimization
# ---------------------------------------------------------------------------

DEFAULT_OPTIMIZATION_MISSION_NAME = "Daily Command Center"
DEFAULT_OPTIMIZATION_SCHEDULE = "Weekdays at 10:15 ET"
_OPTIMIZATION_MISSION_JSON_FIELDS = ("scenario_json", "source_config_json", "thresholds_json")
_OPTIMIZATION_RUN_JSON_FIELDS = ("summary_json", "source_freshness_json")
_OPTIMIZATION_SNAPSHOT_JSON_FIELDS = ("evidence_json", "source_links_json")
_OPTIMIZATION_ALERT_JSON_FIELDS = ("evidence_json",)


def _default_optimization_scenario() -> dict[str, Any]:
    return {
        "preset": "balanced",
        "factor_weights": {
            "quality": 0.30,
            "price_momentum": 0.40,
            "fundamental_momentum": 0.30,
            "valuation": 0.0,
        },
        "fundamental_momentum_weights": {"revenue": 0.67, "eps": 0.33},
        "valuation_weights": {
            "price_sales": 0.25,
            "price_operating_income": 0.25,
            "price_fcf": 0.25,
            "price_earnings": 0.25,
        },
        "brakes": {
            "drawdown_sensitivity": 0.0,
            "contrarian_penalty": 0.0,
            "short_squeeze_brake": 0.0,
        },
    }


def _default_optimization_sources() -> dict[str, Any]:
    return {
        "modules": [
            "portfolio_analyzer",
            "portfolio_risk",
            "position_risk",
            "portfolio_sizer",
            "hedging",
            "thesis_pressure",
            "watch_triggers",
            "report_recommendations",
            "workflow_runs",
            "macro_signal_regime",
        ],
        "mode": "recommend_and_stage",
    }


def _default_optimization_thresholds() -> dict[str, Any]:
    return {
        "confidence_bucket_edges": [0.35, 0.65, 0.8],
        "priority_bucket_edges": [0.75, 1.5, 2.5],
        "stage_actions": True,
        "suppress_low_severity_holds": True,
    }


def _parse_optimization_mission_json_fields(d: dict) -> dict:
    for field in _OPTIMIZATION_MISSION_JSON_FIELDS:
        _parse_json_field(d, field)
    d["scenario"] = d.get("scenario_json") if isinstance(d.get("scenario_json"), dict) else {}
    d["source_config"] = d.get("source_config_json") if isinstance(d.get("source_config_json"), dict) else {}
    d["thresholds"] = d.get("thresholds_json") if isinstance(d.get("thresholds_json"), dict) else {}
    return d


def _parse_optimization_run_json_fields(d: dict) -> dict:
    for field in _OPTIMIZATION_RUN_JSON_FIELDS:
        _parse_json_field(d, field)
    d["summary"] = d.get("summary_json") if isinstance(d.get("summary_json"), dict) else {}
    d["source_freshness"] = d.get("source_freshness_json") if isinstance(d.get("source_freshness_json"), dict) else {}
    return d


def _parse_optimization_snapshot_json_fields(d: dict) -> dict:
    for field in _OPTIMIZATION_SNAPSHOT_JSON_FIELDS:
        _parse_json_field(d, field)
    d["evidence"] = d.get("evidence_json") if isinstance(d.get("evidence_json"), dict) else {}
    d["source_links"] = d.get("source_links_json") if isinstance(d.get("source_links_json"), dict) else {}
    return d


def _parse_optimization_alert_json_fields(d: dict) -> dict:
    for field in _OPTIMIZATION_ALERT_JSON_FIELDS:
        _parse_json_field(d, field)
    d["evidence"] = d.get("evidence_json") if isinstance(d.get("evidence_json"), dict) else {}
    return d


def ensure_default_optimization_mission() -> dict:
    """Create the default command-center mission if it does not already exist."""

    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute(
            "SELECT * FROM optimization_missions WHERE name = ? ORDER BY id LIMIT 1",
            (DEFAULT_OPTIMIZATION_MISSION_NAME,),
        ).fetchone()
        if not row:
            cur = conn.execute(
                "INSERT INTO optimization_missions "
                "(name, status, schedule_label, scenario_json, source_config_json, thresholds_json, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    DEFAULT_OPTIMIZATION_MISSION_NAME,
                    "active",
                    DEFAULT_OPTIMIZATION_SCHEDULE,
                    json.dumps(_default_optimization_scenario(), sort_keys=True),
                    json.dumps(_default_optimization_sources(), sort_keys=True),
                    json.dumps(_default_optimization_thresholds(), sort_keys=True),
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM optimization_missions WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _parse_optimization_mission_json_fields(_require_row_dict(row))


def get_optimization_missions(status: str | None = None) -> list[dict]:
    ensure_default_optimization_mission()
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM optimization_missions{where} ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'paused' THEN 1 ELSE 2 END, id",
            params,
        ).fetchall()
    return [_parse_optimization_mission_json_fields(row) for row in _rows_to_list(rows)]


def get_optimization_mission(mission_id: int | None = None) -> dict | None:
    if mission_id is None:
        return ensure_default_optimization_mission()
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM optimization_missions WHERE id = ?", (mission_id,)).fetchone()
    return _parse_optimization_mission_json_fields(_require_row_dict(row)) if row else None


def create_optimization_run(mission: dict, *, run_id: str | None = None, input_hash: str | None = None) -> dict:
    conn = _get_conn()
    now = _now()
    rid = run_id or f"opt-{uuid.uuid4().hex}"
    mission_id = int(mission["id"])
    mission_name = str(mission.get("name") or DEFAULT_OPTIMIZATION_MISSION_NAME)
    with _lock:
        conn.execute(
            "INSERT INTO optimization_runs "
            "(run_id, mission_id, mission_name, status, started_at, input_hash, summary_json, source_freshness_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (rid, mission_id, mission_name, "running", now, input_hash, json.dumps({}), json.dumps({})),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM optimization_runs WHERE run_id = ?", (rid,)).fetchone()
    return _parse_optimization_run_json_fields(_require_row_dict(row))


def complete_optimization_run(
    run_id: str,
    *,
    summary: dict | None = None,
    source_freshness: dict | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        conn.execute(
            "UPDATE optimization_runs SET status = 'completed', completed_at = ?, summary_json = ?, "
            "source_freshness_json = ?, input_hash = COALESCE(?, input_hash), output_hash = ? WHERE run_id = ?",
            (
                now,
                json.dumps(summary or {}, default=str, sort_keys=True),
                json.dumps(source_freshness or {}, default=str, sort_keys=True),
                input_hash,
                output_hash,
                run_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM optimization_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        raise ValueError(f"No optimization run with id {run_id}")
    return _parse_optimization_run_json_fields(_require_row_dict(row))


def fail_optimization_run(
    run_id: str,
    error: str,
    *,
    summary: dict | None = None,
    source_freshness: dict | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        conn.execute(
            "UPDATE optimization_runs SET status = 'failed', completed_at = ?, error = ?, summary_json = ?, "
            "source_freshness_json = ? WHERE run_id = ?",
            (
                now,
                error,
                json.dumps(summary or {}, default=str, sort_keys=True),
                json.dumps(source_freshness or {}, default=str, sort_keys=True),
                run_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM optimization_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        raise ValueError(f"No optimization run with id {run_id}")
    return _parse_optimization_run_json_fields(_require_row_dict(row))


def get_optimization_runs(mission_id: int | None = None, limit: int = 20) -> list[dict]:
    conn = _get_conn()
    safe_limit = max(1, min(int(limit), 100))
    clauses: list[str] = []
    params: list[Any] = []
    if mission_id is not None:
        clauses.append("mission_id = ?")
        params.append(int(mission_id))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM optimization_runs{where} ORDER BY started_at DESC LIMIT ?",
            (*params, safe_limit),
        ).fetchall()
    return [_parse_optimization_run_json_fields(row) for row in _rows_to_list(rows)]


def get_optimization_run(run_id: str) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM optimization_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return None
    result = _parse_optimization_run_json_fields(_require_row_dict(row))
    result["snapshots"] = get_optimization_snapshots(run_id=run_id)
    return result


def get_latest_successful_optimization_run(mission_id: int, *, before_run_id: str | None = None) -> dict | None:
    conn = _get_conn()
    params: list[Any] = [int(mission_id)]
    before_clause = ""
    if before_run_id:
        before_clause = (
            " AND started_at < COALESCE((SELECT started_at FROM optimization_runs WHERE run_id = ?), datetime('now'))"
        )
        params.append(before_run_id)
    with _lock:
        row = conn.execute(
            "SELECT * FROM optimization_runs WHERE mission_id = ? AND status = 'completed'"
            f"{before_clause} ORDER BY started_at DESC LIMIT 1",
            params,
        ).fetchone()
    return _parse_optimization_run_json_fields(_require_row_dict(row)) if row else None


def create_optimization_action_snapshot(record: dict) -> dict:
    conn = _get_conn()
    now = _now()
    ticker = str(record.get("ticker") or "").upper() or None
    with _lock:
        cur = conn.execute(
            "INSERT INTO optimization_action_snapshots "
            "(run_id, mission_id, ticker, asset, direction, action, conviction_band, priority_score, confidence, "
            "gate_status, severity, state_hash, evidence_json, source_links_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record["run_id"],
                int(record["mission_id"]),
                ticker,
                record.get("asset"),
                record.get("direction"),
                record["action"],
                record.get("conviction_band"),
                record.get("priority_score"),
                record.get("confidence"),
                record.get("gate_status"),
                record.get("severity"),
                record["state_hash"],
                json.dumps(record.get("evidence") or {}, default=str, sort_keys=True),
                json.dumps(record.get("source_links") or {}, default=str, sort_keys=True),
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM optimization_action_snapshots WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _parse_optimization_snapshot_json_fields(_require_row_dict(row))


def get_optimization_snapshots(
    *,
    run_id: str | None = None,
    mission_id: int | None = None,
    ticker: str | None = None,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if mission_id is not None:
        clauses.append("mission_id = ?")
        params.append(int(mission_id))
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM optimization_action_snapshots{where} ORDER BY priority_score DESC, ticker",
            params,
        ).fetchall()
    return [_parse_optimization_snapshot_json_fields(row) for row in _rows_to_list(rows)]


def get_optimization_snapshot(snapshot_id: int | None) -> dict | None:
    if snapshot_id is None:
        return None
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM optimization_action_snapshots WHERE id = ?", (int(snapshot_id),)).fetchone()
    return _parse_optimization_snapshot_json_fields(_require_row_dict(row)) if row else None


def create_optimization_alert(record: dict) -> dict:
    conn = _get_conn()
    now = _now()
    ticker = str(record.get("ticker") or "").upper() or None
    with _lock:
        cur = conn.execute(
            "INSERT INTO optimization_alerts "
            "(mission_id, run_id, ticker, alert_type, severity, status, previous_snapshot_id, current_snapshot_id, "
            "change_summary, evidence_json, approval_id, recommendation_id, action_item_approval_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                int(record["mission_id"]),
                record["run_id"],
                ticker,
                record.get("alert_type") or "action_state_changed",
                record.get("severity") or "normal",
                record.get("status") or "open",
                record.get("previous_snapshot_id"),
                record.get("current_snapshot_id"),
                record["change_summary"],
                json.dumps(record.get("evidence") or {}, default=str, sort_keys=True),
                record.get("approval_id"),
                record.get("recommendation_id"),
                record.get("action_item_approval_id"),
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM optimization_alerts WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _hydrate_optimization_alert(_parse_optimization_alert_json_fields(_require_row_dict(row)))


def update_optimization_alert_links(
    alert_id: int,
    *,
    approval_id: int | None = None,
    recommendation_id: int | None = None,
    action_item_approval_id: int | None = None,
) -> dict:
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE optimization_alerts SET approval_id = COALESCE(?, approval_id), "
            "recommendation_id = COALESCE(?, recommendation_id), "
            "action_item_approval_id = COALESCE(?, action_item_approval_id) WHERE id = ?",
            (approval_id, recommendation_id, action_item_approval_id, int(alert_id)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM optimization_alerts WHERE id = ?", (int(alert_id),)).fetchone()
    if not row:
        raise ValueError(f"No optimization alert with id {alert_id}")
    return _hydrate_optimization_alert(_parse_optimization_alert_json_fields(_require_row_dict(row)))


def _hydrate_optimization_alert(alert: dict) -> dict:
    alert["previous_snapshot"] = get_optimization_snapshot(alert.get("previous_snapshot_id"))
    alert["current_snapshot"] = get_optimization_snapshot(alert.get("current_snapshot_id"))
    return alert


def get_optimization_alerts(
    *,
    status: str | None = None,
    mission_id: int | None = None,
    limit: int = 50,
) -> list[dict]:
    conn = _get_conn()
    safe_limit = max(1, min(int(limit), 200))
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if mission_id is not None:
        clauses.append("mission_id = ?")
        params.append(int(mission_id))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM optimization_alerts{where} ORDER BY "
            "CASE severity WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END, "
            "created_at DESC LIMIT ?",
            (*params, safe_limit),
        ).fetchall()
    return [_hydrate_optimization_alert(_parse_optimization_alert_json_fields(row)) for row in _rows_to_list(rows)]


def dismiss_optimization_alert(alert_id: int, note: str | None = None) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        conn.execute(
            "UPDATE optimization_alerts SET status = 'dismissed', dismissed_at = ?, dismissed_note = ? WHERE id = ?",
            (now, note, int(alert_id)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM optimization_alerts WHERE id = ?", (int(alert_id),)).fetchone()
    if not row:
        raise ValueError(f"No optimization alert with id {alert_id}")
    return _hydrate_optimization_alert(_parse_optimization_alert_json_fields(_require_row_dict(row)))


# ---------------------------------------------------------------------------
# Investment Ideas / Watchlist
# ---------------------------------------------------------------------------

_UNSET = object()
_IDEA_JSON_FIELDS = ("tags_json", "metadata_json")
_IDEA_EVALUATION_JSON_FIELDS = (
    "factor_scores_json",
    "missing_information_json",
    "data_quality_json",
    "evidence_json",
    "disconfirming_evidence_json",
    "portfolio_fit_json",
    "recommendation_record_json",
    "raw_result_json",
)


def _normalize_investment_idea_status(status: str | None) -> str:
    normalized = str(status or "watching").strip().lower()
    if normalized not in INVESTMENT_IDEA_STATUSES:
        raise ValueError(f"Invalid investment idea status: {status}")
    return normalized


def _normalize_idea_action(action: str | None) -> str:
    normalized = str(action or "watch").strip().lower()
    if normalized not in IDEA_RECOMMENDATION_ACTIONS:
        raise ValueError(f"Invalid idea recommendation action: {action}")
    return normalized


def _normalize_idea_recommendation_status(status: str | None) -> str:
    normalized = str(status or "clear").strip().lower()
    if normalized not in IDEA_RECOMMENDATION_STATUSES:
        return "review_required"
    return normalized


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_investment_idea_json_fields(d: dict) -> dict:
    for field in _IDEA_JSON_FIELDS:
        _parse_json_field(d, field)
    d["tags"] = d.get("tags_json") if isinstance(d.get("tags_json"), list) else []
    d["metadata"] = d.get("metadata_json") if isinstance(d.get("metadata_json"), dict) else {}
    return d


def _parse_idea_evaluation_json_fields(d: dict) -> dict:
    for field in _IDEA_EVALUATION_JSON_FIELDS:
        _parse_json_field(d, field)
    d["factor_scores"] = d.get("factor_scores_json") if isinstance(d.get("factor_scores_json"), dict) else {}
    d["missing_information"] = (
        d.get("missing_information_json") if isinstance(d.get("missing_information_json"), list) else []
    )
    d["data_quality"] = d.get("data_quality_json") if isinstance(d.get("data_quality_json"), dict) else {}
    d["evidence"] = d.get("evidence_json") if isinstance(d.get("evidence_json"), list) else []
    d["disconfirming_evidence"] = (
        d.get("disconfirming_evidence_json") if isinstance(d.get("disconfirming_evidence_json"), list) else []
    )
    d["portfolio_fit"] = d.get("portfolio_fit_json") if isinstance(d.get("portfolio_fit_json"), dict) else {}
    d["recommendation_record"] = (
        d.get("recommendation_record_json") if isinstance(d.get("recommendation_record_json"), dict) else {}
    )
    d["raw_result"] = d.get("raw_result_json") if isinstance(d.get("raw_result_json"), dict) else {}
    return d


def create_investment_idea(
    ticker: str,
    *,
    company_name: str | None = None,
    user_notes: str | None = None,
    tags: list[str] | None = None,
    status: str = "watching",
    source_type: str = "user",
    source_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    _guard_legacy_domain_write("core_db.create_investment_idea")
    normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        raise ValueError("Ticker cannot be empty.")
    normalized_status = _normalize_investment_idea_status(status)
    now = _now()
    conn = _get_conn()
    with _lock:
        existing = conn.execute(
            "SELECT * FROM investment_ideas WHERE ticker = ? AND status != 'archived' ORDER BY id DESC LIMIT 1",
            (normalized_ticker,),
        ).fetchone()
        if existing:
            row = _parse_investment_idea_json_fields(_require_row_dict(existing))
            updates: dict[str, Any] = {"updated_at": now}
            if company_name is not None:
                updates["company_name"] = str(company_name).strip() or None
            if user_notes is not None:
                updates["user_notes"] = str(user_notes)
            if tags is not None:
                updates["tags_json"] = json.dumps([str(tag).strip() for tag in tags if str(tag).strip()])
            if normalized_status != "watching" or row.get("status") == "watching":
                updates["status"] = normalized_status
            if metadata is not None:
                updates["metadata_json"] = json.dumps(metadata, default=str)
            set_sql = ", ".join(f"{key} = ?" for key in updates)
            conn.execute(
                f"UPDATE investment_ideas SET {set_sql} WHERE id = ?",
                (*updates.values(), int(row["id"])),
            )
            conn.commit()
            refreshed = conn.execute("SELECT * FROM investment_ideas WHERE id = ?", (int(row["id"]),)).fetchone()
            return _parse_investment_idea_json_fields(_require_row_dict(refreshed))

        cur = conn.execute(
            "INSERT INTO investment_ideas "
            "(ticker, company_name, status, user_notes, tags_json, created_at, updated_at, source_type, source_id, metadata_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                normalized_ticker,
                str(company_name).strip() if company_name else None,
                normalized_status,
                str(user_notes or ""),
                json.dumps([str(tag).strip() for tag in tags or [] if str(tag).strip()]),
                now,
                now,
                source_type,
                source_id,
                json.dumps(metadata or {}, default=str),
            ),
        )
        conn.commit()
        inserted = conn.execute("SELECT * FROM investment_ideas WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _parse_investment_idea_json_fields(_require_row_dict(inserted))


def update_investment_idea(
    idea_id: int,
    *,
    ticker: str | object = _UNSET,
    company_name: str | None | object = _UNSET,
    status: str | object = _UNSET,
    user_notes: str | object = _UNSET,
    tags: list[str] | object = _UNSET,
    latest_job_id: str | None | object = _UNSET,
    latest_evaluation_id: int | None | object = _UNSET,
    accepted_recommendation_id: int | None | object = _UNSET,
    metadata: dict[str, Any] | object = _UNSET,
) -> dict:
    _guard_legacy_domain_write("core_db.update_investment_idea")
    conn = _get_conn()
    updates: dict[str, Any] = {"updated_at": _now()}
    if ticker is not _UNSET:
        normalized_ticker = str(ticker or "").strip().upper()
        if not normalized_ticker:
            raise ValueError("Ticker cannot be empty.")
        updates["ticker"] = normalized_ticker
    if company_name is not _UNSET:
        updates["company_name"] = str(company_name).strip() if company_name else None
    if status is not _UNSET:
        updates["status"] = _normalize_investment_idea_status(cast(str, status))
    if user_notes is not _UNSET:
        updates["user_notes"] = str(user_notes or "")
    if tags is not _UNSET:
        updates["tags_json"] = json.dumps([str(tag).strip() for tag in cast(list[str], tags) if str(tag).strip()])
    if latest_job_id is not _UNSET:
        updates["latest_job_id"] = latest_job_id
    if latest_evaluation_id is not _UNSET:
        updates["latest_evaluation_id"] = latest_evaluation_id
    if accepted_recommendation_id is not _UNSET:
        updates["accepted_recommendation_id"] = accepted_recommendation_id
    if metadata is not _UNSET:
        updates["metadata_json"] = json.dumps(metadata if isinstance(metadata, dict) else {}, default=str)
    with _lock:
        row = conn.execute("SELECT * FROM investment_ideas WHERE id = ?", (int(idea_id),)).fetchone()
        if not row:
            raise ValueError(f"No investment idea with id {idea_id}")
        set_sql = ", ".join(f"{key} = ?" for key in updates)
        conn.execute(f"UPDATE investment_ideas SET {set_sql} WHERE id = ?", (*updates.values(), int(idea_id)))
        conn.commit()
        updated = conn.execute("SELECT * FROM investment_ideas WHERE id = ?", (int(idea_id),)).fetchone()
    return _parse_investment_idea_json_fields(_require_row_dict(updated))


def archive_investment_idea(idea_id: int) -> dict:
    return update_investment_idea(idea_id, status="archived")


def get_investment_idea(idea_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM investment_ideas WHERE id = ?", (int(idea_id),)).fetchone()
    return _parse_investment_idea_json_fields(_require_row_dict(row)) if row else None


def list_investment_ideas(
    *,
    status: str | None = None,
    include_archived: bool = False,
    limit: int = 200,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(_normalize_investment_idea_status(status))
    elif not include_archived:
        clauses.append("status != 'archived'")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    safe_limit = max(1, min(int(limit), 500))
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM investment_ideas{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, safe_limit),
        ).fetchall()
    return [_parse_investment_idea_json_fields(d) for d in _rows_to_list(rows)]


def create_idea_evaluation(
    idea_id: int,
    result: dict[str, Any],
    *,
    job_id: str | None = None,
) -> dict:
    _guard_legacy_domain_write("core_db.create_idea_evaluation")
    idea = get_investment_idea(int(idea_id))
    if not idea:
        raise ValueError(f"No investment idea with id {idea_id}")
    action = _normalize_idea_action(cast(str | None, result.get("action")))
    evaluated_at = str(result.get("evaluated_at") or _now())
    created_at = _now()
    factor_scores = result.get("factor_scores") if isinstance(result.get("factor_scores"), dict) else {}
    missing_information = (
        result.get("missing_information") if isinstance(result.get("missing_information"), list) else []
    )
    data_quality = result.get("data_quality") if isinstance(result.get("data_quality"), dict) else {}
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    disconfirming = (
        result.get("disconfirming_evidence") if isinstance(result.get("disconfirming_evidence"), list) else []
    )
    portfolio_fit = result.get("portfolio_fit") if isinstance(result.get("portfolio_fit"), dict) else {}
    recommendation_record = (
        result.get("recommendation_record") if isinstance(result.get("recommendation_record"), dict) else {}
    )
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO idea_evaluations "
            "(idea_id, ticker, job_id, evaluated_at, action, recommendation_status, score, confidence, "
            "thesis_statement, rationale, factor_scores_json, missing_information_json, data_quality_json, "
            "evidence_json, disconfirming_evidence_json, catalyst, invalidation, portfolio_fit_json, "
            "recommendation_record_json, recommendation_id, approval_id, action_approval_id, raw_result_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                int(idea_id),
                str(idea["ticker"]).upper(),
                job_id or result.get("job_id"),
                evaluated_at,
                action,
                _normalize_idea_recommendation_status(cast(str | None, result.get("recommendation_status"))),
                _optional_float(result.get("score")),
                _optional_float(result.get("confidence")),
                result.get("thesis_statement"),
                str(result.get("rationale") or ""),
                json.dumps(factor_scores, default=str),
                json.dumps(missing_information, default=str),
                json.dumps(data_quality, default=str),
                json.dumps(evidence, default=str),
                json.dumps(disconfirming, default=str),
                result.get("catalyst"),
                result.get("invalidation"),
                json.dumps(portfolio_fit, default=str),
                json.dumps(recommendation_record, default=str),
                result.get("recommendation_id"),
                result.get("approval_id"),
                result.get("action_approval_id"),
                json.dumps(result, default=str),
                created_at,
            ),
        )
        evaluation_id = cast(int, cur.lastrowid)
        next_status = "ready_for_review"
        if str(idea.get("status") or "") in {"accepted", "rejected", "archived"}:
            next_status = str(idea["status"])
        conn.execute(
            "UPDATE investment_ideas SET latest_evaluation_id = ?, latest_job_id = ?, status = ?, updated_at = ? WHERE id = ?",
            (evaluation_id, job_id or result.get("job_id"), next_status, created_at, int(idea_id)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM idea_evaluations WHERE id = ?", (evaluation_id,)).fetchone()
    return _parse_idea_evaluation_json_fields(_require_row_dict(row))


def get_idea_evaluation(evaluation_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM idea_evaluations WHERE id = ?", (int(evaluation_id),)).fetchone()
    return _parse_idea_evaluation_json_fields(_require_row_dict(row)) if row else None


def get_idea_evaluations(idea_id: int, *, limit: int = 20) -> list[dict]:
    conn = _get_conn()
    safe_limit = max(1, min(int(limit), 100))
    with _lock:
        rows = conn.execute(
            "SELECT * FROM idea_evaluations WHERE idea_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(idea_id), safe_limit),
        ).fetchall()
    return [_parse_idea_evaluation_json_fields(d) for d in _rows_to_list(rows)]


def mark_idea_evaluation_accepted(
    evaluation_id: int,
    *,
    recommendation_id: int,
    action_approval_id: int | None = None,
    accepted_by: str | None = None,
) -> dict:
    _guard_legacy_domain_write("core_db.mark_idea_evaluation_accepted")
    evaluation = get_idea_evaluation(evaluation_id)
    if not evaluation:
        raise ValueError(f"No idea evaluation with id {evaluation_id}")
    accepted_at = _now()
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE idea_evaluations SET recommendation_id = ?, action_approval_id = ?, accepted_at = ?, accepted_by = ? WHERE id = ?",
            (int(recommendation_id), action_approval_id, accepted_at, accepted_by, int(evaluation_id)),
        )
        conn.execute(
            "UPDATE investment_ideas SET status = 'accepted', accepted_recommendation_id = ?, updated_at = ? WHERE id = ?",
            (int(recommendation_id), accepted_at, int(evaluation["idea_id"])),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM idea_evaluations WHERE id = ?", (int(evaluation_id),)).fetchone()
    return _parse_idea_evaluation_json_fields(_require_row_dict(row))


# ---------------------------------------------------------------------------
# Action Items
# ---------------------------------------------------------------------------


def create_action_item(
    description: str,
    action_type: str = "review",
    *,
    ticker: str | None = None,
    urgency: str = "normal",
    source_type: str = "user",
    source_id: str | None = None,
) -> dict:
    _guard_legacy_domain_write("core_db.create_action_item")
    conn = _get_conn()
    now = _now()
    with _lock:
        cur = conn.execute(
            "INSERT INTO action_items (ticker, action_type, description, urgency, source_type, source_id, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (ticker.upper() if ticker else None, action_type, description, urgency, source_type, source_id, now),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper() if ticker else None,
        "action_type": action_type,
        "description": description,
        "urgency": urgency,
        "status": "open",
        "source_type": source_type,
        "source_id": source_id,
        "created_at": now,
        "completed_at": None,
        "resolution_note": None,
    }


def create_action_item_once(
    description: str,
    action_type: str = "review",
    *,
    ticker: str | None = None,
    urgency: str = "normal",
    source_type: str = "workflow",
    source_id: str | None = None,
) -> dict:
    conn = _get_conn()
    normalized_ticker = ticker.upper() if ticker else None
    if source_id:
        with _lock:
            row = conn.execute(
                "SELECT * FROM action_items WHERE source_type = ? AND source_id = ? AND COALESCE(ticker, '') = COALESCE(?, '') AND description = ? ORDER BY id DESC LIMIT 1",
                (source_type, source_id, normalized_ticker, description),
            ).fetchone()
        if row:
            return _require_row_dict(row)
    return create_action_item(
        description=description,
        action_type=action_type,
        ticker=normalized_ticker,
        urgency=urgency,
        source_type=source_type,
        source_id=source_id,
    )


def get_action_items(
    status: str | None = None,
    ticker: str | None = None,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM action_items{where} ORDER BY CASE urgency WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END, created_at DESC",
            params,
        ).fetchall()
    return _rows_to_list(rows)


def complete_action_item(item_id: int, resolution_note: str = "") -> dict:
    _guard_legacy_domain_write("core_db.complete_action_item")
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM action_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"No action item with id {item_id}")
        conn.execute(
            "UPDATE action_items SET status = 'completed', completed_at = ?, resolution_note = ? WHERE id = ?",
            (now, resolution_note or None, item_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM action_items WHERE id = ?", (item_id,)).fetchone()
    return _require_row_dict(updated)


def dismiss_action_item(item_id: int) -> dict:
    _guard_legacy_domain_write("core_db.dismiss_action_item")
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM action_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"No action item with id {item_id}")
        conn.execute(
            "UPDATE action_items SET status = 'dismissed', completed_at = ? WHERE id = ?",
            (now, item_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM action_items WHERE id = ?", (item_id,)).fetchone()
    return _require_row_dict(updated)


# ---------------------------------------------------------------------------
# Watch Triggers
# ---------------------------------------------------------------------------


def create_watch_trigger(
    condition: str,
    trigger_type: str = "custom",
    *,
    ticker: str | None = None,
    source_type: str = "user",
    source_id: str | None = None,
    expires_at: str | None = None,
    definition: dict | None = None,
) -> dict:
    _guard_legacy_domain_write("core_db.create_watch_trigger")
    conn = _get_conn()
    now = _now()
    trigger_type = str(trigger_type or "custom").strip().lower()
    if trigger_type not in WATCH_TRIGGER_TYPES:
        raise ValueError(f"Invalid watch trigger type: {trigger_type}")
    definition_json = json.dumps(definition, default=str) if definition else None
    with _lock:
        cur = conn.execute(
            "INSERT INTO watch_triggers (ticker, trigger_type, condition, source_type, source_id, created_at, expires_at, definition_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                ticker.upper() if ticker else None,
                trigger_type,
                condition,
                source_type,
                source_id,
                now,
                expires_at,
                definition_json,
            ),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper() if ticker else None,
        "trigger_type": trigger_type,
        "condition": condition,
        "status": "active",
        "source_type": source_type,
        "source_id": source_id,
        "created_at": now,
        "fired_at": None,
        "expires_at": expires_at,
        "definition_json": definition,
        "definition": definition,
        "last_checked_at": None,
        "last_result_json": None,
        "last_result": None,
        "last_evidence": None,
    }


def create_watch_trigger_once(
    condition: str,
    trigger_type: str = "custom",
    *,
    ticker: str | None = None,
    source_type: str = "workflow",
    source_id: str | None = None,
    expires_at: str | None = None,
    definition: dict | None = None,
) -> dict:
    conn = _get_conn()
    normalized_ticker = ticker.upper() if ticker else None
    if source_id:
        with _lock:
            row = conn.execute(
                "SELECT * FROM watch_triggers WHERE source_type = ? AND source_id = ? AND COALESCE(ticker, '') = COALESCE(?, '') AND condition = ? ORDER BY id DESC LIMIT 1",
                (source_type, source_id, normalized_ticker, condition),
            ).fetchone()
        if row:
            return _parse_watch_trigger_json_fields(_require_row_dict(row))
    return create_watch_trigger(
        condition=condition,
        trigger_type=trigger_type,
        ticker=normalized_ticker,
        source_type=source_type,
        source_id=source_id,
        expires_at=expires_at,
        definition=definition,
    )


def _parse_watch_trigger_json_fields(d: dict) -> dict:
    _parse_json_field(d, "definition_json")
    _parse_json_field(d, "last_result_json")
    d["definition"] = d.get("definition_json")
    d["last_result"] = d.get("last_result_json")
    return d


def get_watch_triggers(
    status: str | None = None,
    ticker: str | None = None,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM watch_triggers{where} ORDER BY created_at DESC",
            params,
        ).fetchall()
    return [_parse_watch_trigger_json_fields(d) for d in _rows_to_list(rows)]


def update_watch_trigger_check(
    trigger_id: int,
    *,
    result: dict | None = None,
    evidence: str | None = None,
) -> dict:
    _guard_legacy_domain_write("core_db.update_watch_trigger_check")
    conn = _get_conn()
    now = _now()
    result_json = json.dumps(result or {}, default=str)
    with _lock:
        row = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
        if not row:
            raise ValueError(f"No watch trigger with id {trigger_id}")
        conn.execute(
            "UPDATE watch_triggers SET last_checked_at = ?, last_result_json = ?, last_evidence = ? WHERE id = ?",
            (now, result_json, evidence, trigger_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
    return _parse_watch_trigger_json_fields(_require_row_dict(updated))


def update_watch_trigger_definition(trigger_id: int, definition: dict) -> dict:
    _guard_legacy_domain_write("core_db.update_watch_trigger_definition")
    conn = _get_conn()
    definition_json = json.dumps(definition, default=str)
    with _lock:
        row = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
        if not row:
            raise ValueError(f"No watch trigger with id {trigger_id}")
        conn.execute(
            "UPDATE watch_triggers SET definition_json = ? WHERE id = ?",
            (definition_json, trigger_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
    return _parse_watch_trigger_json_fields(_require_row_dict(updated))


def fire_watch_trigger(
    trigger_id: int,
    *,
    result: dict | None = None,
    evidence: str | None = None,
) -> dict:
    _guard_legacy_domain_write("core_db.fire_watch_trigger")
    conn = _get_conn()
    now = _now()
    result_json = json.dumps(result or {}, default=str)
    with _lock:
        row = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
        if not row:
            raise ValueError(f"No watch trigger with id {trigger_id}")
        conn.execute(
            "UPDATE watch_triggers SET status = 'fired', fired_at = ?, last_checked_at = ?, last_result_json = ?, last_evidence = ? WHERE id = ?",
            (now, now, result_json, evidence, trigger_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
    return _parse_watch_trigger_json_fields(_require_row_dict(updated))


def cancel_watch_trigger(trigger_id: int) -> dict:
    _guard_legacy_domain_write("core_db.cancel_watch_trigger")
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
        if not row:
            raise ValueError(f"No watch trigger with id {trigger_id}")
        conn.execute(
            "UPDATE watch_triggers SET status = 'cancelled' WHERE id = ?",
            (trigger_id,),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM watch_triggers WHERE id = ?", (trigger_id,)).fetchone()
    return _parse_watch_trigger_json_fields(_require_row_dict(updated))


# ---------------------------------------------------------------------------
# Thesis Claims
# ---------------------------------------------------------------------------

_THESIS_CLAIM_JSON_FIELDS = (
    "source_requirements_json",
    "linked_catalyst_ids_json",
    "linked_kill_condition_ids_json",
)

_THESIS_CLAIM_STATUSES = {"active", "supported", "challenged", "disconfirmed", "retired"}


def normalize_source_requirements(value: Any) -> list[dict[str, Any]]:
    """Normalize legacy string source requirements into typed requirement objects."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            normalized.append(
                {
                    "type": "custom",
                    "description": text,
                    "required": True,
                    "freshness_days": None,
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        req_type = str(item.get("type") or "custom").strip() or "custom"
        description = str(item.get("description") or item.get("label") or req_type).strip()
        required_raw = item.get("required", True)
        if isinstance(required_raw, str):
            required = required_raw.strip().lower() not in {"false", "0", "no", "optional"}
        else:
            required = bool(required_raw)
        freshness_raw = item.get("freshness_days")
        freshness_days = None
        if freshness_raw is not None and freshness_raw != "":
            try:
                freshness_days = max(0, int(freshness_raw))
            except (TypeError, ValueError):
                freshness_days = None
        normalized.append(
            {
                "type": req_type,
                "description": description,
                "required": required,
                "freshness_days": freshness_days,
            }
        )
    return normalized


def _normalize_thesis_claim_status(status: Any) -> str:
    normalized = str(status or "active").strip().lower()
    if normalized not in _THESIS_CLAIM_STATUSES:
        raise ValueError(f"Invalid thesis claim status: {normalized}")
    return normalized


def _normalize_claim_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    confidence = float(value)
    if confidence < 0 or confidence > 1:
        raise ValueError("Thesis claim confidence must be between 0 and 1")
    return confidence


def _normalize_claim_id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    ids: list[int] = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def _parse_thesis_claim_json_fields(d: dict) -> dict:
    for field in _THESIS_CLAIM_JSON_FIELDS:
        _parse_json_field(d, field)
    d["source_requirements_json"] = normalize_source_requirements(d.get("source_requirements_json"))
    d["linked_catalyst_ids_json"] = _normalize_claim_id_list(d.get("linked_catalyst_ids_json"))
    d["linked_kill_condition_ids_json"] = _normalize_claim_id_list(d.get("linked_kill_condition_ids_json"))
    # Friendly aliases for API/UI callers while retaining the DB column-shaped keys.
    d["source_requirements"] = d["source_requirements_json"]
    d["linked_catalyst_ids"] = d["linked_catalyst_ids_json"]
    d["linked_kill_condition_ids"] = d["linked_kill_condition_ids_json"]
    return d


def create_thesis_claim(record: dict) -> dict:
    _guard_legacy_domain_write("core_db.create_thesis_claim")
    conn = _get_conn()
    now = _now()
    ticker = str(record["ticker"]).upper()
    params = {
        "ticker": ticker,
        "claim": record["claim"],
        "expected_evidence": record.get("expected_evidence"),
        "disconfirming_evidence": record.get("disconfirming_evidence"),
        "source_requirements_json": _encode_json(normalize_source_requirements(record.get("source_requirements", []))),
        "cadence": record.get("cadence"),
        "confidence": _normalize_claim_confidence(record.get("confidence")),
        "status": _normalize_thesis_claim_status(record.get("status")),
        "linked_catalyst_ids_json": _encode_json(_normalize_claim_id_list(record.get("linked_catalyst_ids", []))),
        "linked_kill_condition_ids_json": _encode_json(
            _normalize_claim_id_list(record.get("linked_kill_condition_ids", []))
        ),
        "source_type": record.get("source_type") or "user",
        "source_id": record.get("source_id"),
        "created_at": record.get("created_at") or now,
        "updated_at": now,
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    with _lock:
        cur = conn.execute(
            f"INSERT INTO thesis_claims ({columns}) VALUES ({placeholders})",
            tuple(params.values()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM thesis_claims WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _parse_thesis_claim_json_fields(_require_row_dict(row))


def create_thesis_claim_once(record: dict) -> dict:
    conn = _get_conn()
    ticker = str(record["ticker"]).upper()
    claim = str(record["claim"])
    source_type = record.get("source_type") or "workflow"
    source_id = record.get("source_id")
    if source_id:
        with _lock:
            row = conn.execute(
                "SELECT * FROM thesis_claims WHERE ticker = ? AND claim = ? AND source_type = ? AND source_id = ? ORDER BY id DESC LIMIT 1",
                (ticker, claim, source_type, source_id),
            ).fetchone()
        if row:
            return _parse_thesis_claim_json_fields(_require_row_dict(row))
    return create_thesis_claim({**record, "ticker": ticker, "source_type": source_type})


def get_thesis_claim(claim_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM thesis_claims WHERE id = ?", (claim_id,)).fetchone()
    if not row:
        return None
    return _parse_thesis_claim_json_fields(_require_row_dict(row))


def get_thesis_claims(
    ticker: str | None = None,
    status: str | None = None,
    limit: int = 100,
    *,
    source_type: str | None = None,
    source_id: str | None = None,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if status:
        clauses.append("status = ?")
        params.append(status)
    if source_type:
        clauses.append("source_type = ?")
        params.append(source_type)
    if source_id:
        clauses.append("source_id = ?")
        params.append(source_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    safe_limit = max(1, min(int(limit), 500))
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM thesis_claims{where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*params, safe_limit),
        ).fetchall()
    return [_parse_thesis_claim_json_fields(d) for d in _rows_to_list(rows)]


def delete_thesis_claims_by_ticker(
    ticker: str,
    *,
    source_type: str | None = None,
    source_id: str | None = None,
    exclude_ids: list[int] | None = None,
) -> int:
    """Delete thesis claims for a ticker, optionally scoped to a source."""
    _guard_legacy_domain_write("core_db.delete_thesis_claims_by_ticker")
    conn = _get_conn()
    clauses = ["ticker = ?"]
    params: list[Any] = [ticker.upper()]
    if source_type:
        clauses.append("source_type = ?")
        params.append(source_type)
    if source_id:
        clauses.append("source_id = ?")
        params.append(source_id)
    exclude_ids = _normalize_claim_id_list(exclude_ids or [])
    if exclude_ids:
        placeholders = ", ".join("?" for _ in exclude_ids)
        clauses.append(f"id NOT IN ({placeholders})")
        params.extend(exclude_ids)
    with _lock:
        cur = conn.execute(f"DELETE FROM thesis_claims WHERE {' AND '.join(clauses)}", params)
        conn.commit()
    return cur.rowcount


def update_thesis_claim(claim_id: int, updates: dict) -> dict:
    _guard_legacy_domain_write("core_db.update_thesis_claim")
    allowed = {
        "claim",
        "expected_evidence",
        "disconfirming_evidence",
        "cadence",
        "confidence",
        "status",
        "source_requirements",
        "linked_catalyst_ids",
        "linked_kill_condition_ids",
    }
    params: dict[str, Any] = {}
    for key, value in updates.items():
        if key not in allowed:
            continue
        if key == "source_requirements":
            params["source_requirements_json"] = _encode_json(normalize_source_requirements(value))
        elif key == "linked_catalyst_ids":
            params["linked_catalyst_ids_json"] = _encode_json(_normalize_claim_id_list(value))
        elif key == "linked_kill_condition_ids":
            params["linked_kill_condition_ids_json"] = _encode_json(_normalize_claim_id_list(value))
        elif key == "confidence":
            params[key] = _normalize_claim_confidence(value)
        elif key == "status":
            params[key] = _normalize_thesis_claim_status(value)
        else:
            params[key] = value
    if not params:
        raise ValueError("No valid thesis claim updates supplied")
    params["updated_at"] = _now()
    conn = _get_conn()
    set_clause = ", ".join(f"{key} = ?" for key in params)
    with _lock:
        row = conn.execute("SELECT * FROM thesis_claims WHERE id = ?", (claim_id,)).fetchone()
        if not row:
            raise ValueError(f"No thesis claim with id {claim_id}")
        conn.execute(
            f"UPDATE thesis_claims SET {set_clause} WHERE id = ?",
            (*params.values(), claim_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM thesis_claims WHERE id = ?", (claim_id,)).fetchone()
    return _parse_thesis_claim_json_fields(_require_row_dict(updated))


# ---------------------------------------------------------------------------
# Research Notes
# ---------------------------------------------------------------------------


def create_research_note(
    title: str,
    content: str,
    *,
    ticker: str | None = None,
    note_type: str = "general",
    source_type: str = "user",
    source_id: str | None = None,
) -> dict:
    _guard_legacy_domain_write("core_db.create_research_note")
    conn = _get_conn()
    now = _now()
    with _lock:
        cur = conn.execute(
            "INSERT INTO research_notes (ticker, title, content, note_type, source_type, source_id, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (ticker.upper() if ticker else None, title, content, note_type, source_type, source_id, now),
        )
        conn.commit()
    return {
        "id": cur.lastrowid,
        "ticker": ticker.upper() if ticker else None,
        "title": title,
        "content": content,
        "note_type": note_type,
        "source_type": source_type,
        "source_id": source_id,
        "created_at": now,
    }


def create_research_note_once(
    title: str,
    content: str,
    *,
    ticker: str | None = None,
    note_type: str = "general",
    source_type: str = "workflow",
    source_id: str | None = None,
) -> dict:
    conn = _get_conn()
    normalized_ticker = ticker.upper() if ticker else None
    if source_id:
        with _lock:
            row = conn.execute(
                "SELECT * FROM research_notes WHERE source_type = ? AND source_id = ? AND COALESCE(ticker, '') = COALESCE(?, '') AND title = ? ORDER BY id DESC LIMIT 1",
                (source_type, source_id, normalized_ticker, title),
            ).fetchone()
        if row:
            return _require_row_dict(row)
    return create_research_note(
        title=title,
        content=content,
        ticker=normalized_ticker,
        note_type=note_type,
        source_type=source_type,
        source_id=source_id,
    )


def get_research_notes(
    ticker: str | None = None,
    limit: int = 20,
) -> list[dict]:
    conn = _get_conn()
    if ticker:
        with _lock:
            rows = conn.execute(
                "SELECT * FROM research_notes WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
                (ticker.upper(), limit),
            ).fetchall()
    else:
        with _lock:
            rows = conn.execute(
                "SELECT * FROM research_notes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return _rows_to_list(rows)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

_RECOMMENDATION_JSON_FIELDS = (
    "blocked_reasons_json",
    "what_changed_json",
    "evidence_json",
    "disconfirming_evidence_json",
    "alternatives_json",
    "opportunity_cost_json",
    "outcome_json",
    "source_quality_summary_json",
    "policy_gate_failures_json",
    "policy_gate_warnings_json",
    "policy_gate_disclosures_json",
    "trade_proposal_json",
    "risk_source_status_json",
    "risk_bindings_json",
)


def _encode_json(value: Any) -> str:
    return json.dumps(value if value is not None else [], default=str)


def _parse_recommendation_json_fields(d: dict) -> dict:
    for field in _RECOMMENDATION_JSON_FIELDS:
        _parse_json_field(d, field)
    if "risk_source_status_json" in d:
        d["risk_source_status"] = d.get("risk_source_status_json")
    if "risk_bindings_json" in d:
        d["risk_bindings"] = d.get("risk_bindings_json")
    return d


def _parse_policy_gate_result_row(row: Any) -> dict:
    d = _require_row_dict(row)
    _parse_json_field(d, "result_json")
    d["review_required"] = bool(d.get("review_required"))
    d["override_acknowledged"] = bool(d.get("override_acknowledged"))
    return d


def create_policy_gate_result(
    result: dict,
    *,
    action_id: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    target_type: str | None = None,
    target_id: str | int | None = None,
    payload: dict | None = None,
) -> dict:
    from api import governance

    conn = _get_conn()
    created_at = result.get("evaluated_at") or _now()
    decision = str(result.get("decision") or "error")
    params = {
        "created_at": created_at,
        "decision": decision,
        "review_required": 1 if result.get("review_required") else 0,
        "override_acknowledged": 1 if result.get("override_acknowledged") else 0,
        "account_id": result.get("account_id"),
        "portfolio_id": result.get("portfolio_id"),
        "policy_id": result.get("policy_id"),
        "mandate_id": result.get("mandate_id"),
        "action_id": action_id or result.get("action_id"),
        "source_type": source_type,
        "source_id": source_id,
        "target_type": target_type,
        "target_id": str(target_id) if target_id is not None else None,
        "payload_hash": _json_hash(payload) if payload is not None else None,
        "result_json": json.dumps(result, default=str),
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    with _lock:
        try:
            cur = conn.execute(
                f"INSERT INTO policy_gate_results ({columns}) VALUES ({placeholders})",
                tuple(params.values()),
            )
            result_id = cast(int, cur.lastrowid)
            lineage_root_id = governance.lineage_root(governance.REF_POLICY_GATE_RESULT, result_id)
            provenance_event_id = governance.deterministic_id("pv:policy_gate_result", result_id)
            bundle = {
                "lineage_root_id": lineage_root_id,
                "provenance_events": [
                    governance.provenance_event(
                        event_id=provenance_event_id,
                        event_type="policy_gate_result",
                        event_name=governance.EVENT_POLICY_GATE_EVALUATED,
                        lineage_root_id=lineage_root_id,
                        output_value=result,
                        summary={
                            "policy_gate_result_id": result_id,
                            "decision": decision,
                            "review_required": bool(result.get("review_required")),
                            "action_id": action_id or result.get("action_id"),
                            "target_type": target_type,
                            "target_id": str(target_id) if target_id is not None else None,
                        },
                        metadata={
                            "payload_hash": params["payload_hash"],
                            "account_id": result.get("account_id"),
                            "portfolio_id": result.get("portfolio_id"),
                            "policy_id": result.get("policy_id"),
                            "mandate_id": result.get("mandate_id"),
                        },
                    )
                ],
                "audit_events": [
                    governance.audit_event(
                        action_name=governance.EVENT_POLICY_GATE_EVALUATED,
                        status=decision,
                        lineage_root_id=lineage_root_id,
                        object_refs=[{"type": governance.REF_POLICY_GATE_RESULT, "id": result_id}],
                        after_summary={
                            "policy_gate_result_id": result_id,
                            "decision": decision,
                            "review_required": bool(result.get("review_required")),
                            "payload_hash": params["payload_hash"],
                        },
                        source_lineage={
                            "source_type": source_type,
                            "source_id": source_id,
                            "target_type": target_type,
                            "target_id": str(target_id) if target_id is not None else None,
                        },
                    )
                ],
                "policy_gate_result_updates": [
                    {
                        "policy_gate_result_id": result_id,
                        "provenance_event_id": provenance_event_id,
                        "lineage_root_id": lineage_root_id,
                        "lineage_completeness": "complete",
                    }
                ],
            }
            if target_type and target_id is not None:
                bundle["provenance_links"] = [
                    governance.provenance_link(
                        event_id=provenance_event_id,
                        source_ref_type=governance.REF_POLICY_GATE_RESULT,
                        source_ref_id=result_id,
                        target_ref_type=str(target_type),
                        target_ref_id=str(target_id),
                        link_type="evaluated",
                        lineage_root_id=lineage_root_id,
                    )
                ]
            _materialize_governance_bundle_tx(conn, bundle)
            row = conn.execute("SELECT * FROM policy_gate_results WHERE id = ?", (result_id,)).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return _parse_policy_gate_result_row(row)


def get_policy_gate_result(result_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM policy_gate_results WHERE id = ?", (result_id,)).fetchone()
    if not row:
        return None
    return _parse_policy_gate_result_row(row)


def list_policy_gate_results(
    *,
    decision: str | None = None,
    target_type: str | None = None,
    target_id: str | int | None = None,
    action_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if decision:
        clauses.append("decision = ?")
        params.append(decision)
    if target_type:
        clauses.append("target_type = ?")
        params.append(target_type)
    if target_id is not None:
        clauses.append("target_id = ?")
        params.append(str(target_id))
    if action_id:
        clauses.append("action_id = ?")
        params.append(action_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM policy_gate_results{where} ORDER BY created_at DESC, id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [_parse_policy_gate_result_row(row) for row in rows]


def _ensure_policy_gate_result_for_recommendation(record: dict) -> int | None:
    if record.get("policy_gate_result_id"):
        return int(record["policy_gate_result_id"])
    result = record.get("policy_gate_result")
    if not isinstance(result, dict):
        return None
    row = create_policy_gate_result(
        result,
        action_id="create_recommendation",
        source_type=record.get("source_type") or "recommendation",
        source_id=record.get("report_id") or record.get("idempotency_key"),
        target_type="recommendation",
        target_id=record.get("idempotency_key") or record.get("ticker") or record.get("instrument"),
        payload=record,
    )
    return int(row["id"])


def _risk_projection_enabled() -> bool:
    try:
        from api.position_risk import risk_compat_projections_enabled
    except Exception:
        return True
    return risk_compat_projections_enabled()


def _quality_rank(value: Any) -> int:
    state = str(value or "ok").strip().lower()
    if state == "ok":
        return 0
    if state == "degraded":
        return 1
    if state == "stale":
        return 2
    return 3


def _quality_from_rank(rank: int) -> str:
    if rank <= 0:
        return "ok"
    if rank == 1:
        return "degraded"
    if rank == 2:
        return "stale"
    return "failed"


def _project_recommendation_risk_quality(record: dict) -> dict:
    if not _risk_projection_enabled() or not record.get("risk_quality"):
        return record
    out = dict(record)
    projected = _quality_from_rank(
        max(_quality_rank(out.get("critical_data_quality")), _quality_rank(out.get("risk_quality")))
    )
    out["critical_data_quality"] = projected
    if out.get("source_quality"):
        out["source_quality"] = _quality_from_rank(
            max(_quality_rank(out.get("source_quality")), _quality_rank(out.get("risk_quality")))
        )
    return out


def _materialize_recommendation_risk_binding_tx(
    conn: sqlite3.Connection | PostgresCompatConnection,
    recommendation: dict,
    record: dict,
) -> None:
    binding = record.get("risk_bindings")
    if not isinstance(binding, dict):
        binding = {}
    risk_snapshot_id = record.get("risk_snapshot_id") or recommendation.get("risk_snapshot_id")
    portfolio_risk_snapshot_id = record.get("portfolio_risk_snapshot_id") or recommendation.get(
        "portfolio_risk_snapshot_id"
    )
    if not risk_snapshot_id and not portfolio_risk_snapshot_id and not binding:
        return

    recommendation_id = int(recommendation["id"])
    conn.execute("DELETE FROM recommendation_risk_bindings WHERE recommendation_id = ?", (recommendation_id,))
    conn.execute(
        """
        INSERT INTO recommendation_risk_bindings (
            recommendation_id,
            created_at,
            ticker,
            risk_snapshot_id,
            portfolio_risk_snapshot_id,
            risk_quality,
            risk_confidence,
            risk_score,
            risk_level,
            source_status_json,
            binding_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            recommendation_id,
            _now(),
            recommendation.get("ticker"),
            risk_snapshot_id,
            portfolio_risk_snapshot_id,
            record.get("risk_quality") or recommendation.get("risk_quality"),
            record.get("risk_confidence") or recommendation.get("risk_confidence"),
            record.get("risk_score") or recommendation.get("risk_score"),
            record.get("risk_level") or recommendation.get("risk_level"),
            _encode_json(record.get("risk_source_status") or recommendation.get("risk_source_status_json") or {}),
            _encode_json(
                binding
                or {
                    "risk_snapshot_id": risk_snapshot_id,
                    "portfolio_risk_snapshot_id": portfolio_risk_snapshot_id,
                }
            ),
        ),
    )


def _recommendation_governance_bundle(recommendation: dict, record: dict) -> dict[str, Any]:
    from api import governance

    recommendation_id = int(recommendation["id"])
    lineage_root_id = governance.lineage_root(governance.REF_RECOMMENDATION, recommendation_id)
    provenance_event_id = governance.deterministic_id("pv:recommendation", recommendation_id, "generated")
    policy_gate_result_id = recommendation.get("policy_gate_result_id")
    object_refs: list[dict[str, Any]] = [{"type": governance.REF_RECOMMENDATION, "id": recommendation_id}]
    if policy_gate_result_id:
        object_refs.append({"type": governance.REF_POLICY_GATE_RESULT, "id": policy_gate_result_id})
    if recommendation.get("approval_id"):
        object_refs.append({"type": governance.REF_APPROVAL, "id": recommendation.get("approval_id")})
    bundle: dict[str, Any] = {
        "lineage_root_id": lineage_root_id,
        "provenance_events": [
            governance.provenance_event(
                event_id=provenance_event_id,
                event_type="recommendation",
                event_name=governance.EVENT_RECOMMENDATION_GENERATED,
                lineage_root_id=lineage_root_id,
                input_value={
                    "input_hash": recommendation.get("input_hash"),
                    "prompt_hash": recommendation.get("prompt_hash"),
                    "report_id": recommendation.get("report_id"),
                    "policy_gate_result_id": policy_gate_result_id,
                    "risk_snapshot_id": recommendation.get("risk_snapshot_id"),
                    "portfolio_risk_snapshot_id": recommendation.get("portfolio_risk_snapshot_id"),
                },
                output_value={
                    "recommendation_id": recommendation_id,
                    "status": recommendation.get("status"),
                    "recommendation_status": recommendation.get("recommendation_status"),
                    "idempotency_key": recommendation.get("idempotency_key"),
                },
                summary={
                    "recommendation_id": recommendation_id,
                    "report_type": recommendation.get("report_type"),
                    "ticker": recommendation.get("ticker"),
                    "instrument": recommendation.get("instrument"),
                    "action": recommendation.get("action"),
                    "status": recommendation.get("status"),
                    "recommendation_status": recommendation.get("recommendation_status"),
                    "policy_gate_result_id": policy_gate_result_id,
                    "risk_snapshot_id": recommendation.get("risk_snapshot_id"),
                    "portfolio_risk_snapshot_id": recommendation.get("portfolio_risk_snapshot_id"),
                },
                metadata={
                    "model": recommendation.get("model"),
                    "prompt_hash": recommendation.get("prompt_hash"),
                    "input_hash": recommendation.get("input_hash"),
                    "report_id": recommendation.get("report_id"),
                    "source_quality": recommendation.get("source_quality"),
                    "risk_quality": recommendation.get("risk_quality"),
                    "risk_confidence": recommendation.get("risk_confidence"),
                    "validation_status": recommendation.get("validation_status"),
                },
            )
        ],
        "audit_events": [
            governance.audit_event(
                action_name=governance.EVENT_RECOMMENDATION_GENERATED,
                status=str(recommendation.get("status") or "open"),
                lineage_root_id=lineage_root_id,
                object_refs=object_refs,
                after_summary={
                    "recommendation_id": recommendation_id,
                    "report_type": recommendation.get("report_type"),
                    "ticker": recommendation.get("ticker"),
                    "instrument": recommendation.get("instrument"),
                    "action": recommendation.get("action"),
                    "status": recommendation.get("status"),
                    "recommendation_status": recommendation.get("recommendation_status"),
                    "policy_gate_result_id": policy_gate_result_id,
                    "risk_snapshot_id": recommendation.get("risk_snapshot_id"),
                    "portfolio_risk_snapshot_id": recommendation.get("portfolio_risk_snapshot_id"),
                    "model": recommendation.get("model"),
                    "prompt_hash": recommendation.get("prompt_hash"),
                    "input_hash": recommendation.get("input_hash"),
                },
                source_lineage={
                    "report_id": recommendation.get("report_id"),
                    "source_report_path_hash": _json_hash(recommendation.get("source_report_path"))
                    if recommendation.get("source_report_path")
                    else None,
                    "source_json_path_hash": _json_hash(recommendation.get("source_json_path"))
                    if recommendation.get("source_json_path")
                    else None,
                },
            )
        ],
        "recommendation_updates": [
            {
                "recommendation_id": recommendation_id,
                "provenance_event_id": provenance_event_id,
                "lineage_root_id": lineage_root_id,
                "lineage_completeness": "complete",
            }
        ],
    }
    links: list[dict[str, Any]] = []
    if recommendation.get("report_id"):
        links.append(
            governance.provenance_link(
                event_id=provenance_event_id,
                source_ref_type=governance.REF_REPORT_RUN,
                source_ref_id=str(recommendation["report_id"]),
                target_ref_type=governance.REF_RECOMMENDATION,
                target_ref_id=recommendation_id,
                link_type="produced",
                lineage_root_id=lineage_root_id,
            )
        )
    if policy_gate_result_id:
        links.append(
            governance.provenance_link(
                event_id=provenance_event_id,
                source_ref_type=governance.REF_POLICY_GATE_RESULT,
                source_ref_id=str(policy_gate_result_id),
                target_ref_type=governance.REF_RECOMMENDATION,
                target_ref_id=recommendation_id,
                link_type="gated",
                lineage_root_id=lineage_root_id,
            )
        )
    if recommendation.get("risk_snapshot_id"):
        links.append(
            governance.provenance_link(
                event_id=provenance_event_id,
                source_ref_type="position_risk_snapshot",
                source_ref_id=str(recommendation["risk_snapshot_id"]),
                target_ref_type=governance.REF_RECOMMENDATION,
                target_ref_id=recommendation_id,
                link_type="used",
                lineage_root_id=lineage_root_id,
            )
        )
    if recommendation.get("portfolio_risk_snapshot_id"):
        links.append(
            governance.provenance_link(
                event_id=provenance_event_id,
                source_ref_type="portfolio_risk_snapshot",
                source_ref_id=str(recommendation["portfolio_risk_snapshot_id"]),
                target_ref_type=governance.REF_RECOMMENDATION,
                target_ref_id=recommendation_id,
                link_type="used",
                lineage_root_id=lineage_root_id,
            )
        )
    if recommendation.get("model"):
        links.append(
            governance.provenance_link(
                event_id=provenance_event_id,
                source_ref_type=governance.REF_MODEL_CALL,
                source_ref_id=str(recommendation.get("model")),
                target_ref_type=governance.REF_RECOMMENDATION,
                target_ref_id=recommendation_id,
                link_type="used",
                lineage_root_id=lineage_root_id,
                metadata={"model": recommendation.get("model")},
            )
        )
    if record.get("prompt_hash") or recommendation.get("prompt_hash"):
        links.append(
            governance.provenance_link(
                event_id=provenance_event_id,
                source_ref_type=governance.REF_PROMPT_TEMPLATE,
                source_ref_id=str(record.get("prompt_hash") or recommendation.get("prompt_hash")),
                target_ref_type=governance.REF_RECOMMENDATION,
                target_ref_id=recommendation_id,
                link_type="used",
                lineage_root_id=lineage_root_id,
            )
        )
    if links:
        bundle["provenance_links"] = links
    return bundle


def create_recommendation(record: dict) -> dict:
    _guard_legacy_domain_write("core_db.create_recommendation")
    record = _project_recommendation_risk_quality(record)
    conn = _get_conn()
    now = record.get("created_at") or _now()
    ticker = record.get("ticker")
    policy_gate_result_id = _ensure_policy_gate_result_for_recommendation(record)
    status = record.get("status") or (
        "blocked"
        if record.get("recommendation_status") == "blocked"
        else "error"
        if record.get("recommendation_status") == "error"
        else "open"
    )
    params = {
        "report_type": record["report_type"],
        "as_of": record["as_of"],
        "created_at": now,
        "source_report_path": record.get("source_report_path"),
        "source_json_path": record.get("source_json_path"),
        "stance": record["stance"],
        "recommendation_status": record.get("recommendation_status", "clear"),
        "critical_data_quality": record.get("critical_data_quality", "ok"),
        "blocked_reasons_json": _encode_json(record.get("blocked_reasons", [])),
        "what_changed_json": _encode_json(record.get("what_changed", [])),
        "do_nothing_rationale": record.get("do_nothing_rationale", ""),
        "action": record["action"],
        "ticker": ticker.upper() if isinstance(ticker, str) and ticker else None,
        "instrument": record.get("instrument") or "portfolio",
        "horizon": record.get("horizon"),
        "target_change": record.get("target_change"),
        "rationale": record.get("rationale") or "",
        "confidence": record.get("confidence"),
        "source_quality": record.get("source_quality", "ok"),
        "status": status,
        "evidence_json": _encode_json(record.get("evidence", [])),
        "disconfirming_evidence_json": _encode_json(record.get("disconfirming_evidence", [])),
        "catalyst": record.get("catalyst"),
        "invalidation": record.get("invalidation"),
        "expected_onset_window": record.get("expected_onset_window"),
        "alternatives_json": _encode_json(record.get("alternatives", [])),
        "opportunity_cost_json": _encode_json(record.get("opportunity_cost", [])),
        "approval_id": record.get("approval_id"),
        "approval_status": record.get("approval_status", "none"),
        "outcome_status": record.get("outcome_status", "pending"),
        "outcome_json": _encode_json(record.get("outcome", {})),
        "model": record.get("model"),
        "prompt_hash": record.get("prompt_hash"),
        "input_hash": record.get("input_hash"),
        "validation_status": record.get("validation_status"),
        "source_quality_summary_json": _encode_json(record.get("source_quality_summary", {})),
        "report_id": record.get("report_id"),
        "idempotency_key": record.get("idempotency_key"),
        "policy_gate_result_id": policy_gate_result_id,
        "policy_gate_status": record.get("policy_gate_status") or record.get("policy_gate_decision"),
        "policy_gate_decision": record.get("policy_gate_decision"),
        "policy_gate_review_required": 1 if record.get("policy_gate_review_required") else 0,
        "policy_gate_failures_json": _encode_json(record.get("policy_gate_failures", [])),
        "policy_gate_warnings_json": _encode_json(record.get("policy_gate_warnings", [])),
        "policy_gate_disclosures_json": _encode_json(record.get("policy_gate_disclosures", [])),
        "account_id": record.get("account_id"),
        "portfolio_id": record.get("portfolio_id"),
        "policy_id": record.get("policy_id"),
        "trade_proposal_json": _encode_json(record.get("trade_proposal", {})),
        "risk_snapshot_id": record.get("risk_snapshot_id"),
        "portfolio_risk_snapshot_id": record.get("portfolio_risk_snapshot_id"),
        "risk_quality": record.get("risk_quality"),
        "risk_confidence": record.get("risk_confidence"),
        "risk_score": record.get("risk_score"),
        "risk_level": record.get("risk_level"),
        "risk_source_status_json": _encode_json(record.get("risk_source_status", {})),
        "risk_bindings_json": _encode_json(record.get("risk_bindings", {})),
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    with _lock:
        try:
            cur = conn.execute(
                f"INSERT INTO recommendations ({columns}) VALUES ({placeholders})",
                tuple(params.values()),
            )
            row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (cur.lastrowid,)).fetchone()
            result = _parse_recommendation_json_fields(_require_row_dict(row))
            _materialize_recommendation_risk_binding_tx(conn, result, record)
            _materialize_governance_bundle_tx(conn, _recommendation_governance_bundle(result, record))
            row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (cur.lastrowid,)).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return _parse_recommendation_json_fields(_require_row_dict(row))


def upsert_recommendation(record: dict) -> dict:
    _guard_legacy_domain_write("core_db.upsert_recommendation")
    record = _project_recommendation_risk_quality(record)
    if not record.get("idempotency_key"):
        return create_recommendation(record)

    conn = _get_conn()
    now = record.get("created_at") or _now()
    ticker = record.get("ticker")
    policy_gate_result_id = _ensure_policy_gate_result_for_recommendation(record)
    status = record.get("status") or (
        "blocked"
        if record.get("recommendation_status") == "blocked"
        else "error"
        if record.get("recommendation_status") == "error"
        else "open"
    )
    params = {
        "report_type": record["report_type"],
        "as_of": record["as_of"],
        "created_at": now,
        "source_report_path": record.get("source_report_path"),
        "source_json_path": record.get("source_json_path"),
        "stance": record["stance"],
        "recommendation_status": record.get("recommendation_status", "clear"),
        "critical_data_quality": record.get("critical_data_quality", "ok"),
        "blocked_reasons_json": _encode_json(record.get("blocked_reasons", [])),
        "what_changed_json": _encode_json(record.get("what_changed", [])),
        "do_nothing_rationale": record.get("do_nothing_rationale", ""),
        "action": record["action"],
        "ticker": ticker.upper() if isinstance(ticker, str) and ticker else None,
        "instrument": record.get("instrument") or "portfolio",
        "horizon": record.get("horizon"),
        "target_change": record.get("target_change"),
        "rationale": record.get("rationale") or "",
        "confidence": record.get("confidence"),
        "source_quality": record.get("source_quality", "ok"),
        "status": status,
        "evidence_json": _encode_json(record.get("evidence", [])),
        "disconfirming_evidence_json": _encode_json(record.get("disconfirming_evidence", [])),
        "catalyst": record.get("catalyst"),
        "invalidation": record.get("invalidation"),
        "expected_onset_window": record.get("expected_onset_window"),
        "alternatives_json": _encode_json(record.get("alternatives", [])),
        "opportunity_cost_json": _encode_json(record.get("opportunity_cost", [])),
        "approval_id": record.get("approval_id"),
        "approval_status": record.get("approval_status", "none"),
        "outcome_status": record.get("outcome_status", "pending"),
        "outcome_json": _encode_json(record.get("outcome", {})),
        "model": record.get("model"),
        "prompt_hash": record.get("prompt_hash"),
        "input_hash": record.get("input_hash"),
        "validation_status": record.get("validation_status"),
        "source_quality_summary_json": _encode_json(record.get("source_quality_summary", {})),
        "report_id": record.get("report_id"),
        "idempotency_key": record["idempotency_key"],
        "policy_gate_result_id": policy_gate_result_id,
        "policy_gate_status": record.get("policy_gate_status") or record.get("policy_gate_decision"),
        "policy_gate_decision": record.get("policy_gate_decision"),
        "policy_gate_review_required": 1 if record.get("policy_gate_review_required") else 0,
        "policy_gate_failures_json": _encode_json(record.get("policy_gate_failures", [])),
        "policy_gate_warnings_json": _encode_json(record.get("policy_gate_warnings", [])),
        "policy_gate_disclosures_json": _encode_json(record.get("policy_gate_disclosures", [])),
        "account_id": record.get("account_id"),
        "portfolio_id": record.get("portfolio_id"),
        "policy_id": record.get("policy_id"),
        "trade_proposal_json": _encode_json(record.get("trade_proposal", {})),
        "risk_snapshot_id": record.get("risk_snapshot_id"),
        "portfolio_risk_snapshot_id": record.get("portfolio_risk_snapshot_id"),
        "risk_quality": record.get("risk_quality"),
        "risk_confidence": record.get("risk_confidence"),
        "risk_score": record.get("risk_score"),
        "risk_level": record.get("risk_level"),
        "risk_source_status_json": _encode_json(record.get("risk_source_status", {})),
        "risk_bindings_json": _encode_json(record.get("risk_bindings", {})),
    }
    columns = ", ".join(params)
    placeholders = ", ".join("?" for _ in params)
    updates = ", ".join(
        f"{column} = COALESCE(excluded.{column}, {column})"
        if column in {"approval_id"}
        else f"{column} = excluded.{column}"
        for column in params
        if column
        not in {"created_at", "idempotency_key", "approval_status", "approval_id", "outcome_status", "outcome_json"}
    )
    updates = (
        f"{updates}, "
        "approval_id = COALESCE(recommendations.approval_id, excluded.approval_id), "
        "approval_status = CASE WHEN recommendations.approval_id IS NOT NULL THEN recommendations.approval_status ELSE excluded.approval_status END, "
        "outcome_status = recommendations.outcome_status, "
        "outcome_json = recommendations.outcome_json"
    )
    with _lock:
        try:
            conn.execute(
                f"INSERT INTO recommendations ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(idempotency_key) DO UPDATE SET {updates}",
                tuple(params.values()),
            )
            row = conn.execute(
                "SELECT * FROM recommendations WHERE idempotency_key = ?",
                (record["idempotency_key"],),
            ).fetchone()
            result = _parse_recommendation_json_fields(_require_row_dict(row))
            _materialize_recommendation_risk_binding_tx(conn, result, record)
            _materialize_governance_bundle_tx(conn, _recommendation_governance_bundle(result, record))
            row = conn.execute(
                "SELECT * FROM recommendations WHERE idempotency_key = ?",
                (record["idempotency_key"],),
            ).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return _parse_recommendation_json_fields(_require_row_dict(row))


def get_recommendations(
    report_type: str | None = None,
    status: str | None = None,
    ticker: str | None = None,
    approval_status: str | None = None,
    outcome_status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if report_type:
        clauses.append("report_type = ?")
        params.append(report_type)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if approval_status:
        clauses.append("approval_status = ?")
        params.append(approval_status)
    if outcome_status:
        clauses.append("outcome_status = ?")
        params.append(outcome_status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM recommendations{where} ORDER BY as_of DESC, created_at DESC, id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [_parse_recommendation_json_fields(d) for d in _rows_to_list(rows)]


def get_recommendation(recommendation_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    if not row:
        return None
    return _parse_recommendation_json_fields(_require_row_dict(row))


def get_recommendation_risk_bindings(recommendation_id: int) -> list[dict]:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            """
            SELECT *
            FROM recommendation_risk_bindings
            WHERE recommendation_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (recommendation_id,),
        ).fetchall()
    out = []
    for row in _rows_to_list(rows):
        _parse_json_field(row, "source_status_json")
        _parse_json_field(row, "binding_json")
        out.append(row)
    return out


def get_latest_recommendation(report_type: str | None = None) -> dict | None:
    results = get_recommendations(report_type=report_type, limit=1)
    return results[0] if results else None


def _update_recommendation_approval_tx(
    conn: sqlite3.Connection | PostgresCompatConnection,
    recommendation_id: int,
    approval_id: int | None,
    approval_status: str,
) -> dict:
    row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    if not row:
        raise ValueError(f"No recommendation with id {recommendation_id}")
    conn.execute(
        "UPDATE recommendations SET approval_id = COALESCE(?, approval_id), approval_status = ? WHERE id = ?",
        (approval_id, approval_status, recommendation_id),
    )
    updated = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    return _parse_recommendation_json_fields(_require_row_dict(updated))


def update_recommendation_approval(
    recommendation_id: int,
    approval_id: int | None,
    approval_status: str,
) -> dict:
    conn = _get_conn()
    with _lock:
        updated = _update_recommendation_approval_tx(conn, recommendation_id, approval_id, approval_status)
        conn.commit()
    _emit_core_audit(
        "recommendation.approval.updated",
        status="succeeded",
        object_refs=[
            {"type": "recommendation", "id": recommendation_id},
            {"type": "approval", "id": approval_id} if approval_id is not None else {"type": "approval", "id": "none"},
        ],
        after_summary={
            "recommendation_id": recommendation_id,
            "approval_id": approval_id,
            "approval_status": approval_status,
            "ticker": updated.get("ticker"),
            "report_id": updated.get("report_id"),
        },
    )
    return updated


def supersede_report_recommendations(report_id: str, active_idempotency_keys: list[str]) -> int:
    conn = _get_conn()
    if not report_id:
        return 0
    active = set(active_idempotency_keys)
    with _lock:
        rows = conn.execute(
            "SELECT id, idempotency_key FROM recommendations WHERE report_id = ? AND status IN ('open', 'blocked', 'error')",
            (report_id,),
        ).fetchall()
        superseded_ids = [int(row["id"]) for row in rows if str(row["idempotency_key"] or "") not in active]
        if not superseded_ids:
            return 0
        placeholders = ", ".join("?" for _ in superseded_ids)
        conn.execute(
            f"UPDATE recommendations SET status = 'superseded' WHERE id IN ({placeholders})",
            tuple(superseded_ids),
        )
        conn.commit()
    return len(superseded_ids)


def update_recommendation_outcome(
    recommendation_id: int,
    outcome_status: str,
    outcome: dict,
) -> dict:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
        if not row:
            raise ValueError(f"No recommendation with id {recommendation_id}")
        conn.execute(
            "UPDATE recommendations SET outcome_status = ?, outcome_json = ? WHERE id = ?",
            (outcome_status, json.dumps(outcome, default=str), recommendation_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    result = _parse_recommendation_json_fields(_require_row_dict(updated))
    _emit_core_audit(
        "recommendation.outcome.updated",
        status="succeeded",
        object_refs=[{"type": "recommendation", "id": recommendation_id}],
        after_summary={
            "recommendation_id": recommendation_id,
            "outcome_status": outcome_status,
            "outcome_hash": _json_hash(outcome),
            "ticker": result.get("ticker"),
            "report_id": result.get("report_id"),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Pending Approvals
# ---------------------------------------------------------------------------

ApprovalPostCommitCallback = Callable[[], None]
ApprovalSideEffectHandler = Callable[
    [sqlite3.Connection | PostgresCompatConnection, dict, dict, list[ApprovalPostCommitCallback]],
    None,
]


class ApprovalApplicationError(RuntimeError):
    """Raised when an approval side effect fails and remains retryable."""

    def __init__(self, approval_id: int, error: str):
        self.approval_id = approval_id
        self.error = error
        super().__init__(f"Approval {approval_id} application failed: {error}")


def _approval_governance_bundle(
    approval: dict,
    *,
    event_name: str,
    status: str,
    action_run_id: int | None = None,
    actor_id: str | None = None,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    from api import governance

    approval_id = int(approval["id"])
    lineage_root_id = governance.lineage_root(governance.REF_APPROVAL, approval_id)
    provenance_event_id = governance.deterministic_id("pv:approval", approval_id, event_name, status)
    object_refs: list[dict[str, Any]] = [{"type": governance.REF_APPROVAL, "id": approval_id}]
    if action_run_id is not None:
        object_refs.append({"type": governance.REF_ACTION_RUN, "id": action_run_id})
    if approval.get("entity_type"):
        object_refs.append({"type": approval.get("entity_type"), "id": approval.get("entity_id")})
    payload_summary = {
        "approval_id": approval_id,
        "entity_type": approval.get("entity_type"),
        "entity_id": approval.get("entity_id"),
        "ticker": approval.get("ticker"),
        "action_id": approval.get("action_id"),
        "status": approval.get("status"),
        "application_status": approval.get("application_status"),
        **(summary or {}),
    }
    bundle: dict[str, Any] = {
        "lineage_root_id": lineage_root_id,
        "provenance_events": [
            governance.provenance_event(
                event_id=provenance_event_id,
                event_type="approval",
                event_name=event_name,
                status="failed" if status == "failed" else "succeeded",
                lineage_root_id=lineage_root_id,
                action_run_id=action_run_id,
                approval_id=approval_id,
                output_value=payload_summary,
                summary=payload_summary,
                metadata={
                    "source_type": approval.get("source_type"),
                    "source_id": approval.get("source_id"),
                    "action_input_hash": approval.get("action_input_hash"),
                    "resolved_by_actor_id": actor_id,
                },
                error=error,
            )
        ],
        "audit_events": [
            governance.audit_event(
                action_name=event_name,
                status=status,
                lineage_root_id=lineage_root_id,
                actor_id=actor_id,
                object_refs=object_refs,
                after_summary=payload_summary,
                source_lineage={
                    "source_type": approval.get("source_type"),
                    "source_id": approval.get("source_id"),
                    "action_input_hash": approval.get("action_input_hash"),
                    "approval_provenance_event_id": approval.get("provenance_event_id"),
                },
                error=error,
            )
        ],
    }
    if action_run_id is not None:
        bundle["provenance_links"] = [
            governance.provenance_link(
                event_id=provenance_event_id,
                source_ref_type=governance.REF_APPROVAL,
                source_ref_id=str(approval_id),
                target_ref_type=governance.REF_ACTION_RUN,
                target_ref_id=str(action_run_id),
                link_type="resolved_by" if event_name == governance.EVENT_APPROVAL_RESOLVED else "applied_by",
                lineage_root_id=lineage_root_id,
            )
        ]
    return bundle


def _parse_pending_approval_row(row: Any) -> dict:
    d = _require_row_dict(row)
    _parse_json_field(d, "proposed_change")
    if d.get("application_attempts") is None:
        d["application_attempts"] = 0
    if not d.get("application_status"):
        d["application_status"] = "pending"
    d["approval_note_required"] = bool(d.get("approval_note_required"))
    return d


def create_pending_approval(
    entity_type: str,
    proposed_change: dict,
    *,
    entity_id: int | None = None,
    ticker: str | None = None,
    reason: str | None = None,
    source_type: str = "workflow",
    source_id: str | None = None,
    action_id: str | None = None,
    action_schema_name: str | None = None,
    action_schema_version: int | None = None,
    action_input_hash: str | None = None,
    request_schema_name: str | None = None,
    request_schema_version: int | None = None,
    risk_class: str | None = None,
    approval_mode: str | None = None,
    base_state_hash: str | None = None,
    requested_by_actor_id: str | None = None,
    approval_note_required: bool = False,
    reason_code: str | None = None,
    supersedes_approval_id: int | None = None,
) -> dict:
    from api import governance

    conn = _get_conn()
    now = _now()
    change_json = json.dumps(proposed_change, default=str)
    with _lock:
        try:
            cur = conn.execute(
                "INSERT INTO pending_approvals (entity_type, entity_id, ticker, action_id, action_schema_name, "
                "action_schema_version, action_input_hash, request_schema_name, request_schema_version, proposed_change, "
                "reason, source_type, source_id, created_at, risk_class, approval_mode, base_state_hash, "
                "requested_by_actor_id, approval_note_required, reason_code, supersedes_approval_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    entity_type,
                    entity_id,
                    ticker.upper() if ticker else None,
                    action_id,
                    action_schema_name,
                    action_schema_version,
                    action_input_hash,
                    request_schema_name,
                    request_schema_version,
                    change_json,
                    reason,
                    source_type,
                    source_id,
                    now,
                    risk_class,
                    approval_mode,
                    base_state_hash,
                    requested_by_actor_id,
                    1 if approval_note_required else 0,
                    reason_code,
                    supersedes_approval_id,
                ),
            )
            approval_id = cast(int, cur.lastrowid)
            lineage_root_id = governance.lineage_root(governance.REF_APPROVAL, approval_id)
            provenance_event_id = governance.deterministic_id("pv:approval", approval_id, "created")
            object_refs: list[dict[str, Any]] = [
                {"type": governance.REF_APPROVAL, "id": approval_id},
                {"type": entity_type, "id": entity_id},
            ]
            bundle: dict[str, Any] = {
                "lineage_root_id": lineage_root_id,
                "provenance_events": [
                    governance.provenance_event(
                        event_id=provenance_event_id,
                        event_type="approval",
                        event_name=governance.EVENT_APPROVAL_CREATED,
                        lineage_root_id=lineage_root_id,
                        approval_id=approval_id,
                        input_value=proposed_change,
                        output_value={"approval_id": approval_id, "status": "pending"},
                        summary={
                            "approval_id": approval_id,
                            "entity_type": entity_type,
                            "entity_id": entity_id,
                            "ticker": ticker.upper() if ticker else None,
                            "action_id": action_id,
                            "status": "pending",
                        },
                        metadata={
                            "change_hash": _json_hash(proposed_change),
                            "source_type": source_type,
                            "source_id": source_id,
                            "action_input_hash": action_input_hash,
                            "risk_class": risk_class,
                            "approval_mode": approval_mode,
                            "base_state_hash": base_state_hash,
                            "requested_by_actor_id": requested_by_actor_id,
                        },
                    )
                ],
                "audit_events": [
                    governance.audit_event(
                        action_name=governance.EVENT_APPROVAL_CREATED,
                        status="pending",
                        lineage_root_id=lineage_root_id,
                        object_refs=object_refs,
                        after_summary={
                            "approval_id": approval_id,
                            "entity_type": entity_type,
                            "entity_id": entity_id,
                            "ticker": ticker.upper() if ticker else None,
                            "action_id": action_id,
                            "status": "pending",
                            "change_hash": _json_hash(proposed_change),
                            "risk_class": risk_class,
                            "approval_mode": approval_mode,
                            "base_state_hash": base_state_hash,
                        },
                        source_lineage={
                            "source_type": source_type,
                            "source_id": source_id,
                            "action_input_hash": action_input_hash,
                            "requested_by_actor_id": requested_by_actor_id,
                        },
                    )
                ],
            }
            if source_type and source_id:
                bundle["provenance_links"] = [
                    governance.provenance_link(
                        event_id=provenance_event_id,
                        source_ref_type=source_type,
                        source_ref_id=str(source_id),
                        target_ref_type=governance.REF_APPROVAL,
                        target_ref_id=str(approval_id),
                        link_type="proposed",
                        lineage_root_id=lineage_root_id,
                        metadata={"action_id": action_id, "entity_type": entity_type},
                    )
                ]
            _materialize_governance_bundle_tx(conn, bundle)
            conn.execute(
                "UPDATE pending_approvals SET provenance_event_id = ?, lineage_completeness = 'complete' WHERE id = ?",
                (provenance_event_id, approval_id),
            )
            row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return _parse_pending_approval_row(row)


def create_pending_approval_once(
    entity_type: str,
    proposed_change: dict,
    *,
    entity_id: int | None = None,
    ticker: str | None = None,
    reason: str | None = None,
    source_type: str = "workflow",
    source_id: str | None = None,
    action_id: str | None = None,
    action_schema_name: str | None = None,
    action_schema_version: int | None = None,
    action_input_hash: str | None = None,
    request_schema_name: str | None = None,
    request_schema_version: int | None = None,
    risk_class: str | None = None,
    approval_mode: str | None = None,
    base_state_hash: str | None = None,
    requested_by_actor_id: str | None = None,
    approval_note_required: bool = False,
    reason_code: str | None = None,
    supersedes_approval_id: int | None = None,
) -> dict:
    proposed_hash = _json_hash(proposed_change)
    normalized_ticker = ticker.upper() if ticker else None
    if source_id:
        for approval in get_pending_approvals(status=None, ticker=normalized_ticker):
            if approval.get("entity_type") != entity_type:
                continue
            if action_id and approval.get("action_id") != action_id:
                continue
            if approval.get("source_type") != source_type or approval.get("source_id") != source_id:
                continue
            if _json_hash(approval.get("proposed_change")) == proposed_hash:
                return approval
    return create_pending_approval(
        entity_type=entity_type,
        proposed_change=proposed_change,
        entity_id=entity_id,
        ticker=normalized_ticker,
        reason=reason,
        source_type=source_type,
        source_id=source_id,
        action_id=action_id,
        action_schema_name=action_schema_name,
        action_schema_version=action_schema_version,
        action_input_hash=action_input_hash,
        request_schema_name=request_schema_name,
        request_schema_version=request_schema_version,
        risk_class=risk_class,
        approval_mode=approval_mode,
        base_state_hash=base_state_hash,
        requested_by_actor_id=requested_by_actor_id,
        approval_note_required=approval_note_required,
        reason_code=reason_code,
        supersedes_approval_id=supersedes_approval_id,
    )


def get_pending_approvals(
    status: str | None = "pending",
    ticker: str | None = None,
    application_status: str | None = None,
) -> list[dict]:
    conn = _get_conn()
    clauses: list[str] = []
    params: list = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if application_status:
        if application_status not in APPROVAL_APPLICATION_STATUSES:
            raise ValueError(f"Invalid application_status: {application_status}")
        clauses.append("application_status = ?")
        params.append(application_status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with _lock:
        rows = conn.execute(
            f"SELECT * FROM pending_approvals{where} ORDER BY created_at DESC",
            params,
        ).fetchall()
    results = [_parse_pending_approval_row(row) for row in rows]
    _emit_core_audit(
        "approvals.read",
        status="succeeded",
        after_summary={
            "status": status,
            "ticker": ticker.upper() if ticker else None,
            "application_status": application_status,
            "result_count": len(results),
        },
    )
    return results


def get_pending_approval(approval_id: int) -> dict | None:
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
    if not row:
        return None
    result = _parse_pending_approval_row(row)
    _emit_core_audit(
        "approval.read",
        status="succeeded",
        object_refs=[
            {"type": "approval", "id": approval_id},
            {"type": result.get("entity_type"), "id": result.get("entity_id")},
        ],
        after_summary={
            "approval_id": approval_id,
            "status": result.get("status"),
            "application_status": result.get("application_status"),
            "entity_type": result.get("entity_type"),
        },
    )
    return result


def resolve_approval(
    approval_id: int,
    status: str,
    resolved_note: str | None = None,
    *,
    actor_type: str = "user",
    actor_id: str | None = None,
    parent_action_run_id: int | None = None,
) -> dict:
    """Resolve a pending approval and apply approved side effects safely."""
    run = create_action_run(
        action_id="resolve_approval",
        action_schema_version=1,
        actor_type=actor_type,
        actor_id=actor_id,
        approval_id=approval_id,
        parent_action_run_id=parent_action_run_id,
        input_hash=_json_hash({"approval_id": approval_id, "status": status, "resolved_note": resolved_note}),
        input_payload={"approval_id": approval_id, "status": status, "resolved_note": resolved_note},
    )
    run_id = int(run["id"])
    provenance_event_id: str | None = None
    try:
        from api import provenance

        provenance_event_id = provenance.deterministic_id("pv:action_run", run_id)
        provenance.start_event(
            event_id=provenance_event_id,
            event_type="action_run",
            event_name="resolve_approval",
            actor={"actor_type": actor_type, "actor_id": actor_id},
            action_run_id=run_id,
            approval_id=approval_id,
            input_value={"approval_id": approval_id, "status": status, "resolved_note": resolved_note},
            summary={"action_id": "resolve_approval", "approval_id": approval_id, "status": status},
            metadata={"parent_action_run_id": parent_action_run_id},
            criticality=GOVERNANCE_CRITICAL_FINANCIAL,
            lineage_root_id=f"approval:{approval_id}",
            idempotency_key=f"action_run.resolve_approval:{run_id}:start",
            retention_class=GOVERNANCE_FINANCIAL_RETENTION_CLASS,
            fail_closed=True,
        )
        set_action_run_provenance_event(run_id, provenance_event_id)
        provenance.link_refs(
            event_id=provenance_event_id,
            source_ref_type="approval",
            source_ref_id=str(approval_id),
            target_ref_type="action_run",
            target_ref_id=str(run_id),
            link_type="resolved_by",
            lineage_root_id=f"approval:{approval_id}",
            fail_closed=True,
        )
    except Exception as exc:
        error = _approval_error_message(exc)
        record_action_event(run_id, "error", message=error)
        complete_action_run(run_id, status="failed", error=error)
        raise
    record_action_event(run_id, "start", payload={"approval_id": approval_id, "status": status})
    try:
        result = _resolve_approval_impl(
            approval_id,
            status,
            resolved_note,
            parent_action_run_id=run_id,
            resolved_by_actor_id=actor_id,
        )
    except Exception as exc:
        error = _approval_error_message(exc)
        record_action_event(run_id, "error", message=error)
        complete_action_run(run_id, status="failed", error=error)
        try:
            from api import provenance

            provenance.finish_event(
                provenance_event_id,
                status="failed",
                summary={"approval_id": approval_id, "requested_status": status},
                error=error,
                fail_closed=True,
            )
        except Exception:
            pass
        _emit_core_audit(
            "approval.resolve.failed",
            status="failed",
            object_refs=[{"type": "approval", "id": approval_id}, {"type": "action_run", "id": run_id}],
            after_summary={"approval_id": approval_id, "requested_status": status},
            error=error,
        )
        raise
    record_action_event(run_id, "complete", payload={"status": result.get("status")})
    complete_action_run(run_id, status="succeeded", output_payload=result)
    try:
        from api import provenance

        provenance.finish_event(
            provenance_event_id,
            status="succeeded",
            output_value=result,
            summary={
                "approval_id": approval_id,
                "status": result.get("status"),
                "application_status": result.get("application_status"),
            },
            fail_closed=True,
        )
    except Exception:
        pass
    _emit_core_audit(
        "approval.resolved",
        status=str(result.get("status") or status),
        object_refs=[{"type": "approval", "id": approval_id}, {"type": "action_run", "id": run_id}],
        after_summary={
            "approval_id": approval_id,
            "status": result.get("status"),
            "application_status": result.get("application_status"),
            "entity_type": result.get("entity_type"),
            "ticker": result.get("ticker"),
        },
    )
    return result


def apply_approval_resolution(
    approval_id: int,
    status: str,
    resolved_note: str | None = None,
    *,
    parent_action_run_id: int | None = None,
    resolved_by_actor_id: str | None = None,
) -> dict:
    """Resolve an approval without creating a top-level audit run."""
    return _resolve_approval_impl(
        approval_id,
        status,
        resolved_note,
        parent_action_run_id=parent_action_run_id,
        resolved_by_actor_id=resolved_by_actor_id,
    )


def _resolve_approval_impl(
    approval_id: int,
    status: str,
    resolved_note: str | None = None,
    *,
    parent_action_run_id: int | None = None,
    resolved_by_actor_id: str | None = None,
) -> dict:
    if status not in ("approved", "rejected"):
        raise ValueError(f"Resolution status must be 'approved' or 'rejected', got '{status}'")

    if status == "rejected":
        return _reject_approval(
            approval_id,
            resolved_note,
            parent_action_run_id=parent_action_run_id,
            resolved_by_actor_id=resolved_by_actor_id,
        )

    conn = _get_conn()
    approval, should_apply = _claim_approval_for_application(conn, approval_id)
    if not should_apply:
        return approval
    if approval.get("approval_note_required") and not str(resolved_note or "").strip():
        _mark_approval_application_failed(approval_id, ValueError("Approval note is required for this action"))
        raise ValueError("Approval note is required for this action")

    callbacks: list[ApprovalPostCommitCallback] = []
    try:
        with _lock:
            try:
                _apply_approval_side_effect_tx(conn, approval, callbacks, parent_action_run_id=parent_action_run_id)
                _update_linked_recommendation_approval_tx(conn, approval, approval_id, "approved")
                now = _now()
                conn.execute(
                    "UPDATE pending_approvals "
                    "SET status = 'approved', resolved_at = ?, resolved_note = ?, resolved_by_actor_id = ?, "
                    "application_status = 'applied', application_completed_at = ?, application_error = NULL "
                    "WHERE id = ?",
                    (now, resolved_note, resolved_by_actor_id, now, approval_id),
                )
                updated = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
                updated_approval = _parse_pending_approval_row(updated)
                _materialize_governance_bundle_tx(
                    conn,
                    _approval_governance_bundle(
                        updated_approval,
                        event_name="approval.resolved",
                        status="approved",
                        action_run_id=parent_action_run_id,
                        actor_id=resolved_by_actor_id,
                        summary={"resolution": "approved"},
                    ),
                )
                _materialize_governance_bundle_tx(
                    conn,
                    _approval_governance_bundle(
                        updated_approval,
                        event_name="action.applied",
                        status="succeeded",
                        action_run_id=parent_action_run_id,
                        actor_id=resolved_by_actor_id,
                        summary={"resolution": "approved", "action_id": updated_approval.get("action_id")},
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    except Exception as exc:
        _mark_approval_application_failed(approval_id, exc)
        error = _approval_error_message(exc)
        raise ApprovalApplicationError(approval_id, error) from exc

    _run_approval_post_commit_callbacks(callbacks)
    result = _parse_pending_approval_row(updated)
    _emit_core_audit(
        "approval.applied",
        status="succeeded",
        object_refs=[
            {"type": "approval", "id": approval_id},
            {"type": result.get("entity_type"), "id": result.get("entity_id")},
        ],
        after_summary={
            "approval_id": approval_id,
            "entity_type": result.get("entity_type"),
            "entity_id": result.get("entity_id"),
            "ticker": result.get("ticker"),
            "status": result.get("status"),
            "application_status": result.get("application_status"),
            "application_attempts": result.get("application_attempts"),
        },
        source_lineage={"source_type": result.get("source_type"), "source_id": result.get("source_id")},
    )
    return result


def _reject_approval(
    approval_id: int,
    resolved_note: str | None,
    *,
    parent_action_run_id: int | None = None,
    resolved_by_actor_id: str | None = None,
) -> dict:
    conn = _get_conn()
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
        if not row:
            raise ValueError(f"No pending approval with id {approval_id}")
        current = _parse_pending_approval_row(row)
        if current["status"] != "pending":
            raise ValueError(f"Approval {approval_id} is already {current['status']}")
        application_status = str(current.get("application_status") or "pending")
        if application_status == "applying" and not _approval_application_is_stale(current):
            raise ValueError(f"Approval {approval_id} application is already in progress")

        try:
            _update_linked_recommendation_approval_tx(conn, current, approval_id, "rejected")
            conn.execute(
                "UPDATE pending_approvals "
                "SET status = 'rejected', resolved_at = ?, resolved_note = ?, resolved_by_actor_id = ?, "
                "application_status = 'not_applicable', application_completed_at = ?, application_error = NULL "
                "WHERE id = ?",
                (now, resolved_note, resolved_by_actor_id, now, approval_id),
            )
            updated = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
            updated_approval = _parse_pending_approval_row(updated)
            _materialize_governance_bundle_tx(
                conn,
                _approval_governance_bundle(
                    updated_approval,
                    event_name="approval.resolved",
                    status="rejected",
                    action_run_id=parent_action_run_id,
                    actor_id=resolved_by_actor_id,
                    summary={"resolution": "rejected"},
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    result = _parse_pending_approval_row(updated)
    _emit_core_audit(
        "approval.rejected",
        status="rejected",
        object_refs=[
            {"type": "approval", "id": approval_id},
            {"type": result.get("entity_type"), "id": result.get("entity_id")},
        ],
        after_summary={
            "approval_id": approval_id,
            "entity_type": result.get("entity_type"),
            "entity_id": result.get("entity_id"),
            "ticker": result.get("ticker"),
            "status": result.get("status"),
            "application_status": result.get("application_status"),
        },
        source_lineage={"source_type": result.get("source_type"), "source_id": result.get("source_id")},
    )
    return result


def _claim_approval_for_application(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval_id: int,
) -> tuple[dict, bool]:
    now = _now()
    with _lock:
        row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
        if not row:
            raise ValueError(f"No pending approval with id {approval_id}")
        current = _parse_pending_approval_row(row)
        application_status = str(current.get("application_status") or "pending")
        if current["status"] == "approved" and application_status == "applied":
            return current, False
        if current["status"] != "pending":
            raise ValueError(f"Approval {approval_id} is already {current['status']}")
        if application_status == "applying" and not _approval_application_is_stale(current):
            raise ValueError(f"Approval {approval_id} application is already in progress")
        if application_status not in {"pending", "failed", "applying"}:
            raise ValueError(f"Approval {approval_id} cannot be applied from state {application_status}")

        conn.execute(
            "UPDATE pending_approvals "
            "SET application_status = 'applying', application_attempts = COALESCE(application_attempts, 0) + 1, "
            "application_started_at = ?, application_completed_at = NULL, application_error = NULL "
            "WHERE id = ?",
            (now, approval_id),
        )
        updated = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
        updated_approval = _parse_pending_approval_row(updated)
        _materialize_governance_bundle_tx(
            conn,
            _approval_governance_bundle(
                updated_approval,
                event_name="approval.apply.started",
                status="started",
                summary={"application_attempts": updated_approval.get("application_attempts")},
            ),
        )
        conn.commit()
    result = _parse_pending_approval_row(updated)
    _emit_core_audit(
        "approval.apply.started",
        status="started",
        object_refs=[
            {"type": "approval", "id": approval_id},
            {"type": result.get("entity_type"), "id": result.get("entity_id")},
        ],
        after_summary={
            "approval_id": approval_id,
            "entity_type": result.get("entity_type"),
            "application_status": result.get("application_status"),
            "application_attempts": result.get("application_attempts"),
        },
        source_lineage={"source_type": result.get("source_type"), "source_id": result.get("source_id")},
    )
    return result, True


def _approval_application_is_stale(approval: dict) -> bool:
    started_at = approval.get("application_started_at")
    if not started_at:
        return True
    try:
        started = datetime.fromisoformat(str(started_at))
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return datetime.now(UTC) - started >= APPROVAL_APPLICATION_LEASE


def _mark_approval_application_failed(approval_id: int, exc: Exception) -> None:
    conn = _get_conn()
    now = _now()
    error = _approval_error_message(exc)
    with _lock:
        conn.execute(
            "UPDATE pending_approvals "
            "SET application_status = 'failed', application_completed_at = ?, application_error = ? "
            "WHERE id = ? AND status = 'pending'",
            (now, error, approval_id),
        )
        row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()
        if row:
            approval = _parse_pending_approval_row(row)
            _materialize_governance_bundle_tx(
                conn,
                _approval_governance_bundle(
                    approval,
                    event_name="approval.apply.failed",
                    status="failed",
                    summary={"application_status": "failed"},
                    error=error,
                ),
            )
        conn.commit()
    _emit_core_audit(
        "approval.apply.failed",
        status="failed",
        object_refs=[{"type": "approval", "id": approval_id}],
        after_summary={"approval_id": approval_id, "application_status": "failed"},
        error=error,
    )


def _approval_error_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[:1000]


def _approval_change(approval: dict) -> dict:
    change = approval.get("proposed_change")
    if isinstance(change, str):
        try:
            change = json.loads(change)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Approval proposed_change must be a JSON object") from exc
    if not isinstance(change, dict):
        raise ValueError("Approval proposed_change must be a JSON object")
    return change


def _update_linked_recommendation_approval_tx(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    approval_id: int,
    approval_status: str,
) -> None:
    change = approval.get("proposed_change")
    if not isinstance(change, dict) or change.get("recommendation_id") is None:
        return
    _update_recommendation_approval_tx(conn, int(change["recommendation_id"]), approval_id, approval_status)


def _apply_approval_side_effect_tx(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    callbacks: list[ApprovalPostCommitCallback],
    *,
    parent_action_run_id: int | None = None,
) -> None:
    action_id = str(approval.get("action_id") or "").strip()
    if action_id:
        from portfolio.action_registry import ActionContext, compute_action_base_state_hash, execute_action

        change = _approval_change(approval)
        stored_base_state_hash = str(approval.get("base_state_hash") or "").strip()
        if stored_base_state_hash:
            current_base_state_hash = compute_action_base_state_hash(action_id, change)
            if current_base_state_hash and current_base_state_hash != stored_base_state_hash:
                raise ValueError("Approval base state changed before application; refresh and create a new proposal")

        execute_action(
            action_id,
            change,
            ActionContext(
                actor_type="approval_apply",
                source_type=approval.get("source_type"),
                source_id=approval.get("source_id"),
                approval_id=int(approval["id"]),
                parent_action_run_id=parent_action_run_id,
            ),
            input_schema_version=int(approval.get("action_schema_version") or 1),
        )
        return
    entity_type = str(approval.get("entity_type") or "")
    handler = _APPROVAL_SIDE_EFFECT_HANDLERS.get(entity_type)
    if handler is None:
        raise ValueError(f"Unsupported approval entity_type: {entity_type}")
    handler(conn, approval, _approval_change(approval), callbacks)


def _run_approval_post_commit_callbacks(callbacks: list[ApprovalPostCommitCallback]) -> None:
    for callback in callbacks:
        try:
            callback()
        except Exception:
            logger.warning("Approval post-commit callback failed", exc_info=True)


def _optional_ticker(value: Any) -> str | None:
    ticker = str(value or "").strip().upper()
    return ticker or None


def _required_ticker(value: Any, entity_type: str) -> str:
    ticker = _optional_ticker(value)
    if not ticker:
        raise ValueError(f"{entity_type} approval requires ticker")
    return ticker


def _sync_markdown_callback(ticker: str) -> ApprovalPostCommitCallback:
    def _callback() -> None:
        from portfolio.thesis_sync import sync_markdown_from_entities

        sync_markdown_from_entities(ticker)

    return _callback


def _handle_thesis_status_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, callbacks
    ticker = _required_ticker(change.get("ticker") or approval.get("ticker"), "thesis_status")
    new_status = str(change.get("new_status") or "").strip().lower()
    if not new_status:
        raise ValueError("thesis_status approval requires new_status")
    reason = str(change.get("reason") or "")

    from portfolio.thesis_db import get_thesis_meta, update_thesis_status

    current = get_thesis_meta(ticker)
    if current and current.get("status") == new_status:
        return
    update_thesis_status(ticker, new_status, reason)


def _handle_evaluation_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, callbacks
    from portfolio.thesis_db import save_evaluations

    evaluated_at = str(change.get("evaluated_at") or _now())
    evaluations = change.get("evaluations", [change])
    if not isinstance(evaluations, list):
        raise ValueError("evaluation approval requires evaluations to be a list")
    save_evaluations(evaluated_at, evaluations)


def _handle_catalyst_status_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    catalyst_id = change.get("catalyst_id") or approval.get("entity_id")
    if not catalyst_id:
        raise ValueError("catalyst_status approval requires catalyst_id")
    now = _now()
    row = conn.execute("SELECT * FROM catalysts WHERE id = ?", (int(catalyst_id),)).fetchone()
    if not row:
        raise ValueError(f"No catalyst with id {catalyst_id}")
    updates = {"status": change.get("status", "played_out"), "updated_at": now}
    if change.get("evidence") is not None:
        updates["evidence"] = change.get("evidence")
    set_clause = ", ".join(f"{key} = ?" for key in updates)
    conn.execute(f"UPDATE catalysts SET {set_clause} WHERE id = ?", (*updates.values(), int(catalyst_id)))


def _handle_kill_condition_status_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    kc_id = change.get("kill_condition_id") or approval.get("entity_id")
    if not kc_id:
        raise ValueError("kill_condition_status approval requires kill_condition_id")
    now = _now()
    row = conn.execute("SELECT * FROM kill_conditions WHERE id = ?", (int(kc_id),)).fetchone()
    if not row:
        raise ValueError(f"No kill condition with id {kc_id}")
    current = _require_row_dict(row)
    status = change.get("status", "triggered")
    triggered_at = now if status == "triggered" else current.get("triggered_at")
    conn.execute(
        "UPDATE kill_conditions SET status = ?, triggered_at = ?, updated_at = ? WHERE id = ?",
        (status, triggered_at, now, int(kc_id)),
    )


def _handle_action_item_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    now = _now()
    conn.execute(
        "INSERT INTO action_items (ticker, action_type, description, urgency, source_type, source_id, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            _optional_ticker(change.get("ticker") or approval.get("ticker")),
            change.get("action_type", "review"),
            change.get("description", ""),
            change.get("urgency", "normal"),
            approval.get("source_type", "workflow"),
            approval.get("source_id"),
            now,
        ),
    )


def _handle_watch_trigger_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    now = _now()
    trigger_type = str(change.get("trigger_type") or "custom").strip().lower()
    if trigger_type not in WATCH_TRIGGER_TYPES:
        raise ValueError(f"Invalid watch trigger type: {trigger_type}")
    definition = change.get("definition")
    definition_json = json.dumps(definition, default=str) if definition else None
    conn.execute(
        "INSERT INTO watch_triggers (ticker, trigger_type, condition, source_type, source_id, created_at, expires_at, definition_json) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            _optional_ticker(change.get("ticker") or approval.get("ticker")),
            trigger_type,
            change.get("condition", ""),
            approval.get("source_type", "workflow"),
            approval.get("source_id"),
            now,
            change.get("expires_at"),
            definition_json,
        ),
    )


def _handle_portfolio_positions_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, callbacks
    from portfolio.action_registry import ActionContext, execute_action

    execute_action(
        "update_portfolio_positions",
        {"positions": change.get("positions") or []},
        ActionContext(
            actor_type="approval_apply",
            source_type=approval.get("source_type"),
            source_id=approval.get("source_id"),
            approval_id=int(approval["id"]),
        ),
    )


def _handle_hedge_positions_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, callbacks
    from portfolio.action_registry import ActionContext, execute_action

    execute_action(
        "update_hedge_positions",
        {"positions": change.get("positions") or []},
        ActionContext(
            actor_type="approval_apply",
            source_type=approval.get("source_type"),
            source_id=approval.get("source_id"),
            approval_id=int(approval["id"]),
        ),
    )


def _handle_thesis_content_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, callbacks
    from portfolio.action_registry import ActionContext, execute_action

    execute_action(
        "save_thesis_content",
        {
            "ticker": _required_ticker(change.get("ticker") or approval.get("ticker"), "thesis_content"),
            "content": change.get("content", ""),
            "preserve_exact_content": bool(change.get("preserve_exact_content") or False),
        },
        ActionContext(
            actor_type="approval_apply",
            source_type=approval.get("source_type"),
            source_id=approval.get("source_id"),
            approval_id=int(approval["id"]),
        ),
    )


def _handle_catalyst_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    now = _now()
    ticker = _required_ticker(change.get("ticker") or approval.get("ticker"), "catalyst")
    conn.execute(
        "INSERT INTO catalysts (ticker, description, category, target_date, evidence, created_at, updated_at, created_by) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            ticker,
            change.get("description", ""),
            change.get("category", "fundamental"),
            change.get("target_date"),
            change.get("evidence"),
            now,
            now,
            "agent",
        ),
    )
    callbacks.append(_sync_markdown_callback(ticker))


def _handle_kill_condition_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    now = _now()
    ticker = _required_ticker(change.get("ticker") or approval.get("ticker"), "kill_condition")
    conn.execute(
        "INSERT INTO kill_conditions (ticker, condition, metric, threshold, created_at, updated_at, created_by) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            ticker,
            change.get("condition", ""),
            change.get("metric"),
            change.get("threshold"),
            now,
            now,
            "agent",
        ),
    )
    callbacks.append(_sync_markdown_callback(ticker))


def _handle_research_note_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del callbacks
    now = _now()
    conn.execute(
        "INSERT INTO research_notes (ticker, title, content, note_type, source_type, source_id, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            _optional_ticker(change.get("ticker") or approval.get("ticker")),
            change.get("title", ""),
            change.get("content", ""),
            change.get("note_type", "general"),
            approval.get("source_type", "workflow"),
            approval.get("source_id"),
            now,
        ),
    )


def _handle_news_digest_delete_approval(
    conn: sqlite3.Connection | PostgresCompatConnection,
    approval: dict,
    change: dict,
    callbacks: list[ApprovalPostCommitCallback],
) -> None:
    del conn, approval
    from api.routers.portfolio_news import _delete_digest_index_best_effort
    from portfolio.news_digests import delete_digest, validate_digest_id

    digest_id = validate_digest_id(str(change.get("digest_id") or ""))
    deleted = delete_digest(digest_id)
    if deleted:

        def delete_digest_index() -> None:
            _delete_digest_index_best_effort(digest_id)

        callbacks.append(delete_digest_index)


_APPROVAL_SIDE_EFFECT_HANDLERS: dict[str, ApprovalSideEffectHandler] = {
    "thesis_status": _handle_thesis_status_approval,
    "evaluation": _handle_evaluation_approval,
    "catalyst_status": _handle_catalyst_status_approval,
    "kill_condition_status": _handle_kill_condition_status_approval,
    "action_item": _handle_action_item_approval,
    "watch_trigger": _handle_watch_trigger_approval,
    "portfolio_positions": _handle_portfolio_positions_approval,
    "hedge_positions": _handle_hedge_positions_approval,
    "thesis_content": _handle_thesis_content_approval,
    "catalyst": _handle_catalyst_approval,
    "kill_condition": _handle_kill_condition_approval,
    "research_note": _handle_research_note_approval,
    "news_digest_delete": _handle_news_digest_delete_approval,
}
