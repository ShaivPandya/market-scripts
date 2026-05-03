"""Add unified provenance tables.

Revision ID: 20260503_0009
Revises: 20260503_0008
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260503_0009"
down_revision: str | None = "20260503_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "provenance_events",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("event_name", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("started_at", sa.Text, nullable=False),
        sa.Column("completed_at", sa.Text),
        sa.Column("actor_type", sa.Text),
        sa.Column("actor_id", sa.Text),
        sa.Column("parent_actor_id", sa.Text),
        sa.Column("request_id", sa.Text),
        sa.Column("parent_event_id", sa.Text),
        sa.Column("workflow_run_id", sa.Text),
        sa.Column("ontology_run_id", sa.Text),
        sa.Column("agent_session_id", sa.Text),
        sa.Column("action_run_id", sa.Integer),
        sa.Column("approval_id", sa.Integer),
        sa.Column("audit_event_id", sa.Text),
        sa.Column("input_hash", sa.Text),
        sa.Column("output_hash", sa.Text),
        sa.Column("summary_json", sa.Text),
        sa.Column("metadata_json", sa.Text),
        sa.Column("redaction_policy", sa.Text, nullable=False, server_default="audit_summary_v1"),
        sa.Column("retention_class", sa.Text, nullable=False, server_default="provenance_365d"),
        sa.Column("error", sa.Text),
    )
    op.create_index("idx_provenance_events_type_time", "provenance_events", ["event_type", "started_at"])
    op.create_index("idx_provenance_events_workflow", "provenance_events", ["workflow_run_id"])
    op.create_index("idx_provenance_events_ontology", "provenance_events", ["ontology_run_id"])
    op.create_index("idx_provenance_events_agent_session", "provenance_events", ["agent_session_id"])
    op.create_index("idx_provenance_events_action_run", "provenance_events", ["action_run_id"])
    op.create_index("idx_provenance_events_approval", "provenance_events", ["approval_id"])
    op.create_index("idx_provenance_events_parent", "provenance_events", ["parent_event_id"])

    op.create_table(
        "provenance_links",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("event_id", sa.Text, nullable=False),
        sa.Column("source_ref_type", sa.Text, nullable=False),
        sa.Column("source_ref_id", sa.Text, nullable=False),
        sa.Column("source_ref_version", sa.Text),
        sa.Column("target_ref_type", sa.Text, nullable=False),
        sa.Column("target_ref_id", sa.Text, nullable=False),
        sa.Column("target_ref_version", sa.Text),
        sa.Column("link_type", sa.Text, nullable=False),
        sa.Column("metadata_json", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
    )
    op.create_index("idx_provenance_links_event", "provenance_links", ["event_id"])
    op.create_index("idx_provenance_links_source", "provenance_links", ["source_ref_type", "source_ref_id"])
    op.create_index("idx_provenance_links_target", "provenance_links", ["target_ref_type", "target_ref_id"])

    op.create_table(
        "source_record_refs",
        sa.Column("record_ref_id", sa.Text, primary_key=True),
        sa.Column("adapter_run_event_id", sa.Text, nullable=False),
        sa.Column("source_name", sa.Text, nullable=False),
        sa.Column("record_kind", sa.Text, nullable=False),
        sa.Column("record_key_hash", sa.Text, nullable=False),
        sa.Column("record_hash", sa.Text, nullable=False),
        sa.Column("as_of", sa.Text),
        sa.Column("summary_json", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
    )
    op.create_index("idx_source_record_refs_adapter", "source_record_refs", ["adapter_run_event_id"])
    op.create_index("idx_source_record_refs_source", "source_record_refs", ["source_name", "record_kind"])

    op.create_table(
        "workflow_artifact_records",
        sa.Column("artifact_id", sa.Text, primary_key=True),
        sa.Column("workflow_run_id", sa.Text, nullable=False),
        sa.Column("artifact_key", sa.Text, nullable=False),
        sa.Column("artifact_index", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("artifact_hash", sa.Text, nullable=False),
        sa.Column("summary_json", sa.Text),
        sa.Column("approval_id", sa.Integer),
        sa.Column("provenance_event_id", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
    )
    op.create_index("idx_workflow_artifact_records_run", "workflow_artifact_records", ["workflow_run_id"])
    op.create_index("idx_workflow_artifact_records_approval", "workflow_artifact_records", ["approval_id"])

    op.execute("ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS provenance_event_id text")
    op.execute("ALTER TABLE action_runs ADD COLUMN IF NOT EXISTS provenance_event_id text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS provenance_event_id text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS origin_provenance_event_id text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS origin_artifact_id text")
    op.execute("ALTER TABLE ontology_runs ADD COLUMN IF NOT EXISTS provenance_event_id text")


def downgrade() -> None:
    op.execute("ALTER TABLE ontology_runs DROP COLUMN IF EXISTS provenance_event_id")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS origin_artifact_id")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS origin_provenance_event_id")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS provenance_event_id")
    op.execute("ALTER TABLE action_runs DROP COLUMN IF EXISTS provenance_event_id")
    op.execute("ALTER TABLE workflow_runs DROP COLUMN IF EXISTS provenance_event_id")

    op.drop_index("idx_workflow_artifact_records_approval", table_name="workflow_artifact_records")
    op.drop_index("idx_workflow_artifact_records_run", table_name="workflow_artifact_records")
    op.drop_table("workflow_artifact_records")
    op.drop_index("idx_source_record_refs_source", table_name="source_record_refs")
    op.drop_index("idx_source_record_refs_adapter", table_name="source_record_refs")
    op.drop_table("source_record_refs")
    op.drop_index("idx_provenance_links_target", table_name="provenance_links")
    op.drop_index("idx_provenance_links_source", table_name="provenance_links")
    op.drop_index("idx_provenance_links_event", table_name="provenance_links")
    op.drop_table("provenance_links")
    op.drop_index("idx_provenance_events_parent", table_name="provenance_events")
    op.drop_index("idx_provenance_events_approval", table_name="provenance_events")
    op.drop_index("idx_provenance_events_action_run", table_name="provenance_events")
    op.drop_index("idx_provenance_events_agent_session", table_name="provenance_events")
    op.drop_index("idx_provenance_events_ontology", table_name="provenance_events")
    op.drop_index("idx_provenance_events_workflow", table_name="provenance_events")
    op.drop_index("idx_provenance_events_type_time", table_name="provenance_events")
    op.drop_table("provenance_events")
