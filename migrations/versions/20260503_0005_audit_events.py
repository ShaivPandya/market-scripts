"""Add append-only audit events.

Revision ID: 20260503_0005
Revises: 20260503_0004
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260503_0005"
down_revision: str | None = "20260503_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("event_id", sa.Text(), nullable=False, unique=True),
        sa.Column("occurred_at", sa.Text(), nullable=False),
        sa.Column("received_at", sa.Text(), nullable=False),
        sa.Column("request_id", sa.Text()),
        sa.Column("actor_id", sa.Text()),
        sa.Column("actor_type", sa.Text(), nullable=False, server_default="system"),
        sa.Column("parent_actor_id", sa.Text()),
        sa.Column("action_name", sa.Text(), nullable=False),
        sa.Column("action_category", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("object_type", sa.Text()),
        sa.Column("object_id", sa.Text()),
        sa.Column("object_refs_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("before_summary_json", sa.Text()),
        sa.Column("after_summary_json", sa.Text()),
        sa.Column("source_lineage_json", sa.Text()),
        sa.Column("metadata_json", sa.Text()),
        sa.Column("error", sa.Text()),
    )
    op.create_index("idx_audit_events_occurred_at", "audit_events", ["occurred_at"])
    op.create_index("idx_audit_events_request", "audit_events", ["request_id"])
    op.create_index("idx_audit_events_actor_time", "audit_events", ["actor_id", "occurred_at"])
    op.create_index("idx_audit_events_action_time", "audit_events", ["action_name", "occurred_at"])
    op.create_index("idx_audit_events_object_time", "audit_events", ["object_type", "object_id", "occurred_at"])
    op.create_index("idx_audit_events_status_time", "audit_events", ["status", "occurred_at"])


def downgrade() -> None:
    op.drop_index("idx_audit_events_status_time", table_name="audit_events")
    op.drop_index("idx_audit_events_object_time", table_name="audit_events")
    op.drop_index("idx_audit_events_action_time", table_name="audit_events")
    op.drop_index("idx_audit_events_actor_time", table_name="audit_events")
    op.drop_index("idx_audit_events_request", table_name="audit_events")
    op.drop_index("idx_audit_events_occurred_at", table_name="audit_events")
    op.drop_table("audit_events")
