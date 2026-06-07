"""Add human-reviewed agent response feedback store.

Revision ID: 20260607_0002
Revises: 20260607_0001
Create Date: 2026-06-07
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260607_0002"
down_revision: str | None = "20260607_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.create_table(
        "agent_response_feedback",
        sa.Column("feedback_id", sa.Text, primary_key=True),
        sa.Column("trajectory_id", sa.Text, nullable=False),
        sa.Column("session_id", sa.Text),
        sa.Column("client_turn_id", sa.Text),
        sa.Column("response_version", sa.Text, nullable=False),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column("decision", sa.Text, nullable=False),
        sa.Column("reviewer_actor_id", sa.Text, nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model", sa.Text),
        sa.Column("provider", sa.Text),
        sa.Column("corrected_response", sa.Text),
        sa.Column("failure_tags_json", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("notes", sa.Text),
        sa.Column("training_eligible", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("signal_source", sa.Text, nullable=False, server_default=sa.text("'human_reviewed'")),
        sa.Column("provenance_json", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("retention_class", sa.Text, nullable=False, server_default=sa.text("'agent_feedback_365d'")),
        sa.Column("tombstoned_at", sa.DateTime(timezone=True)),
        sa.Column("deletion_reason", sa.Text),
        sa.UniqueConstraint(
            "trajectory_id",
            "reviewer_actor_id",
            "response_version",
            name="uq_agent_response_feedback_reviewer_turn",
        ),
    )
    op.create_index(
        "idx_agent_response_feedback_trajectory",
        "agent_response_feedback",
        ["trajectory_id"],
    )
    op.create_index(
        "idx_agent_response_feedback_session_turn",
        "agent_response_feedback",
        ["session_id", "client_turn_id"],
    )
    op.create_index(
        "idx_agent_response_feedback_reviewed_at",
        "agent_response_feedback",
        ["reviewed_at"],
    )
    op.create_index(
        "idx_agent_response_feedback_training",
        "agent_response_feedback",
        ["training_eligible", "reviewed_at"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.drop_index("idx_agent_response_feedback_training", table_name="agent_response_feedback")
    op.drop_index("idx_agent_response_feedback_reviewed_at", table_name="agent_response_feedback")
    op.drop_index("idx_agent_response_feedback_session_turn", table_name="agent_response_feedback")
    op.drop_index("idx_agent_response_feedback_trajectory", table_name="agent_response_feedback")
    op.drop_table("agent_response_feedback")
