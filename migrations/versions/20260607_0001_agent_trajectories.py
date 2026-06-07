"""Add training-grade agent trajectory store.

Revision ID: 20260607_0001
Revises: 20260602_0001
Create Date: 2026-06-07
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260607_0001"
down_revision: str | None = "20260602_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.create_table(
        "agent_trajectories",
        sa.Column("trajectory_id", sa.Text, primary_key=True),
        sa.Column("session_id", sa.Text),
        sa.Column("client_turn_id", sa.Text),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column("final_disposition", sa.Text, nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("provider", sa.Text),
        sa.Column("model", sa.Text),
        sa.Column("prompt_version", sa.Text),
        sa.Column("code_version", sa.Text),
        sa.Column("sensitivity", sa.Text, nullable=False, server_default=sa.text("'operational_private'")),
        sa.Column("consent_state", sa.Text, nullable=False, server_default=sa.text("'not_requested'")),
        sa.Column("training_eligible", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("exclusion_reasons_json", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("dataset_split_group", sa.Text, nullable=False),
        sa.Column(
            "redaction_policy", sa.Text, nullable=False, server_default=sa.text("'agent_trajectory_training_v1'")
        ),
        sa.Column("retention_class", sa.Text, nullable=False, server_default=sa.text("'agent_trajectory_365d'")),
        sa.Column("redaction_manifest_json", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("source_provenance_json", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("provenance_json", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("raw_payload_json", JSONB, nullable=False),
        sa.Column("sanitized_payload_json", JSONB, nullable=False),
        sa.Column("tombstoned_at", sa.DateTime(timezone=True)),
        sa.Column("deletion_reason", sa.Text),
    )
    op.create_index("idx_agent_trajectories_captured_at", "agent_trajectories", ["captured_at"])
    op.create_index(
        "idx_agent_trajectories_session_turn",
        "agent_trajectories",
        ["session_id", "client_turn_id"],
    )
    op.create_index(
        "idx_agent_trajectories_training",
        "agent_trajectories",
        ["training_eligible", "captured_at"],
    )
    op.create_index(
        "idx_agent_trajectories_split_group",
        "agent_trajectories",
        ["dataset_split_group"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.drop_index("idx_agent_trajectories_split_group", table_name="agent_trajectories")
    op.drop_index("idx_agent_trajectories_training", table_name="agent_trajectories")
    op.drop_index("idx_agent_trajectories_session_turn", table_name="agent_trajectories")
    op.drop_index("idx_agent_trajectories_captured_at", table_name="agent_trajectories")
    op.drop_table("agent_trajectories")
