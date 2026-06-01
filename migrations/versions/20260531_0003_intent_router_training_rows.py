"""Add intent router supervised training row store.

Revision ID: 20260531_0003
Revises: 20260531_0002
Create Date: 2026-05-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260531_0003"
down_revision: str | None = "20260531_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.create_table(
        "intent_router_training_rows",
        sa.Column("row_id", sa.Text, primary_key=True),
        sa.Column("session_id", sa.Text),
        sa.Column("client_turn_id", sa.Text),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("schema_version", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column("capture_policy", sa.Text, nullable=False, server_default=sa.text("'shadow_all'")),
        sa.Column("redaction_policy", sa.Text, nullable=False, server_default=sa.text("'router_training_v1'")),
        sa.Column("retention_class", sa.Text, nullable=False, server_default=sa.text("'router_training_365d'")),
        sa.Column("sampling_reason", sa.Text),
        sa.Column("applied_source", sa.Text),
        sa.Column("confidence_threshold", sa.Float),
        sa.Column("fallback_reason", sa.Text),
        sa.Column("user_text", sa.Text, nullable=False),
        sa.Column("screen_context_json", JSONB),
        sa.Column("recent_session_features_json", JSONB),
        sa.Column("regex_baseline_json", JSONB),
        sa.Column("llm_candidate_json", JSONB),
        sa.Column("shadow_comparison_json", JSONB),
        sa.Column("applied_route_json", JSONB),
        sa.Column("opportunity_candidate_metadata_json", JSONB),
        sa.Column("payload_json", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("label_intent_class", sa.Text),
        sa.Column("label_run_hidden_dq", sa.Boolean),
        sa.Column("label_run_opportunity_preflight", sa.Boolean),
        sa.Column("label_workflow_name", sa.Text),
        sa.Column("label_tool_names_json", JSONB),
        sa.Column("label_reviewer", sa.Text),
        sa.Column("labeled_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "idx_intent_router_training_rows_captured_at",
        "intent_router_training_rows",
        ["captured_at"],
    )
    op.create_index(
        "idx_intent_router_training_rows_session_turn",
        "intent_router_training_rows",
        ["session_id", "client_turn_id"],
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_intent_router_training_rows_session_turn
        ON intent_router_training_rows (session_id, client_turn_id)
        WHERE session_id IS NOT NULL AND client_turn_id IS NOT NULL
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS uq_intent_router_training_rows_session_turn")
    op.drop_index("idx_intent_router_training_rows_session_turn", table_name="intent_router_training_rows")
    op.drop_index("idx_intent_router_training_rows_captured_at", table_name="intent_router_training_rows")
    op.drop_table("intent_router_training_rows")
