"""Add recommendation ledger table.

Revision ID: 20260502_0001
Revises: 20260430_0001
Create Date: 2026-05-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260502_0001"
down_revision: str | None = "20260430_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "recommendations",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("report_type", sa.Text, nullable=False),
        sa.Column("as_of", sa.Text, nullable=False),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("source_report_path", sa.Text),
        sa.Column("source_json_path", sa.Text),
        sa.Column("stance", sa.Text, nullable=False),
        sa.Column("recommendation_status", sa.Text, nullable=False),
        sa.Column("critical_data_quality", sa.Text, nullable=False),
        sa.Column("blocked_reasons_json", sa.Text),
        sa.Column("what_changed_json", sa.Text),
        sa.Column("do_nothing_rationale", sa.Text),
        sa.Column("action", sa.Text, nullable=False),
        sa.Column("ticker", sa.Text),
        sa.Column("instrument", sa.Text, nullable=False),
        sa.Column("horizon", sa.Text),
        sa.Column("target_change", sa.Text),
        sa.Column("rationale", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column("source_quality", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="open"),
        sa.Column("evidence_json", sa.Text),
        sa.Column("disconfirming_evidence_json", sa.Text),
        sa.Column("catalyst", sa.Text),
        sa.Column("invalidation", sa.Text),
        sa.Column("expected_onset_window", sa.Text),
        sa.Column("alternatives_json", sa.Text),
        sa.Column("opportunity_cost_json", sa.Text),
        sa.Column("approval_id", sa.Integer),
        sa.Column("approval_status", sa.Text, nullable=False, server_default="none"),
        sa.Column("outcome_status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("outcome_json", sa.Text),
        sa.Column("model", sa.Text),
        sa.Column("prompt_hash", sa.Text),
        sa.Column("input_hash", sa.Text),
        sa.Column("validation_status", sa.Text),
        sa.Column("source_quality_summary_json", sa.Text),
    )
    op.create_index("idx_recommendations_report", "recommendations", ["report_type", "as_of"])
    op.create_index("idx_recommendations_ticker", "recommendations", ["ticker"])
    op.create_index("idx_recommendations_status", "recommendations", ["status"])
    op.create_index("idx_recommendations_approval", "recommendations", ["approval_status"])
    op.create_index("idx_recommendations_outcome", "recommendations", ["outcome_status"])


def downgrade() -> None:
    op.drop_index("idx_recommendations_outcome", table_name="recommendations")
    op.drop_index("idx_recommendations_approval", table_name="recommendations")
    op.drop_index("idx_recommendations_status", table_name="recommendations")
    op.drop_index("idx_recommendations_ticker", table_name="recommendations")
    op.drop_index("idx_recommendations_report", table_name="recommendations")
    op.drop_table("recommendations")
