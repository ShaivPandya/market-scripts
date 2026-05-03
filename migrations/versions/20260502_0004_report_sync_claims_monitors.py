"""Add report sync, thesis claims, and executable trigger metadata.

Revision ID: 20260502_0004
Revises: 20260502_0003
Create Date: 2026-05-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260502_0004"
down_revision: str | None = "20260502_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "report_runs",
        sa.Column("report_id", sa.Text, primary_key=True),
        sa.Column("report_type", sa.Text, nullable=False),
        sa.Column("as_of", sa.Text, nullable=False),
        sa.Column("source", sa.Text, nullable=False, server_default="github_actions"),
        sa.Column("source_run_id", sa.Text),
        sa.Column("source_url", sa.Text),
        sa.Column("status", sa.Text, nullable=False, server_default="completed"),
        sa.Column("report_hash", sa.Text),
        sa.Column("input_hash", sa.Text),
        sa.Column("summary_json", sa.Text),
        sa.Column("artifact_paths_json", sa.Text),
        sa.Column("issue_url", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
        sa.Column("synced_at", sa.Text, nullable=False),
        sa.Column("error", sa.Text),
    )
    op.create_index("idx_report_runs_type_asof", "report_runs", ["report_type", "as_of"])
    op.create_index("idx_report_runs_source", "report_runs", ["source_run_id"])

    op.create_table(
        "thesis_claims",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("claim", sa.Text, nullable=False),
        sa.Column("expected_evidence", sa.Text),
        sa.Column("disconfirming_evidence", sa.Text),
        sa.Column("source_requirements_json", sa.Text),
        sa.Column("cadence", sa.Text),
        sa.Column("confidence", sa.Float),
        sa.Column("status", sa.Text, nullable=False, server_default="active"),
        sa.Column("linked_catalyst_ids_json", sa.Text),
        sa.Column("linked_kill_condition_ids_json", sa.Text),
        sa.Column("source_type", sa.Text, nullable=False, server_default="user"),
        sa.Column("source_id", sa.Text),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
    )
    op.create_index("idx_thesis_claims_ticker", "thesis_claims", ["ticker"])
    op.create_index("idx_thesis_claims_status", "thesis_claims", ["status"])

    op.add_column("watch_triggers", sa.Column("definition_json", sa.Text))
    op.add_column("watch_triggers", sa.Column("last_checked_at", sa.Text))
    op.add_column("watch_triggers", sa.Column("last_result_json", sa.Text))
    op.add_column("watch_triggers", sa.Column("last_evidence", sa.Text))

    op.add_column("recommendations", sa.Column("report_id", sa.Text))
    op.add_column("recommendations", sa.Column("idempotency_key", sa.Text))
    op.create_index("idx_recommendations_report_id", "recommendations", ["report_id"])
    op.create_index(
        "uq_recommendations_idempotency",
        "recommendations",
        ["idempotency_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_recommendations_idempotency", table_name="recommendations")
    op.drop_index("idx_recommendations_report_id", table_name="recommendations")
    op.drop_column("recommendations", "idempotency_key")
    op.drop_column("recommendations", "report_id")

    op.drop_column("watch_triggers", "last_evidence")
    op.drop_column("watch_triggers", "last_result_json")
    op.drop_column("watch_triggers", "last_checked_at")
    op.drop_column("watch_triggers", "definition_json")

    op.drop_index("idx_thesis_claims_status", table_name="thesis_claims")
    op.drop_index("idx_thesis_claims_ticker", table_name="thesis_claims")
    op.drop_table("thesis_claims")

    op.drop_index("idx_report_runs_source", table_name="report_runs")
    op.drop_index("idx_report_runs_type_asof", table_name="report_runs")
    op.drop_table("report_runs")
