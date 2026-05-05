"""Add comparative idea ranking runs.

Revision ID: 20260505_0005
Revises: 20260505_0004
Create Date: 2026-05-05
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260505_0005"
down_revision: str | None = "20260505_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _grant_postgres() -> None:
    op.execute(
        """
        DO $$
        DECLARE
            table_name text;
            sequence_name text;
        BEGIN
            FOREACH table_name IN ARRAY ARRAY['idea_comparison_runs', 'idea_comparison_rankings']
            LOOP
                sequence_name := pg_get_serial_sequence(table_name, 'id');

                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'talisman_app') THEN
                    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON %I TO talisman_app', table_name);
                    IF sequence_name IS NOT NULL THEN
                        EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE %s TO talisman_app', sequence_name);
                    END IF;
                END IF;

                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'talisman_worker') THEN
                    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON %I TO talisman_worker', table_name);
                    IF sequence_name IS NOT NULL THEN
                        EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE %s TO talisman_worker', sequence_name);
                    END IF;
                END IF;
            END LOOP;
        END $$;
        """
    )


def upgrade() -> None:
    op.create_table(
        "idea_comparison_runs",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("run_id", sa.Text, nullable=False),
        sa.Column("job_id", sa.Text),
        sa.Column("scope_statuses_json", sa.Text, nullable=False, server_default="[]"),
        sa.Column("summary", sa.Text, nullable=False, server_default=""),
        sa.Column("ranking_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("raw_result_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.UniqueConstraint("run_id", name="uq_idea_comparison_runs_run_id"),
    )
    op.create_table(
        "idea_comparison_rankings",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("run_id", sa.Text, nullable=False),
        sa.Column("idea_id", sa.Integer, nullable=False),
        sa.Column("evaluation_id", sa.Integer, nullable=False),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("action", sa.Text, nullable=False),
        sa.Column("score", sa.Float),
        sa.Column("confidence", sa.Float),
        sa.Column("confidence_level", sa.Text, nullable=False),
        sa.Column("rationale", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.CheckConstraint("action IN ('buy', 'watch', 'avoid', 'do_nothing')", name="ck_idea_comparison_action"),
        sa.CheckConstraint(
            "confidence_level IN ('high', 'medium', 'low')",
            name="ck_idea_comparison_confidence_level",
        ),
    )
    for name, table, columns in [
        ("idx_idea_comparison_runs_created", "idea_comparison_runs", ["created_at"]),
        ("idx_idea_comparison_runs_job", "idea_comparison_runs", ["job_id"]),
        ("idx_idea_comparison_rankings_run_rank", "idea_comparison_rankings", ["run_id", "rank"]),
        ("idx_idea_comparison_rankings_idea", "idea_comparison_rankings", ["idea_id"]),
    ]:
        op.create_index(name, table, columns)
    if op.get_bind().dialect.name == "postgresql":
        _grant_postgres()


def downgrade() -> None:
    for name, table in [
        ("idx_idea_comparison_rankings_idea", "idea_comparison_rankings"),
        ("idx_idea_comparison_rankings_run_rank", "idea_comparison_rankings"),
        ("idx_idea_comparison_runs_job", "idea_comparison_runs"),
        ("idx_idea_comparison_runs_created", "idea_comparison_runs"),
    ]:
        op.drop_index(name, table_name=table)
    op.drop_table("idea_comparison_rankings")
    op.drop_table("idea_comparison_runs")
