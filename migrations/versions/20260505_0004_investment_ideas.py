"""Add investment idea watchlist tables.

Revision ID: 20260505_0004
Revises: 20260505_0003
Create Date: 2026-05-05
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260505_0004"
down_revision: str | None = "20260505_0003"
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
            FOREACH table_name IN ARRAY ARRAY['investment_ideas', 'idea_evaluations']
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
        "investment_ideas",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("company_name", sa.Text),
        sa.Column("status", sa.Text, nullable=False, server_default="watching"),
        sa.Column("user_notes", sa.Text, nullable=False, server_default=""),
        sa.Column("tags_json", sa.Text, nullable=False, server_default="[]"),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
        sa.Column("source_type", sa.Text, nullable=False, server_default="user"),
        sa.Column("source_id", sa.Text),
        sa.Column("latest_evaluation_id", sa.Integer),
        sa.Column("latest_job_id", sa.Text),
        sa.Column("accepted_recommendation_id", sa.Integer),
        sa.Column("metadata_json", sa.Text, nullable=False, server_default="{}"),
        sa.CheckConstraint(
            "status IN ('watching', 'researching', 'ready_for_review', 'accepted', 'rejected', 'archived')",
            name="ck_investment_ideas_status",
        ),
        sa.CheckConstraint("source_type IN ('workflow', 'agent', 'user')", name="ck_investment_ideas_source_type"),
    )
    op.create_table(
        "idea_evaluations",
        sa.Column("id", sa.Integer, sa.Identity(), primary_key=True),
        sa.Column("idea_id", sa.Integer, nullable=False),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("job_id", sa.Text),
        sa.Column("evaluated_at", sa.Text, nullable=False),
        sa.Column("action", sa.Text, nullable=False),
        sa.Column("recommendation_status", sa.Text, nullable=False, server_default="clear"),
        sa.Column("score", sa.Float),
        sa.Column("confidence", sa.Float),
        sa.Column("thesis_statement", sa.Text),
        sa.Column("rationale", sa.Text, nullable=False, server_default=""),
        sa.Column("factor_scores_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("missing_information_json", sa.Text, nullable=False, server_default="[]"),
        sa.Column("data_quality_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("evidence_json", sa.Text, nullable=False, server_default="[]"),
        sa.Column("disconfirming_evidence_json", sa.Text, nullable=False, server_default="[]"),
        sa.Column("catalyst", sa.Text),
        sa.Column("invalidation", sa.Text),
        sa.Column("portfolio_fit_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("recommendation_record_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("recommendation_id", sa.Integer),
        sa.Column("approval_id", sa.Integer),
        sa.Column("action_approval_id", sa.Integer),
        sa.Column("accepted_at", sa.Text),
        sa.Column("accepted_by", sa.Text),
        sa.Column("raw_result_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.CheckConstraint("action IN ('buy', 'watch', 'avoid', 'do_nothing')", name="ck_idea_evaluations_action"),
        sa.CheckConstraint(
            "recommendation_status IN ('clear', 'review_required', 'blocked', 'error')",
            name="ck_idea_evaluations_recommendation_status",
        ),
    )
    for name, table, columns in [
        ("idx_investment_ideas_ticker", "investment_ideas", ["ticker"]),
        ("idx_investment_ideas_status", "investment_ideas", ["status"]),
        ("idx_investment_ideas_latest_eval", "investment_ideas", ["latest_evaluation_id"]),
        ("idx_idea_evaluations_idea_created", "idea_evaluations", ["idea_id", "created_at"]),
        ("idx_idea_evaluations_ticker_created", "idea_evaluations", ["ticker", "created_at"]),
        ("idx_idea_evaluations_job", "idea_evaluations", ["job_id"]),
    ]:
        op.create_index(name, table, columns)
    if op.get_bind().dialect.name == "postgresql":
        _grant_postgres()


def downgrade() -> None:
    for name, table in [
        ("idx_idea_evaluations_job", "idea_evaluations"),
        ("idx_idea_evaluations_ticker_created", "idea_evaluations"),
        ("idx_idea_evaluations_idea_created", "idea_evaluations"),
        ("idx_investment_ideas_latest_eval", "investment_ideas"),
        ("idx_investment_ideas_status", "investment_ideas"),
        ("idx_investment_ideas_ticker", "investment_ideas"),
    ]:
        op.drop_index(name, table_name=table)
    op.drop_table("idea_evaluations")
    op.drop_table("investment_ideas")
