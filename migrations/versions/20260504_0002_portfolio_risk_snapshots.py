"""Add portfolio risk snapshots table and runtime grants.

Revision ID: 20260504_0002
Revises: 20260504_0001
Create Date: 2026-05-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260504_0002"
down_revision: str | None = "20260504_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_risk_snapshots (
                id text PRIMARY KEY,
                as_of text,
                computed_at text NOT NULL,
                average_risk_score double precision NOT NULL,
                max_risk_score double precision NOT NULL,
                confidence double precision NOT NULL,
                quality text NOT NULL,
                position_count integer NOT NULL,
                source_status_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                degraded_modules_json jsonb NOT NULL DEFAULT '[]'::jsonb,
                input_snapshots_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                position_snapshot_ids_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                payload_json jsonb NOT NULL
            )
            """
        )
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_portfolio_risk_snapshots_time
            ON portfolio_risk_snapshots (computed_at DESC)
            """
        )
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'talisman_app') THEN
                    GRANT SELECT, INSERT, UPDATE, DELETE ON position_risk_snapshots TO talisman_app;
                    GRANT SELECT, INSERT, UPDATE, DELETE ON portfolio_risk_snapshots TO talisman_app;
                END IF;
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'talisman_worker') THEN
                    GRANT SELECT, INSERT, UPDATE, DELETE ON position_risk_snapshots TO talisman_worker;
                    GRANT SELECT, INSERT, UPDATE, DELETE ON portfolio_risk_snapshots TO talisman_worker;
                END IF;
            END $$;
            """
        )
        return

    json_payload = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
    op.create_table(
        "portfolio_risk_snapshots",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("as_of", sa.Text),
        sa.Column("computed_at", sa.Text, nullable=False),
        sa.Column("average_risk_score", sa.Float, nullable=False),
        sa.Column("max_risk_score", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("quality", sa.Text, nullable=False),
        sa.Column("position_count", sa.Integer, nullable=False),
        sa.Column("source_status_json", json_payload, nullable=False),
        sa.Column("degraded_modules_json", json_payload, nullable=False),
        sa.Column("input_snapshots_json", json_payload, nullable=False),
        sa.Column("position_snapshot_ids_json", json_payload, nullable=False),
        sa.Column("payload_json", json_payload, nullable=False),
    )
    op.create_index("idx_portfolio_risk_snapshots_time", "portfolio_risk_snapshots", ["computed_at"])


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_portfolio_risk_snapshots_time")
    op.drop_table("portfolio_risk_snapshots")
