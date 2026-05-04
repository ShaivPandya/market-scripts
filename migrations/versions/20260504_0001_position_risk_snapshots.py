"""Add position risk snapshots table.

Revision ID: 20260504_0001
Revises: 20260503_0015
Create Date: 2026-05-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260504_0001"
down_revision: str | None = "20260503_0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_payload = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
    op.create_table(
        "position_risk_snapshots",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("as_of", sa.Text),
        sa.Column("computed_at", sa.Text, nullable=False),
        sa.Column("risk_score", sa.Float, nullable=False),
        sa.Column("risk_level", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("quality", sa.Text, nullable=False),
        sa.Column("source_status_json", json_payload, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("evidence_json", json_payload, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("degraded_modules_json", json_payload, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("input_snapshots_json", json_payload, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("payload_json", json_payload, nullable=False),
    )
    op.execute(
        "CREATE INDEX idx_position_risk_snapshots_ticker_time "
        "ON position_risk_snapshots (upper(ticker), computed_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_position_risk_snapshots_ticker_time")
    op.drop_table("position_risk_snapshots")
