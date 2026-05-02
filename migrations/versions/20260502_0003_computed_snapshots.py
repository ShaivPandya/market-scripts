"""Add computed snapshots table.

Revision ID: 20260502_0003
Revises: 20260502_0002
Create Date: 2026-05-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260502_0003"
down_revision: str | None = "20260502_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_payload = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
    op.create_table(
        "computed_snapshots",
        sa.Column("snapshot_key", sa.Text, primary_key=True),
        sa.Column("payload_json", json_payload, nullable=True),
        sa.Column("as_of_date", sa.Text),
        sa.Column("fetched_at", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("error", sa.Text),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("artifact_uri", sa.Text),
    )
    op.create_index("idx_computed_snapshots_status", "computed_snapshots", ["status"])


def downgrade() -> None:
    op.drop_index("idx_computed_snapshots_status", table_name="computed_snapshots")
    op.drop_table("computed_snapshots")
