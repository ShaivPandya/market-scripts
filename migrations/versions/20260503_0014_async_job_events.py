"""Add async job event replay table.

Revision ID: 20260503_0014
Revises: 20260503_0013
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260503_0014"
down_revision: str | None = "20260503_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "async_job_events",
        sa.Column("job_id", sa.Text, nullable=False),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("payload_json", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["async_jobs.job_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id", "seq"),
    )
    op.create_index("idx_async_job_events_job_seq", "async_job_events", ["job_id", "seq"])


def downgrade() -> None:
    op.drop_index("idx_async_job_events_job_seq", table_name="async_job_events")
    op.drop_table("async_job_events")
