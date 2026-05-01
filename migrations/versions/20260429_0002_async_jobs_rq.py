"""Add RQ-backed async job metadata.

Revision ID: 20260429_0002
Revises: 20260429_0001
Create Date: 2026-04-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260429_0002"
down_revision: str | None = "20260429_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("async_jobs", sa.Column("cache_key", sa.Text))
    op.add_column("async_jobs", sa.Column("progress_json", postgresql.JSONB))
    op.add_column("async_jobs", sa.Column("result_expires_at", sa.DateTime(timezone=True)))
    op.add_column("async_jobs", sa.Column("queue_name", sa.Text))
    op.add_column("async_jobs", sa.Column("rq_job_id", sa.Text))

    op.create_index(
        "uq_async_jobs_active_dedupe",
        "async_jobs",
        ["job_type", "cache_key"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running') AND cache_key IS NOT NULL"),
    )
    op.create_index(
        "idx_async_jobs_completed_cache",
        "async_jobs",
        ["job_type", "cache_key", "result_expires_at"],
        postgresql_where=sa.text("status = 'completed' AND cache_key IS NOT NULL"),
    )
    op.create_index(
        "idx_async_jobs_result_expires",
        "async_jobs",
        ["result_expires_at"],
        postgresql_where=sa.text("status IN ('completed', 'failed') AND result_expires_at IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_async_jobs_result_expires", table_name="async_jobs")
    op.drop_index("idx_async_jobs_completed_cache", table_name="async_jobs")
    op.drop_index("uq_async_jobs_active_dedupe", table_name="async_jobs")
    op.drop_column("async_jobs", "rq_job_id")
    op.drop_column("async_jobs", "queue_name")
    op.drop_column("async_jobs", "result_expires_at")
    op.drop_column("async_jobs", "progress_json")
    op.drop_column("async_jobs", "cache_key")
