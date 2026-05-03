"""Backfill pending approval action metadata columns.

Revision ID: 20260503_0008
Revises: 20260503_0007
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260503_0008"
down_revision: str | None = "20260503_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS action_id text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS action_schema_name text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS action_schema_version integer")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS action_input_hash text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS request_schema_name text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS request_schema_version integer")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pending_approvals_action_id ON pending_approvals(action_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pending_approvals_action_id")
