"""Add pending approval resolution metadata columns.

Revision ID: 20260503_0015
Revises: 20260503_0014
Create Date: 2026-05-04
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260503_0015"
down_revision: str | None = "20260503_0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS risk_class text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS approval_mode text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS base_state_hash text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS requested_by_actor_id text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS resolved_by_actor_id text")
    op.execute(
        "ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS approval_note_required integer NOT NULL DEFAULT 0"
    )
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS reason_code text")
    op.execute("ALTER TABLE pending_approvals ADD COLUMN IF NOT EXISTS supersedes_approval_id integer")


def downgrade() -> None:
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS supersedes_approval_id")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS reason_code")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS approval_note_required")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS resolved_by_actor_id")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS requested_by_actor_id")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS base_state_hash")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS approval_mode")
    op.execute("ALTER TABLE pending_approvals DROP COLUMN IF EXISTS risk_class")
