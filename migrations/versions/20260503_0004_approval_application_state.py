"""Add approval application state.

Revision ID: 20260503_0004
Revises: 20260503_0003
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260503_0004"
down_revision: str | None = "20260503_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APPLICATION_STATUS_CHECK = "application_status IN ('pending', 'applying', 'applied', 'failed', 'not_applicable')"


def upgrade() -> None:
    op.add_column(
        "pending_approvals",
        sa.Column("application_status", sa.Text(), nullable=False, server_default="pending"),
    )
    op.add_column(
        "pending_approvals",
        sa.Column("application_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("pending_approvals", sa.Column("application_started_at", sa.Text()))
    op.add_column("pending_approvals", sa.Column("application_completed_at", sa.Text()))
    op.add_column("pending_approvals", sa.Column("application_error", sa.Text()))
    op.create_check_constraint(
        "ck_pending_approvals_application_status",
        "pending_approvals",
        _APPLICATION_STATUS_CHECK,
    )
    op.execute(
        "UPDATE pending_approvals "
        "SET application_status = 'applied', "
        "application_completed_at = COALESCE(application_completed_at, resolved_at, created_at) "
        "WHERE status = 'approved' AND application_status = 'pending'"
    )
    op.execute(
        "UPDATE pending_approvals "
        "SET application_status = 'not_applicable', "
        "application_completed_at = COALESCE(application_completed_at, resolved_at, created_at) "
        "WHERE status IN ('rejected', 'expired') AND application_status = 'pending'"
    )
    op.create_index(
        "idx_pending_approvals_application_status",
        "pending_approvals",
        ["application_status"],
    )


def downgrade() -> None:
    op.drop_index("idx_pending_approvals_application_status", table_name="pending_approvals")
    op.drop_constraint(
        "ck_pending_approvals_application_status",
        "pending_approvals",
        type_="check",
    )
    op.drop_column("pending_approvals", "application_error")
    op.drop_column("pending_approvals", "application_completed_at")
    op.drop_column("pending_approvals", "application_started_at")
    op.drop_column("pending_approvals", "application_attempts")
    op.drop_column("pending_approvals", "application_status")
