"""Add first-party users, roles, and opaque auth sessions.

Revision ID: 20260602_0001
Revises: 20260531_0004
Create Date: 2026-06-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260602_0001"
down_revision: str | None = "20260531_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "auth_users",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("username", sa.Text, nullable=False),
        sa.Column("password_hash", sa.Text, nullable=True),
        sa.Column("email", sa.Text, nullable=True),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
    )
    op.create_index("ix_auth_users_username", "auth_users", ["username"], unique=True)
    op.create_index("ix_auth_users_email", "auth_users", ["email"], unique=True)

    op.create_table(
        "auth_user_roles",
        sa.Column("user_id", sa.Text, sa.ForeignKey("auth_users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.Text, primary_key=True),
    )
    op.create_index("ix_auth_user_roles_role", "auth_user_roles", ["role"])

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("token_hash", sa.Text, nullable=False),
        sa.Column("user_id", sa.Text, sa.ForeignKey("auth_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("csrf_token_hash", sa.Text, nullable=False),
        sa.Column("expires_at", sa.Text, nullable=False),
        sa.Column("revoked_at", sa.Text, nullable=True),
        sa.Column("created_at", sa.Text, nullable=False),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("ip_address", sa.Text, nullable=True),
    )
    op.create_index("ix_auth_sessions_token_hash", "auth_sessions", ["token_hash"], unique=True)
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])
    op.create_index("ix_auth_sessions_expires_at", "auth_sessions", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_auth_sessions_expires_at", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_token_hash", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_index("ix_auth_user_roles_role", table_name="auth_user_roles")
    op.drop_table("auth_user_roles")
    op.drop_index("ix_auth_users_email", table_name="auth_users")
    op.drop_index("ix_auth_users_username", table_name="auth_users")
    op.drop_table("auth_users")
