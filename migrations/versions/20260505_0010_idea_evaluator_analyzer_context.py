"""Add analyzer context to idea evaluations.

Revision ID: 20260505_0010
Revises: 20260505_0009
Create Date: 2026-05-07
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260505_0010"
down_revision: str | None = "20260505_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE idea_evaluations
            ADD COLUMN IF NOT EXISTS analyzer_context_json TEXT NOT NULL DEFAULT '{}',
            ADD COLUMN IF NOT EXISTS evaluation_schema_version TEXT;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE idea_evaluations
            DROP COLUMN IF EXISTS evaluation_schema_version,
            DROP COLUMN IF EXISTS analyzer_context_json;
        """
    )
