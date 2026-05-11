"""Add grouped conviction fields to portfolio positions.

Revision ID: 20260511_0001
Revises: 20260509_0001
Create Date: 2026-05-11
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260511_0001"
down_revision: str | None = "20260509_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE positions ADD COLUMN IF NOT EXISTS group_name text")
        op.execute("ALTER TABLE positions ADD COLUMN IF NOT EXISTS group_conviction integer")
        return

    with op.batch_alter_table("positions") as batch:
        batch.add_column(sa.Column("group_name", sa.Text))
        batch.add_column(sa.Column("group_conviction", sa.Integer))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE positions DROP COLUMN IF EXISTS group_conviction")
        op.execute("ALTER TABLE positions DROP COLUMN IF EXISTS group_name")
        return

    with op.batch_alter_table("positions") as batch:
        batch.drop_column("group_conviction")
        batch.drop_column("group_name")
