"""Add spot FX portfolio position fields.

Revision ID: 20260507_0001
Revises: 20260505_0010
Create Date: 2026-05-07
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260507_0001"
down_revision: str | None = "20260505_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE positions ADD COLUMN IF NOT EXISTS fx_base_currency text")
        op.execute("ALTER TABLE positions ADD COLUMN IF NOT EXISTS fx_quote_currency text")
        return

    with op.batch_alter_table("positions") as batch:
        batch.add_column(sa.Column("fx_base_currency", sa.Text))
        batch.add_column(sa.Column("fx_quote_currency", sa.Text))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE positions DROP COLUMN IF EXISTS fx_quote_currency")
        op.execute("ALTER TABLE positions DROP COLUMN IF EXISTS fx_base_currency")
        return

    with op.batch_alter_table("positions") as batch:
        batch.drop_column("fx_quote_currency")
        batch.drop_column("fx_base_currency")
