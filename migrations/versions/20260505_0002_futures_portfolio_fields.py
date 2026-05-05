"""Add futures-aware portfolio position fields.

Revision ID: 20260505_0002
Revises: 20260505_0001
Create Date: 2026-05-05
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260505_0002"
down_revision: str | None = "20260505_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE positions ADD COLUMN IF NOT EXISTS quantity double precision")
        op.execute("ALTER TABLE positions ADD COLUMN IF NOT EXISTS instrument_type text NOT NULL DEFAULT 'security'")
        op.execute("ALTER TABLE positions ADD COLUMN IF NOT EXISTS price_symbol text")
        op.execute(
            "ALTER TABLE positions ADD COLUMN IF NOT EXISTS contract_multiplier double precision NOT NULL DEFAULT 1.0"
        )
        op.execute("UPDATE positions SET quantity = shares WHERE quantity IS NULL AND shares IS NOT NULL")
        op.execute(
            "UPDATE positions SET instrument_type = 'security' WHERE instrument_type IS NULL OR instrument_type = ''"
        )
        op.execute("UPDATE positions SET price_symbol = ticker WHERE price_symbol IS NULL OR price_symbol = ''")
        op.execute("UPDATE positions SET contract_multiplier = 1.0 WHERE contract_multiplier IS NULL")
        return

    with op.batch_alter_table("positions") as batch:
        batch.add_column(sa.Column("quantity", sa.Float))
        batch.add_column(sa.Column("instrument_type", sa.Text, nullable=False, server_default="security"))
        batch.add_column(sa.Column("price_symbol", sa.Text))
        batch.add_column(sa.Column("contract_multiplier", sa.Float, nullable=False, server_default=sa.text("1.0")))
    op.execute("UPDATE positions SET quantity = shares WHERE quantity IS NULL AND shares IS NOT NULL")
    op.execute(
        "UPDATE positions SET instrument_type = 'security' WHERE instrument_type IS NULL OR instrument_type = ''"
    )
    op.execute("UPDATE positions SET price_symbol = ticker WHERE price_symbol IS NULL OR price_symbol = ''")
    op.execute("UPDATE positions SET contract_multiplier = 1.0 WHERE contract_multiplier IS NULL")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE positions DROP COLUMN IF EXISTS contract_multiplier")
        op.execute("ALTER TABLE positions DROP COLUMN IF EXISTS price_symbol")
        op.execute("ALTER TABLE positions DROP COLUMN IF EXISTS instrument_type")
        op.execute("ALTER TABLE positions DROP COLUMN IF EXISTS quantity")
        return

    with op.batch_alter_table("positions") as batch:
        batch.drop_column("contract_multiplier")
        batch.drop_column("price_symbol")
        batch.drop_column("instrument_type")
        batch.drop_column("quantity")
