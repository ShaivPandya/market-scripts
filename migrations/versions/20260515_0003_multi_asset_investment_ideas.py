"""Add multi-asset instrument metadata to investment ideas.

Revision ID: 20260515_0003
Revises: 20260515_0002
Create Date: 2026-05-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260515_0003"
down_revision: str | None = "20260515_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            ALTER TABLE investment_ideas
                ADD COLUMN IF NOT EXISTS asset text NOT NULL DEFAULT 'equity',
                ADD COLUMN IF NOT EXISTS instrument_type text NOT NULL DEFAULT 'security',
                ADD COLUMN IF NOT EXISTS price_symbol text,
                ADD COLUMN IF NOT EXISTS contract_multiplier double precision NOT NULL DEFAULT 1.0,
                ADD COLUMN IF NOT EXISTS fx_base_currency text,
                ADD COLUMN IF NOT EXISTS fx_quote_currency text,
                ADD COLUMN IF NOT EXISTS currency text,
                ADD COLUMN IF NOT EXISTS country text,
                ADD COLUMN IF NOT EXISTS exchange text;
            UPDATE investment_ideas
            SET asset = COALESCE(NULLIF(asset, ''), 'equity'),
                instrument_type = COALESCE(NULLIF(instrument_type, ''), 'security'),
                price_symbol = COALESCE(NULLIF(price_symbol, ''), ticker),
                contract_multiplier = COALESCE(contract_multiplier, 1.0);
            """
        )
        return

    with op.batch_alter_table("investment_ideas") as batch:
        batch.add_column(sa.Column("asset", sa.Text, nullable=False, server_default="equity"))
        batch.add_column(sa.Column("instrument_type", sa.Text, nullable=False, server_default="security"))
        batch.add_column(sa.Column("price_symbol", sa.Text))
        batch.add_column(sa.Column("contract_multiplier", sa.Float, nullable=False, server_default=sa.text("1.0")))
        batch.add_column(sa.Column("fx_base_currency", sa.Text))
        batch.add_column(sa.Column("fx_quote_currency", sa.Text))
        batch.add_column(sa.Column("currency", sa.Text))
        batch.add_column(sa.Column("country", sa.Text))
        batch.add_column(sa.Column("exchange", sa.Text))
    op.execute(
        """
        UPDATE investment_ideas
        SET asset = COALESCE(NULLIF(asset, ''), 'equity'),
            instrument_type = COALESCE(NULLIF(instrument_type, ''), 'security'),
            price_symbol = COALESCE(NULLIF(price_symbol, ''), ticker),
            contract_multiplier = COALESCE(contract_multiplier, 1.0)
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    columns = [
        "exchange",
        "country",
        "currency",
        "fx_quote_currency",
        "fx_base_currency",
        "contract_multiplier",
        "price_symbol",
        "instrument_type",
        "asset",
    ]
    if bind.dialect.name == "postgresql":
        for column in columns:
            op.execute(f"ALTER TABLE investment_ideas DROP COLUMN IF EXISTS {column}")
        return

    with op.batch_alter_table("investment_ideas") as batch:
        for column in columns:
            batch.drop_column(column)
