"""Add base-currency valuation fields to portfolio positions.

Revision ID: 20260505_0003
Revises: 20260505_0002
Create Date: 2026-05-05
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260505_0003"
down_revision: str | None = "20260505_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

VALUATION_COLUMNS: tuple[tuple[str, sa.TypeEngine, str | None], ...] = (
    ("currency", sa.Text(), None),
    ("country", sa.Text(), None),
    ("exchange", sa.Text(), None),
    ("base_currency", sa.Text(), "'USD'"),
    ("fx_rate_to_base", sa.Float(), None),
    ("fx_rate_as_of", sa.Text(), None),
    ("cost_basis_base", sa.Float(), None),
    ("notional_base", sa.Float(), None),
    ("valuation_status", sa.Text(), "'missing_position_inputs'"),
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for name, column_type, default in VALUATION_COLUMNS:
            default_sql = f" DEFAULT {default}" if default is not None else ""
            not_null = " NOT NULL" if name in {"base_currency", "valuation_status"} else ""
            op.execute(f"ALTER TABLE positions ADD COLUMN IF NOT EXISTS {name} {column_type}{default_sql}{not_null}")
        op.execute("UPDATE positions SET base_currency = 'USD' WHERE base_currency IS NULL OR base_currency = ''")
        op.execute(
            "UPDATE positions SET valuation_status = 'missing_position_inputs' "
            "WHERE valuation_status IS NULL OR valuation_status = ''"
        )
        return

    with op.batch_alter_table("positions") as batch:
        batch.add_column(sa.Column("currency", sa.Text))
        batch.add_column(sa.Column("country", sa.Text))
        batch.add_column(sa.Column("exchange", sa.Text))
        batch.add_column(sa.Column("base_currency", sa.Text, nullable=False, server_default="USD"))
        batch.add_column(sa.Column("fx_rate_to_base", sa.Float))
        batch.add_column(sa.Column("fx_rate_as_of", sa.Text))
        batch.add_column(sa.Column("cost_basis_base", sa.Float))
        batch.add_column(sa.Column("notional_base", sa.Float))
        batch.add_column(
            sa.Column("valuation_status", sa.Text, nullable=False, server_default="missing_position_inputs")
        )
    op.execute("UPDATE positions SET base_currency = 'USD' WHERE base_currency IS NULL OR base_currency = ''")
    op.execute(
        "UPDATE positions SET valuation_status = 'missing_position_inputs' "
        "WHERE valuation_status IS NULL OR valuation_status = ''"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for name, _, _ in reversed(VALUATION_COLUMNS):
            op.execute(f"ALTER TABLE positions DROP COLUMN IF EXISTS {name}")
        return

    with op.batch_alter_table("positions") as batch:
        for name, _, _ in reversed(VALUATION_COLUMNS):
            batch.drop_column(name)
