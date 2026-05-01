"""Add sp500_top50_tickers table for cached top-50 S&P 500 leadership list.

Revision ID: 20260430_0001
Revises: 20260429_0002
Create Date: 2026-04-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260430_0001"
down_revision: str | None = "20260429_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sp500_top50_tickers",
        sa.Column("ticker", sa.Text, primary_key=True),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("one_year_return_pct", sa.Float, nullable=False),
        sa.Column("refreshed_at", sa.Text, nullable=False),
    )
    op.create_index(
        "idx_sp500_top50_tickers_rank",
        "sp500_top50_tickers",
        ["rank"],
    )


def downgrade() -> None:
    op.drop_index("idx_sp500_top50_tickers_rank", table_name="sp500_top50_tickers")
    op.drop_table("sp500_top50_tickers")
