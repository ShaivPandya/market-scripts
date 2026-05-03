"""Allow news-aware watch trigger types.

Revision ID: 20260503_0001
Revises: 20260502_0004
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260503_0001"
down_revision: str | None = "20260502_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT = "ck_watch_triggers_trigger_type"
_TRIGGER_TYPES = (
    "price_level",
    "technical",
    "fundamental",
    "fundamental_news",
    "event",
    "news_event",
    "macro",
    "custom",
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # Local SQLite uses portfolio.core_db's runtime table rebuild because
        # SQLite cannot add CHECK constraints in place.
        return
    allowed = ", ".join(f"'{value}'" for value in _TRIGGER_TYPES)
    op.create_check_constraint(_CONSTRAINT, "watch_triggers", f"trigger_type IN ({allowed})")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.drop_constraint(_CONSTRAINT, "watch_triggers", type_="check")
