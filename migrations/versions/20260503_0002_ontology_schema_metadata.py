"""Add ontology schema metadata columns.

Revision ID: 20260503_0002
Revises: 20260503_0001
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260503_0002"
down_revision: str | None = "20260503_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = (
    "ontology_nodes",
    "ontology_edges",
    "ontology_snapshot_nodes",
    "ontology_snapshot_edges",
)


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS schema_name text NOT NULL DEFAULT 'legacy'")
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS schema_version integer NOT NULL DEFAULT 0")


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS schema_version")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS schema_name")
