"""Add provenance retention metadata.

Revision ID: 20260503_0010
Revises: 20260503_0009
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260503_0010"
down_revision: str | None = "20260503_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE source_record_refs ADD COLUMN IF NOT EXISTS redaction_policy text NOT NULL DEFAULT 'audit_summary_v1'"
    )
    op.execute(
        "ALTER TABLE source_record_refs ADD COLUMN IF NOT EXISTS retention_class text NOT NULL DEFAULT 'source_ref_90d'"
    )
    op.execute(
        "ALTER TABLE workflow_artifact_records ADD COLUMN IF NOT EXISTS redaction_policy text NOT NULL DEFAULT 'audit_summary_v1'"
    )
    op.execute(
        "ALTER TABLE workflow_artifact_records ADD COLUMN IF NOT EXISTS retention_class text NOT NULL DEFAULT 'workflow_artifact_365d'"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_provenance_links_type_time ON provenance_links(link_type, created_at)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_provenance_links_type_time")
    op.execute("ALTER TABLE workflow_artifact_records DROP COLUMN IF EXISTS retention_class")
    op.execute("ALTER TABLE workflow_artifact_records DROP COLUMN IF EXISTS redaction_policy")
    op.execute("ALTER TABLE source_record_refs DROP COLUMN IF EXISTS retention_class")
    op.execute("ALTER TABLE source_record_refs DROP COLUMN IF EXISTS redaction_policy")
