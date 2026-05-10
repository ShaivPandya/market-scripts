"""Add resolved state to optimization alerts.

Revision ID: 20260509_0001
Revises: 20260507_0001
Create Date: 2026-05-09
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260509_0001"
down_revision: str | None = "20260507_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SQLITE_OPTIMIZATION_ALERTS_WITH_RESOLVED = """
CREATE TABLE optimization_alerts (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id               INTEGER NOT NULL,
    run_id                   TEXT NOT NULL,
    ticker                   TEXT,
    alert_type               TEXT NOT NULL,
    severity                 TEXT NOT NULL
                             CHECK (severity IN ('low', 'normal', 'high', 'urgent')),
    status                   TEXT NOT NULL DEFAULT 'open'
                             CHECK (status IN ('open', 'dismissed', 'superseded', 'resolved')),
    previous_snapshot_id     INTEGER,
    current_snapshot_id      INTEGER,
    change_summary           TEXT NOT NULL,
    evidence_json            TEXT,
    approval_id              INTEGER,
    recommendation_id        INTEGER,
    action_item_approval_id  INTEGER,
    created_at               TEXT NOT NULL,
    dismissed_at             TEXT,
    dismissed_note           TEXT,
    resolved_at              TEXT,
    resolved_reason          TEXT
)
"""

SQLITE_OPTIMIZATION_ALERTS_LEGACY = """
CREATE TABLE optimization_alerts (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id               INTEGER NOT NULL,
    run_id                   TEXT NOT NULL,
    ticker                   TEXT,
    alert_type               TEXT NOT NULL,
    severity                 TEXT NOT NULL
                             CHECK (severity IN ('low', 'normal', 'high', 'urgent')),
    status                   TEXT NOT NULL DEFAULT 'open'
                             CHECK (status IN ('open', 'dismissed', 'superseded')),
    previous_snapshot_id     INTEGER,
    current_snapshot_id      INTEGER,
    change_summary           TEXT NOT NULL,
    evidence_json            TEXT,
    approval_id              INTEGER,
    recommendation_id        INTEGER,
    action_item_approval_id  INTEGER,
    created_at               TEXT NOT NULL,
    dismissed_at             TEXT,
    dismissed_note           TEXT
)
"""


def _sqlite_columns(table: str) -> set[str]:
    bind = op.get_bind()
    return {str(row[1]) for row in bind.execute(sa.text(f"PRAGMA table_info({table})")).fetchall()}


def _recreate_sqlite_optimization_alerts(*, include_resolved: bool) -> None:
    legacy_table = "optimization_alerts_status_upgrade_old"
    op.execute(f"DROP TABLE IF EXISTS {legacy_table}")
    if not include_resolved:
        op.execute("UPDATE optimization_alerts SET status = 'superseded' WHERE status = 'resolved'")
    op.execute(f"ALTER TABLE optimization_alerts RENAME TO {legacy_table}")
    op.execute(SQLITE_OPTIMIZATION_ALERTS_WITH_RESOLVED if include_resolved else SQLITE_OPTIMIZATION_ALERTS_LEGACY)

    legacy_cols = _sqlite_columns(legacy_table)
    target_cols = _sqlite_columns("optimization_alerts")
    copy_cols = [col for col in target_cols if col in legacy_cols]
    cols_sql = ", ".join(copy_cols)
    if cols_sql:
        op.execute(f"INSERT INTO optimization_alerts ({cols_sql}) SELECT {cols_sql} FROM {legacy_table}")
    op.execute(f"DROP TABLE {legacy_table}")


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE optimization_alerts ADD COLUMN IF NOT EXISTS resolved_at TEXT")
        op.execute("ALTER TABLE optimization_alerts ADD COLUMN IF NOT EXISTS resolved_reason TEXT")
        op.execute("ALTER TABLE optimization_alerts DROP CONSTRAINT IF EXISTS optimization_alerts_status_check")
        op.execute(
            """
            ALTER TABLE optimization_alerts
            ADD CONSTRAINT optimization_alerts_status_check
            CHECK (status IN ('open', 'dismissed', 'superseded', 'resolved'))
            """
        )
        return

    _recreate_sqlite_optimization_alerts(include_resolved=True)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("UPDATE optimization_alerts SET status = 'superseded' WHERE status = 'resolved'")
        op.execute("ALTER TABLE optimization_alerts DROP CONSTRAINT IF EXISTS optimization_alerts_status_check")
        op.execute(
            """
            ALTER TABLE optimization_alerts
            ADD CONSTRAINT optimization_alerts_status_check
            CHECK (status IN ('open', 'dismissed', 'superseded'))
            """
        )
        op.execute("ALTER TABLE optimization_alerts DROP COLUMN IF EXISTS resolved_reason")
        op.execute("ALTER TABLE optimization_alerts DROP COLUMN IF EXISTS resolved_at")
        return

    _recreate_sqlite_optimization_alerts(include_resolved=False)
