"""Add lineage completeness controls and reporting view.

Revision ID: 20260503_0013
Revises: 20260503_0012
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260503_0013"
down_revision: str | None = "20260503_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LINEAGE_STATES = "'complete', 'retry_pending', 'dead_letter', 'legacy_partial', 'failed_closed'"
_TABLES = (
    "recommendations",
    "policy_gate_results",
    "pending_approvals",
    "action_runs",
    "workflow_runs",
)


def _add_check_constraint(table: str) -> None:
    constraint = f"ck_{table}_lineage_completeness"
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = '{constraint}'
                  AND conrelid = '{table}'::regclass
            ) THEN
                ALTER TABLE {table}
                ADD CONSTRAINT {constraint}
                CHECK (lineage_completeness IN ({_LINEAGE_STATES})) NOT VALID;
            END IF;
        END
        $$;
        """
    )
    op.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {constraint}")


def upgrade() -> None:
    for table in ("pending_approvals", "action_runs", "workflow_runs"):
        op.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS lineage_completeness text NOT NULL DEFAULT 'legacy_partial'"
        )
        op.execute(
            f"UPDATE {table} SET lineage_completeness = 'legacy_partial' "
            "WHERE lineage_completeness IS NULL OR lineage_completeness = ''"
        )

    for table in _TABLES:
        _add_check_constraint(table)

    for stmt in [
        "CREATE INDEX IF NOT EXISTS idx_pending_approvals_lineage_completeness "
        "ON pending_approvals(lineage_completeness)",
        "CREATE INDEX IF NOT EXISTS idx_action_runs_lineage_completeness ON action_runs(lineage_completeness)",
        "CREATE INDEX IF NOT EXISTS idx_workflow_runs_lineage_completeness ON workflow_runs(lineage_completeness)",
        "CREATE INDEX IF NOT EXISTS idx_policy_gate_results_lineage_completeness "
        "ON policy_gate_results(lineage_completeness)",
        "CREATE INDEX IF NOT EXISTS idx_governance_outbox_status_lineage ON governance_outbox(status, lineage_root_id)",
    ]:
        op.execute(stmt)

    op.execute(
        """
        CREATE OR REPLACE VIEW governance_lineage_completeness AS
        SELECT
            'recommendation'::text AS ref_type,
            id::text AS ref_id,
            COALESCE(lineage_root_id, 'recommendation:' || id::text) AS lineage_root_id,
            lineage_completeness,
            status::text AS status,
            created_at AS observed_at
        FROM recommendations
        UNION ALL
        SELECT
            'policy_gate_result'::text AS ref_type,
            id::text AS ref_id,
            COALESCE(lineage_root_id, 'policy_gate_result:' || id::text) AS lineage_root_id,
            lineage_completeness,
            decision::text AS status,
            created_at AS observed_at
        FROM policy_gate_results
        UNION ALL
        SELECT
            'approval'::text AS ref_type,
            id::text AS ref_id,
            'approval:' || id::text AS lineage_root_id,
            lineage_completeness,
            status::text AS status,
            COALESCE(resolved_at, created_at) AS observed_at
        FROM pending_approvals
        UNION ALL
        SELECT
            'action_run'::text AS ref_type,
            id::text AS ref_id,
            'action_run:' || id::text AS lineage_root_id,
            lineage_completeness,
            status::text AS status,
            COALESCE(completed_at, started_at) AS observed_at
        FROM action_runs
        UNION ALL
        SELECT
            'workflow_run'::text AS ref_type,
            run_id::text AS ref_id,
            'workflow_run:' || run_id::text AS lineage_root_id,
            lineage_completeness,
            status::text AS status,
            COALESCE(completed_at, started_at) AS observed_at
        FROM workflow_runs
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS governance_lineage_completeness")
    for stmt in [
        "DROP INDEX IF EXISTS idx_governance_outbox_status_lineage",
        "DROP INDEX IF EXISTS idx_policy_gate_results_lineage_completeness",
        "DROP INDEX IF EXISTS idx_workflow_runs_lineage_completeness",
        "DROP INDEX IF EXISTS idx_action_runs_lineage_completeness",
        "DROP INDEX IF EXISTS idx_pending_approvals_lineage_completeness",
    ]:
        op.execute(stmt)
    for table in reversed(_TABLES):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS ck_{table}_lineage_completeness")
    for table in ("workflow_runs", "action_runs", "pending_approvals"):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS lineage_completeness")
