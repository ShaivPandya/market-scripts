"""Add ontology query indexes for paginated snapshot traversal.

Revision ID: 20260503_0007
Revises: 20260503_0006
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260503_0007"
down_revision: str | None = "20260503_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "idx_ontology_snapshot_nodes_run_type_id",
        "ontology_snapshot_nodes",
        ["run_id", "type", "id"],
    )
    op.create_index(
        "idx_ontology_snapshot_edges_run_relation_source_target",
        "ontology_snapshot_edges",
        ["run_id", "relation_type", "source_id", "target_id"],
    )
    op.execute(
        """
        CREATE INDEX idx_ontology_snapshot_nodes_position_asset_lookup
        ON ontology_snapshot_nodes (
            run_id,
            lower((properties_json::jsonb ->> 'asset')),
            id
        )
        WHERE type = 'Position'
        """
    )
    op.execute(
        """
        CREATE INDEX idx_ontology_snapshot_nodes_position_risk_sort
        ON ontology_snapshot_nodes (
            run_id,
            ((NULLIF((properties_json::jsonb ->> 'risk_score'), ''))::double precision) DESC,
            id
        )
        WHERE type = 'Position'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_ontology_snapshot_nodes_position_risk_sort")
    op.execute("DROP INDEX IF EXISTS idx_ontology_snapshot_nodes_position_asset_lookup")
    op.drop_index("idx_ontology_snapshot_edges_run_relation_source_target", table_name="ontology_snapshot_edges")
    op.drop_index("idx_ontology_snapshot_nodes_run_type_id", table_name="ontology_snapshot_nodes")
