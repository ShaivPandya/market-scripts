"""Constrain ontology relation registry edges.

Revision ID: 20260503_0003
Revises: 20260503_0002
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260503_0003"
down_revision: str | None = "20260503_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RELATION_VALUES = (
    "'references_asset', 'belongs_to_sector', 'has_thesis', 'evaluated_by', "
    "'has_catalyst', 'emits_signal', 'affected_by', 'exposed_to_signal'"
)
_RELATION_CHECK = f"relation_type IN ({_RELATION_VALUES})"
_SOURCE_UNIQUE_RELATIONS = "relation_type IN ('references_asset', 'belongs_to_sector')"
_TARGET_UNIQUE_RELATIONS = "relation_type IN ('emits_signal', 'evaluated_by', 'has_catalyst')"
_HAS_THESIS = "relation_type = 'has_thesis'"


def upgrade() -> None:
    op.create_check_constraint("ck_ontology_edges_relation_type", "ontology_edges", _RELATION_CHECK)
    op.create_check_constraint(
        "ck_ontology_snapshot_edges_relation_type",
        "ontology_snapshot_edges",
        _RELATION_CHECK,
    )

    op.create_foreign_key(
        "fk_ontology_edges_source_node",
        "ontology_edges",
        "ontology_nodes",
        ["source_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_ontology_edges_target_node",
        "ontology_edges",
        "ontology_nodes",
        ["target_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_ontology_snapshot_edges_source_node",
        "ontology_snapshot_edges",
        "ontology_snapshot_nodes",
        ["run_id", "source_id"],
        ["run_id", "id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_ontology_snapshot_edges_target_node",
        "ontology_snapshot_edges",
        "ontology_snapshot_nodes",
        ["run_id", "target_id"],
        ["run_id", "id"],
        ondelete="CASCADE",
    )

    op.create_index(
        "uq_ontology_edges_source_relation",
        "ontology_edges",
        ["source_id", "relation_type"],
        unique=True,
        postgresql_where=sa.text(_SOURCE_UNIQUE_RELATIONS),
    )
    op.create_index(
        "uq_ontology_edges_target_relation",
        "ontology_edges",
        ["target_id", "relation_type"],
        unique=True,
        postgresql_where=sa.text(_TARGET_UNIQUE_RELATIONS),
    )
    op.create_index(
        "uq_ontology_edges_has_thesis_source",
        "ontology_edges",
        ["source_id", "relation_type"],
        unique=True,
        postgresql_where=sa.text(_HAS_THESIS),
    )
    op.create_index(
        "uq_ontology_edges_has_thesis_target",
        "ontology_edges",
        ["target_id", "relation_type"],
        unique=True,
        postgresql_where=sa.text(_HAS_THESIS),
    )
    op.create_index(
        "uq_ontology_snapshot_edges_source_relation",
        "ontology_snapshot_edges",
        ["run_id", "source_id", "relation_type"],
        unique=True,
        postgresql_where=sa.text(_SOURCE_UNIQUE_RELATIONS),
    )
    op.create_index(
        "uq_ontology_snapshot_edges_target_relation",
        "ontology_snapshot_edges",
        ["run_id", "target_id", "relation_type"],
        unique=True,
        postgresql_where=sa.text(_TARGET_UNIQUE_RELATIONS),
    )
    op.create_index(
        "uq_ontology_snapshot_edges_has_thesis_source",
        "ontology_snapshot_edges",
        ["run_id", "source_id", "relation_type"],
        unique=True,
        postgresql_where=sa.text(_HAS_THESIS),
    )
    op.create_index(
        "uq_ontology_snapshot_edges_has_thesis_target",
        "ontology_snapshot_edges",
        ["run_id", "target_id", "relation_type"],
        unique=True,
        postgresql_where=sa.text(_HAS_THESIS),
    )


def downgrade() -> None:
    for index_name, table_name in [
        ("uq_ontology_snapshot_edges_has_thesis_target", "ontology_snapshot_edges"),
        ("uq_ontology_snapshot_edges_has_thesis_source", "ontology_snapshot_edges"),
        ("uq_ontology_snapshot_edges_target_relation", "ontology_snapshot_edges"),
        ("uq_ontology_snapshot_edges_source_relation", "ontology_snapshot_edges"),
        ("uq_ontology_edges_has_thesis_target", "ontology_edges"),
        ("uq_ontology_edges_has_thesis_source", "ontology_edges"),
        ("uq_ontology_edges_target_relation", "ontology_edges"),
        ("uq_ontology_edges_source_relation", "ontology_edges"),
    ]:
        op.drop_index(index_name, table_name=table_name)

    for constraint_name, table_name in [
        ("fk_ontology_snapshot_edges_target_node", "ontology_snapshot_edges"),
        ("fk_ontology_snapshot_edges_source_node", "ontology_snapshot_edges"),
        ("fk_ontology_edges_target_node", "ontology_edges"),
        ("fk_ontology_edges_source_node", "ontology_edges"),
    ]:
        op.drop_constraint(constraint_name, table_name, type_="foreignkey")

    op.drop_constraint("ck_ontology_snapshot_edges_relation_type", "ontology_snapshot_edges", type_="check")
    op.drop_constraint("ck_ontology_edges_relation_type", "ontology_edges", type_="check")
