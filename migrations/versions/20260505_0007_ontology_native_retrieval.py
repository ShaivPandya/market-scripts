"""Add ontology-native retrieval metadata.

Revision ID: 20260505_0007
Revises: 20260505_0006
Create Date: 2026-05-05
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260505_0007"
down_revision: str | None = "20260505_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("Ontology-native retrieval metadata requires PostgreSQL.")

    for name, column in (
        ("object_uid", sa.Column("object_uid", sa.Text(), nullable=True)),
        ("object_version_id", sa.Column("object_version_id", sa.Text(), nullable=True)),
        ("source_record_id", sa.Column("source_record_id", sa.Text(), nullable=True)),
        ("source_record_version_id", sa.Column("source_record_version_id", sa.Text(), nullable=True)),
        ("citation_span_start", sa.Column("citation_span_start", sa.Integer(), nullable=True)),
        ("citation_span_end", sa.Column("citation_span_end", sa.Integer(), nullable=True)),
        ("permission_scope", sa.Column("permission_scope", sa.Text(), nullable=False, server_default="owner")),
        ("freshness_as_of", sa.Column("freshness_as_of", sa.DateTime(timezone=True), nullable=True)),
        ("stale_after", sa.Column("stale_after", sa.DateTime(timezone=True), nullable=True)),
        ("is_stale", sa.Column("is_stale", sa.Boolean(), nullable=False, server_default=sa.text("false"))),
        ("content_hash", sa.Column("content_hash", sa.Text(), nullable=True)),
    ):
        if not _has_column("retrieval_chunks", name):
            op.add_column("retrieval_chunks", column)

    op.create_index(
        "idx_retrieval_chunks_object_uid",
        "retrieval_chunks",
        ["object_uid"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_retrieval_chunks_source_record",
        "retrieval_chunks",
        ["source_record_id"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_retrieval_chunks_permission_freshness",
        "retrieval_chunks",
        ["permission_scope", "is_stale"],
        if_not_exists=True,
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("Ontology-native retrieval metadata requires PostgreSQL.")

    for index_name in (
        "idx_retrieval_chunks_permission_freshness",
        "idx_retrieval_chunks_source_record",
        "idx_retrieval_chunks_object_uid",
    ):
        op.drop_index(index_name, table_name="retrieval_chunks", if_exists=True)
    for name in (
        "content_hash",
        "is_stale",
        "stale_after",
        "freshness_as_of",
        "permission_scope",
        "citation_span_end",
        "citation_span_start",
        "source_record_version_id",
        "source_record_id",
        "object_version_id",
        "object_uid",
    ):
        if _has_column("retrieval_chunks", name):
            op.drop_column("retrieval_chunks", name)


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))
