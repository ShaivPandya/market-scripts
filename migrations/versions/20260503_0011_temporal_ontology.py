"""Add authoritative temporal ontology tables.

Revision ID: 20260503_0011
Revises: 20260503_0010
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260503_0011"
down_revision: str | None = "20260503_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB
TSTZ = sa.DateTime(timezone=True)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("Temporal ontology migration requires PostgreSQL.")

    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    op.create_table(
        "source_record_versions",
        sa.Column("source_record_id", UUID, primary_key=True),
        sa.Column("vendor", sa.Text, nullable=False),
        sa.Column("source_name", sa.Text, nullable=False),
        sa.Column("source_version", sa.Text, nullable=False),
        sa.Column("dataset", sa.Text, nullable=False),
        sa.Column("record_kind", sa.Text, nullable=False),
        sa.Column("record_key", sa.Text, nullable=False),
        sa.Column("record_key_hash", sa.Text, nullable=False),
        sa.Column("payload_hash", sa.Text, nullable=False),
        sa.Column("payload_json", JSONB),
        sa.Column("artifact_uri", sa.Text),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("quality", sa.Text, nullable=False),
        sa.Column("as_of", TSTZ),
        sa.Column("load_time", TSTZ, nullable=False, server_default=sa.text("clock_timestamp()")),
        sa.Column("valid_from", TSTZ, nullable=False),
        sa.Column("valid_to", TSTZ),
        sa.Column("tx_from", TSTZ, nullable=False, server_default=sa.text("clock_timestamp()")),
        sa.Column("tx_to", TSTZ),
        sa.Column("provenance_event_id", sa.Text),
        sa.CheckConstraint("valid_to IS NULL OR valid_to > valid_from", name="ck_source_record_valid_interval"),
        sa.CheckConstraint("tx_to IS NULL OR tx_to > tx_from", name="ck_source_record_tx_interval"),
    )
    op.create_index(
        "idx_source_record_versions_natural_current",
        "source_record_versions",
        ["vendor", "source_name", "dataset", "record_kind", "record_key_hash"],
        postgresql_where=sa.text("tx_to IS NULL"),
    )
    op.create_index("idx_source_record_versions_payload_hash", "source_record_versions", ["payload_hash"])
    op.create_index("idx_source_record_versions_as_of", "source_record_versions", ["as_of"])
    op.create_index("idx_source_record_versions_tx", "source_record_versions", ["tx_from", "tx_to"])

    op.create_table(
        "ontology_object_versions",
        sa.Column("version_id", UUID, primary_key=True),
        sa.Column("object_uid", sa.Text, nullable=False),
        sa.Column("object_type", sa.Text, nullable=False),
        sa.Column("business_key", sa.Text, nullable=False),
        sa.Column("schema_name", sa.Text, nullable=False),
        sa.Column("schema_version", sa.Integer, nullable=False),
        sa.Column("properties_json", JSONB, nullable=False),
        sa.Column("valid_from", TSTZ, nullable=False),
        sa.Column("valid_to", TSTZ),
        sa.Column("tx_from", TSTZ, nullable=False, server_default=sa.text("clock_timestamp()")),
        sa.Column("tx_to", TSTZ),
        sa.Column("source_record_id", UUID),
        sa.Column("provenance_event_id", sa.Text),
        sa.Column("action_run_id", sa.BigInteger),
        sa.Column("approval_id", sa.BigInteger),
        sa.Column("actor_type", sa.Text),
        sa.Column("actor_id", sa.Text),
        sa.Column("input_hash", sa.Text),
        sa.Column("supersedes_version_id", UUID),
        sa.Column("temporal_confidence", sa.Text, nullable=False, server_default="native"),
        sa.ForeignKeyConstraint(["source_record_id"], ["source_record_versions.source_record_id"]),
        sa.ForeignKeyConstraint(["supersedes_version_id"], ["ontology_object_versions.version_id"]),
        sa.CheckConstraint("valid_to IS NULL OR valid_to > valid_from", name="ck_object_version_valid_interval"),
        sa.CheckConstraint("tx_to IS NULL OR tx_to > tx_from", name="ck_object_version_tx_interval"),
    )
    op.create_index(
        "idx_ontology_object_versions_current_lookup",
        "ontology_object_versions",
        ["object_type", "business_key"],
        postgresql_where=sa.text("tx_to IS NULL"),
    )
    op.create_index(
        "idx_ontology_object_versions_uid_current",
        "ontology_object_versions",
        ["object_uid"],
        postgresql_where=sa.text("tx_to IS NULL"),
    )
    op.create_index(
        "idx_ontology_object_versions_valid", "ontology_object_versions", ["object_uid", "valid_from", "valid_to"]
    )
    op.create_index("idx_ontology_object_versions_tx", "ontology_object_versions", ["object_uid", "tx_from", "tx_to"])
    op.create_index("idx_ontology_object_versions_source", "ontology_object_versions", ["source_record_id"])

    op.create_table(
        "ontology_relation_versions",
        sa.Column("version_id", UUID, primary_key=True),
        sa.Column("relation_uid", sa.Text, nullable=False),
        sa.Column("source_object_uid", sa.Text, nullable=False),
        sa.Column("target_object_uid", sa.Text, nullable=False),
        sa.Column("relation_type", sa.Text, nullable=False),
        sa.Column("relation_schema_name", sa.Text, nullable=False),
        sa.Column("relation_schema_version", sa.Integer, nullable=False),
        sa.Column("properties_json", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("valid_from", TSTZ, nullable=False),
        sa.Column("valid_to", TSTZ),
        sa.Column("tx_from", TSTZ, nullable=False, server_default=sa.text("clock_timestamp()")),
        sa.Column("tx_to", TSTZ),
        sa.Column("source_record_id", UUID),
        sa.Column("provenance_event_id", sa.Text),
        sa.Column("action_run_id", sa.BigInteger),
        sa.Column("approval_id", sa.BigInteger),
        sa.Column("actor_type", sa.Text),
        sa.Column("actor_id", sa.Text),
        sa.Column("input_hash", sa.Text),
        sa.Column("supersedes_version_id", UUID),
        sa.Column("temporal_confidence", sa.Text, nullable=False, server_default="native"),
        sa.ForeignKeyConstraint(["source_record_id"], ["source_record_versions.source_record_id"]),
        sa.ForeignKeyConstraint(["supersedes_version_id"], ["ontology_relation_versions.version_id"]),
        sa.CheckConstraint("valid_to IS NULL OR valid_to > valid_from", name="ck_relation_version_valid_interval"),
        sa.CheckConstraint("tx_to IS NULL OR tx_to > tx_from", name="ck_relation_version_tx_interval"),
    )
    op.create_index(
        "idx_ontology_relation_versions_current_lookup",
        "ontology_relation_versions",
        ["relation_type", "source_object_uid", "target_object_uid"],
        postgresql_where=sa.text("tx_to IS NULL"),
    )
    op.create_index(
        "idx_ontology_relation_versions_uid_current",
        "ontology_relation_versions",
        ["relation_uid"],
        postgresql_where=sa.text("tx_to IS NULL"),
    )
    op.create_index("idx_ontology_relation_versions_source", "ontology_relation_versions", ["source_object_uid"])
    op.create_index("idx_ontology_relation_versions_target", "ontology_relation_versions", ["target_object_uid"])
    op.create_index(
        "idx_ontology_relation_versions_valid", "ontology_relation_versions", ["relation_uid", "valid_from", "valid_to"]
    )
    op.create_index(
        "idx_ontology_relation_versions_tx", "ontology_relation_versions", ["relation_uid", "tx_from", "tx_to"]
    )

    op.create_table(
        "computed_snapshot_versions",
        sa.Column("snapshot_id", UUID, primary_key=True),
        sa.Column("snapshot_key", sa.Text, nullable=False),
        sa.Column("payload_hash", sa.Text, nullable=False),
        sa.Column("payload_json", JSONB),
        sa.Column("artifact_uri", sa.Text),
        sa.Column("as_of", TSTZ),
        sa.Column("load_time", TSTZ, nullable=False, server_default=sa.text("clock_timestamp()")),
        sa.Column("valid_from", TSTZ, nullable=False),
        sa.Column("valid_to", TSTZ),
        sa.Column("tx_from", TSTZ, nullable=False, server_default=sa.text("clock_timestamp()")),
        sa.Column("tx_to", TSTZ),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("quality", sa.Text, nullable=False, server_default="ok"),
        sa.Column("error", sa.Text),
        sa.Column(
            "source_record_ids",
            postgresql.ARRAY(UUID),
            nullable=False,
            server_default=sa.text("ARRAY[]::uuid[]"),
        ),
        sa.Column("provenance_event_id", sa.Text),
        sa.CheckConstraint("valid_to IS NULL OR valid_to > valid_from", name="ck_snapshot_version_valid_interval"),
        sa.CheckConstraint("tx_to IS NULL OR tx_to > tx_from", name="ck_snapshot_version_tx_interval"),
    )
    op.create_index(
        "idx_computed_snapshot_versions_current",
        "computed_snapshot_versions",
        ["snapshot_key"],
        postgresql_where=sa.text("tx_to IS NULL"),
    )
    op.create_index("idx_computed_snapshot_versions_as_of", "computed_snapshot_versions", ["snapshot_key", "as_of"])
    op.create_index("idx_computed_snapshot_versions_payload_hash", "computed_snapshot_versions", ["payload_hash"])
    op.create_index(
        "idx_computed_snapshot_versions_tx", "computed_snapshot_versions", ["snapshot_key", "tx_from", "tx_to"]
    )

    op.execute(
        """
        ALTER TABLE ontology_object_versions
        ADD CONSTRAINT excl_ontology_object_current_valid_overlap
        EXCLUDE USING gist (
            object_uid WITH =,
            tstzrange(valid_from, COALESCE(valid_to, 'infinity'::timestamptz), '[)') WITH &&
        )
        WHERE (tx_to IS NULL)
        """
    )
    op.execute(
        """
        ALTER TABLE ontology_relation_versions
        ADD CONSTRAINT excl_ontology_relation_current_valid_overlap
        EXCLUDE USING gist (
            relation_uid WITH =,
            tstzrange(valid_from, COALESCE(valid_to, 'infinity'::timestamptz), '[)') WITH &&
        )
        WHERE (tx_to IS NULL)
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("Temporal ontology migration requires PostgreSQL.")

    op.execute(
        "ALTER TABLE ontology_relation_versions DROP CONSTRAINT IF EXISTS excl_ontology_relation_current_valid_overlap"
    )
    op.execute(
        "ALTER TABLE ontology_object_versions DROP CONSTRAINT IF EXISTS excl_ontology_object_current_valid_overlap"
    )

    op.drop_index("idx_computed_snapshot_versions_tx", table_name="computed_snapshot_versions")
    op.drop_index("idx_computed_snapshot_versions_payload_hash", table_name="computed_snapshot_versions")
    op.drop_index("idx_computed_snapshot_versions_as_of", table_name="computed_snapshot_versions")
    op.drop_index("idx_computed_snapshot_versions_current", table_name="computed_snapshot_versions")
    op.drop_table("computed_snapshot_versions")

    op.drop_index("idx_ontology_relation_versions_tx", table_name="ontology_relation_versions")
    op.drop_index("idx_ontology_relation_versions_valid", table_name="ontology_relation_versions")
    op.drop_index("idx_ontology_relation_versions_target", table_name="ontology_relation_versions")
    op.drop_index("idx_ontology_relation_versions_source", table_name="ontology_relation_versions")
    op.drop_index("idx_ontology_relation_versions_uid_current", table_name="ontology_relation_versions")
    op.drop_index("idx_ontology_relation_versions_current_lookup", table_name="ontology_relation_versions")
    op.drop_table("ontology_relation_versions")

    op.drop_index("idx_ontology_object_versions_source", table_name="ontology_object_versions")
    op.drop_index("idx_ontology_object_versions_tx", table_name="ontology_object_versions")
    op.drop_index("idx_ontology_object_versions_valid", table_name="ontology_object_versions")
    op.drop_index("idx_ontology_object_versions_uid_current", table_name="ontology_object_versions")
    op.drop_index("idx_ontology_object_versions_current_lookup", table_name="ontology_object_versions")
    op.drop_table("ontology_object_versions")

    op.drop_index("idx_source_record_versions_tx", table_name="source_record_versions")
    op.drop_index("idx_source_record_versions_as_of", table_name="source_record_versions")
    op.drop_index("idx_source_record_versions_payload_hash", table_name="source_record_versions")
    op.drop_index("idx_source_record_versions_natural_current", table_name="source_record_versions")
    op.drop_table("source_record_versions")
