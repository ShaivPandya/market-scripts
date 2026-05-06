"""Bound async completed-result reuse by freshness.

Revision ID: 20260505_0008
Revises: 20260505_0007
Create Date: 2026-05-05
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260505_0008"
down_revision: str | None = "20260505_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WATERMARK_INDEXES = (
    (
        "idx_source_record_versions_watermark",
        "source_record_versions",
    ),
    (
        "idx_ontology_object_versions_watermark",
        "ontology_object_versions",
    ),
    (
        "idx_ontology_relation_versions_watermark",
        "ontology_relation_versions",
    ),
    (
        "idx_computed_snapshot_versions_watermark",
        "computed_snapshot_versions",
    ),
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("Async job freshness TTL migration requires PostgreSQL.")

    for index_name, table_name in _WATERMARK_INDEXES:
        op.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table_name} ((GREATEST(tx_from, COALESCE(tx_to, '-infinity'::timestamptz))))
            """
        )

    op.execute(
        """
        UPDATE async_jobs
        SET result_expires_at = LEAST(
            COALESCE(result_expires_at, completed_at + INTERVAL '5 minutes'),
            completed_at + INTERVAL '5 minutes'
        )
        WHERE status = 'completed'
          AND job_type IN ('analyzer', 'sizer', 'hedging')
          AND completed_at IS NOT NULL
          AND (
            result_expires_at IS NULL
            OR result_expires_at > completed_at + INTERVAL '5 minutes'
          )
        """
    )

    op.execute(
        """
        UPDATE async_jobs
        SET result_expires_at = LEAST(
            COALESCE(result_expires_at, completed_at + INTERVAL '60 seconds'),
            completed_at + INTERVAL '60 seconds'
        )
        WHERE status = 'completed'
          AND job_type = 'ontology'
          AND completed_at IS NOT NULL
          AND NULLIF(payload_json->>'run_id', '') IS NULL
          AND NULLIF(payload_json->>'as_of', '') IS NULL
          AND NULLIF(payload_json->>'tx_as_of', '') IS NULL
          AND (
            result_expires_at IS NULL
            OR result_expires_at > completed_at + INTERVAL '60 seconds'
          )
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("Async job freshness TTL migration requires PostgreSQL.")

    for index_name, _table_name in reversed(_WATERMARK_INDEXES):
        op.execute(f"DROP INDEX IF EXISTS {index_name}")
