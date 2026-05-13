"""Add provenance graph read model.

Revision ID: 20260513_0001
Revises: 20260511_0001
Create Date: 2026-05-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260513_0001"
down_revision: str | None = "20260511_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BASE_READ_MODEL_VIEWS = (
    "ontology_current_position_risk_read_model",
    "ontology_current_position_signal_evidence_read_model",
    "ontology_current_position_thesis_context_read_model",
    "ontology_current_decision_lineage_read_model",
    "ontology_current_source_status_read_model",
    "ontology_current_computed_snapshot_read_model",
    "ontology_current_operational_object_read_model",
)
PROVENANCE_GRAPH_READ_MODEL_VIEW = "ontology_current_provenance_graph_edge_read_model"
READ_MODEL_VIEWS = (*BASE_READ_MODEL_VIEWS, PROVENANCE_GRAPH_READ_MODEL_VIEW)
PROVENANCE_RELATION_TYPES = (
    "provenance_used",
    "provenance_produced",
    "provenance_schema_bound",
    "provenance_executed",
    "provenance_executed_as",
    "provenance_triggered",
    "provenance_proposed",
    "provenance_resolved_by",
    "provenance_approved_execution",
    "provenance_audited_by",
    "provenance_updated",
)


def _pg_text_array(values: Sequence[str]) -> str:
    return "ARRAY[" + ", ".join(f"'{value}'" for value in values) + "]"


def _create_refresh_function(view_names: Sequence[str]) -> None:
    refresh_statements = "\n          ".join(f"REFRESH MATERIALIZED VIEW {view_name};" for view_name in view_names)
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION refresh_ontology_temporal_read_models()
        RETURNS void
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = public, pg_temp
        AS $$
        BEGIN
          {refresh_statements}
        END;
        $$;

        REVOKE ALL ON FUNCTION refresh_ontology_temporal_read_models() FROM PUBLIC;
        """
    )


def _grant_postgres(view_names: Sequence[str]) -> None:
    view_name_array = _pg_text_array(view_names)
    op.execute(
        f"""
        DO $$
        DECLARE
            view_name text;
            supports_maintain boolean := current_setting('server_version_num')::integer >= 170000;
        BEGIN
            FOREACH view_name IN ARRAY {view_name_array}
            LOOP
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'talisman_app') THEN
                    EXECUTE format('GRANT SELECT ON TABLE %I TO talisman_app', view_name);
                    IF supports_maintain THEN
                        EXECUTE format('GRANT MAINTAIN ON TABLE %I TO talisman_app', view_name);
                    END IF;
                END IF;

                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'talisman_worker') THEN
                    EXECUTE format('GRANT SELECT ON TABLE %I TO talisman_worker', view_name);
                    IF supports_maintain THEN
                        EXECUTE format('GRANT MAINTAIN ON TABLE %I TO talisman_worker', view_name);
                    END IF;
                END IF;
            END LOOP;

            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'talisman_app') THEN
                GRANT EXECUTE ON FUNCTION refresh_ontology_temporal_read_models() TO talisman_app;
            END IF;

            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'talisman_worker') THEN
                GRANT EXECUTE ON FUNCTION refresh_ontology_temporal_read_models() TO talisman_worker;
            END IF;
        END $$;
        """
    )


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    provenance_relation_types = _pg_text_array(PROVENANCE_RELATION_TYPES)
    op.execute(
        f"""
        CREATE MATERIALIZED VIEW {PROVENANCE_GRAPH_READ_MODEL_VIEW} AS
        SELECT
          relation_uid,
          version_id::text AS version_id,
          source_object_uid,
          target_object_uid,
          relation_type,
          NULLIF(properties_json->>'event_id', '') AS event_id,
          NULLIF(properties_json->>'source_ref_type', '') AS source_ref_type,
          NULLIF(properties_json->>'source_ref_id', '') AS source_ref_id,
          NULLIF(properties_json->>'source_ref_version', '') AS source_ref_version,
          NULLIF(properties_json->>'target_ref_type', '') AS target_ref_type,
          NULLIF(properties_json->>'target_ref_id', '') AS target_ref_id,
          NULLIF(properties_json->>'target_ref_version', '') AS target_ref_version,
          NULLIF(properties_json->>'lineage_root_id', '') AS lineage_root_id,
          COALESCE(NULLIF(properties_json->>'redaction_policy', ''), 'audit_summary_v1') AS redaction_policy,
          COALESCE(NULLIF(properties_json->>'retention_class', ''), 'provenance_365d') AS retention_class,
          properties_json->'metadata' AS metadata_json,
          valid_from,
          valid_to,
          tx_from,
          tx_to
        FROM ontology_relation_versions
        WHERE tx_to IS NULL
          AND valid_from <= clock_timestamp()
          AND (valid_to IS NULL OR valid_to > clock_timestamp())
          AND relation_type = ANY({provenance_relation_types})
        """
    )
    op.execute(
        f"""
        CREATE UNIQUE INDEX uq_ontology_provenance_graph_edge_read_model_relation
        ON {PROVENANCE_GRAPH_READ_MODEL_VIEW}(relation_uid)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_provenance_graph_edge_source_ref
        ON {PROVENANCE_GRAPH_READ_MODEL_VIEW}(source_ref_type, source_ref_id)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_provenance_graph_edge_target_ref
        ON {PROVENANCE_GRAPH_READ_MODEL_VIEW}(target_ref_type, target_ref_id)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_provenance_graph_edge_event
        ON {PROVENANCE_GRAPH_READ_MODEL_VIEW}(event_id)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_provenance_graph_edge_lineage_root
        ON {PROVENANCE_GRAPH_READ_MODEL_VIEW}(lineage_root_id)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_provenance_graph_edge_relation_time
        ON {PROVENANCE_GRAPH_READ_MODEL_VIEW}(relation_type, valid_from)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_provenance_graph_edge_source_uid
        ON {PROVENANCE_GRAPH_READ_MODEL_VIEW}(source_object_uid)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_provenance_graph_edge_target_uid
        ON {PROVENANCE_GRAPH_READ_MODEL_VIEW}(target_object_uid)
        """
    )
    _create_refresh_function(READ_MODEL_VIEWS)
    _grant_postgres(READ_MODEL_VIEWS)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DROP FUNCTION IF EXISTS refresh_ontology_temporal_read_models()")
    op.execute(f"DROP MATERIALIZED VIEW IF EXISTS {PROVENANCE_GRAPH_READ_MODEL_VIEW}")
    _create_refresh_function(BASE_READ_MODEL_VIEWS)
    _grant_postgres(BASE_READ_MODEL_VIEWS)
