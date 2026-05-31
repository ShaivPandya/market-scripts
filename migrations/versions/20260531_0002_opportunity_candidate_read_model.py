"""Add OpportunityCandidate ontology objects to operational read model.

Revision ID: 20260531_0002
Revises: 20260531_0001
Create Date: 2026-05-31
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260531_0002"
down_revision: str | None = "20260531_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BASE_READ_MODEL_VIEWS = (
    "ontology_current_position_risk_read_model",
    "ontology_current_position_signal_evidence_read_model",
    "ontology_current_position_thesis_context_read_model",
    "ontology_current_decision_lineage_read_model",
    "ontology_current_source_status_read_model",
    "ontology_current_computed_snapshot_read_model",
)
OPERATIONAL_READ_MODEL_VIEW = "ontology_current_operational_object_read_model"
PROVENANCE_GRAPH_READ_MODEL_VIEW = "ontology_current_provenance_graph_edge_read_model"
READ_MODEL_VIEWS = (*BASE_READ_MODEL_VIEWS, OPERATIONAL_READ_MODEL_VIEW, PROVENANCE_GRAPH_READ_MODEL_VIEW)

LEGACY_OPERATIONAL_OBJECT_TYPES = (
    "Position",
    "HedgePosition",
    "Thesis",
    "Evaluation",
    "Catalyst",
    "KillCondition",
    "ThesisClaim",
    "WorkflowRun",
    "ReportRun",
    "ActionItem",
    "WatchTrigger",
    "Approval",
    "Recommendation",
    "OptimizationAlert",
    "OptimizationActionSnapshot",
    "SourceFreshness",
    "ManagementQualityAssessment",
    "ManagementQualityScorecardRow",
    "ManagementQualityAccomplishment",
    "ManagementQualitySetback",
)
COA_OPERATIONAL_OBJECT_TYPES = (
    "CourseOfAction",
    "CourseOfActionComparison",
    "ScenarioAssumption",
    "SimulatedOutcome",
    "CourseOfActionRationale",
    "CourseOfActionDissent",
)
MONITOR_HIT_OPERATIONAL_OBJECT_TYPES = ("MonitorHit",)
DECISION_OUTCOME_OPERATIONAL_OBJECT_TYPES = ("DecisionOutcome",)
OPPORTUNITY_CANDIDATE_OPERATIONAL_OBJECT_TYPES = ("OpportunityCandidate",)
OPERATIONAL_OBJECT_TYPES = (
    *LEGACY_OPERATIONAL_OBJECT_TYPES,
    *COA_OPERATIONAL_OBJECT_TYPES,
    *MONITOR_HIT_OPERATIONAL_OBJECT_TYPES,
    *DECISION_OUTCOME_OPERATIONAL_OBJECT_TYPES,
    *OPPORTUNITY_CANDIDATE_OPERATIONAL_OBJECT_TYPES,
)
PRE_OPPORTUNITY_CANDIDATE_OPERATIONAL_OBJECT_TYPES = (
    *LEGACY_OPERATIONAL_OBJECT_TYPES,
    *COA_OPERATIONAL_OBJECT_TYPES,
    *MONITOR_HIT_OPERATIONAL_OBJECT_TYPES,
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


def _create_operational_view(object_types: Sequence[str]) -> None:
    object_type_array = _pg_text_array(object_types)
    op.execute(
        f"""
        CREATE MATERIALIZED VIEW {OPERATIONAL_READ_MODEL_VIEW} AS
        SELECT
          version_id,
          object_uid,
          object_type,
          business_key,
          properties_json,
          schema_name,
          schema_version,
          source_record_id,
          valid_from,
          valid_to,
          tx_from,
          tx_to,
          actor_id,
          input_hash,
          supersedes_version_id,
          temporal_confidence,
          upper(NULLIF(properties_json->>'ticker', '')) AS ticker,
          lower(NULLIF(properties_json->>'status', '')) AS status,
          lower(NULLIF(properties_json->>'application_status', '')) AS application_status,
          lower(NULLIF(properties_json->>'approval_status', '')) AS approval_status,
          lower(NULLIF(properties_json->>'outcome_status', '')) AS outcome_status,
          lower(NULLIF(properties_json->>'report_type', '')) AS report_type,
          NULLIF(properties_json->>'parent_uid', '') AS parent_uid,
          NULLIF(properties_json->>'assessment_id', '') AS assessment_id,
          NULLIF(properties_json->>'run_id', '') AS run_id,
          NULLIF(properties_json->>'current_snapshot_id', '') AS current_snapshot_id,
          NULLIF(properties_json->>'previous_snapshot_id', '') AS previous_snapshot_id,
          COALESCE(NULLIF(properties_json->>'as_of', ''), '') AS as_of_sort,
          COALESCE(NULLIF(properties_json->>'evaluated_at', ''), '') AS evaluated_at_sort,
          COALESCE(NULLIF(properties_json->>'created_at', ''), '') AS created_at_sort,
          COALESCE(NULLIF(properties_json->>'updated_at', ''), tx_from::text, '') AS updated_sort
        FROM ontology_object_versions
        WHERE tx_to IS NULL
          AND valid_from <= clock_timestamp()
          AND (valid_to IS NULL OR valid_to > clock_timestamp())
          AND object_type = ANY({object_type_array})
        """
    )
    op.execute(
        f"""
        CREATE UNIQUE INDEX uq_ontology_operational_read_model_object_uid
        ON {OPERATIONAL_READ_MODEL_VIEW}(object_uid)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_operational_read_model_ticker_status
        ON {OPERATIONAL_READ_MODEL_VIEW}(object_type, ticker, status)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_operational_read_model_status
        ON {OPERATIONAL_READ_MODEL_VIEW}(object_type, status)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_operational_read_model_approval_status
        ON {OPERATIONAL_READ_MODEL_VIEW}(object_type, approval_status)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_operational_read_model_report_as_of
        ON {OPERATIONAL_READ_MODEL_VIEW}(object_type, report_type, as_of_sort DESC)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_operational_read_model_parent
        ON {OPERATIONAL_READ_MODEL_VIEW}(object_type, parent_uid)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_operational_read_model_assessment
        ON {OPERATIONAL_READ_MODEL_VIEW}(object_type, assessment_id)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_operational_read_model_run
        ON {OPERATIONAL_READ_MODEL_VIEW}(object_type, run_id)
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_ontology_operational_read_model_updated
        ON {OPERATIONAL_READ_MODEL_VIEW}(object_type, updated_sort DESC)
        """
    )


def _replace_operational_view(object_types: Sequence[str]) -> None:
    op.execute("DROP FUNCTION IF EXISTS refresh_ontology_temporal_read_models()")
    op.execute(f"DROP MATERIALIZED VIEW IF EXISTS {OPERATIONAL_READ_MODEL_VIEW}")
    _create_operational_view(object_types)
    _create_refresh_function(READ_MODEL_VIEWS)
    _grant_postgres(READ_MODEL_VIEWS)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    _replace_operational_view(OPERATIONAL_OBJECT_TYPES)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    _replace_operational_view(PRE_OPPORTUNITY_CANDIDATE_OPERATIONAL_OBJECT_TYPES)
