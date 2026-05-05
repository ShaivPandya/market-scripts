"""Add temporal ontology read models.

Revision ID: 20260505_0006
Revises: 20260505_0005
Create Date: 2026-05-05
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260505_0006"
down_revision: str | None = "20260505_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("Temporal ontology read models require PostgreSQL.")

    op.execute(
        """
        CREATE MATERIALIZED VIEW ontology_current_position_risk_read_model AS
        WITH objs AS (
          SELECT *
          FROM ontology_object_versions
          WHERE tx_to IS NULL
            AND valid_from <= clock_timestamp()
            AND (valid_to IS NULL OR valid_to > clock_timestamp())
        ),
        rels AS (
          SELECT *
          FROM ontology_relation_versions
          WHERE tx_to IS NULL
            AND valid_from <= clock_timestamp()
            AND (valid_to IS NULL OR valid_to > clock_timestamp())
        )
        SELECT
          p.object_uid AS position_id,
          p.business_key AS position_business_key,
          COALESCE(p.properties_json->>'ticker', replace(p.object_uid, 'position:', '')) AS ticker,
          p.properties_json->>'asset' AS asset,
          p.properties_json->>'direction' AS direction,
          NULLIF(p.properties_json->>'risk_score', '')::double precision AS risk_score_value,
          p.properties_json AS position_props,
          p.schema_name AS position_schema_name,
          p.schema_version AS position_schema_version,
          p.tx_from AS position_updated_at,
          p.version_id AS position_version_id,
          p.valid_from AS position_valid_from,
          p.valid_to AS position_valid_to,
          p.tx_from AS position_tx_from,
          p.tx_to AS position_tx_to,
          p.temporal_confidence AS position_temporal_confidence,
          COALESCE(p.properties_json->>'ticker', replace(p.object_uid, 'position:', '')) AS position_label,
          a.object_uid AS asset_id,
          a.properties_json AS asset_props,
          a.schema_name AS asset_schema_name,
          a.schema_version AS asset_schema_version,
          a.tx_from AS asset_updated_at,
          a.properties_json->>'ticker' AS asset_label,
          s.object_uid AS sector_id,
          s.properties_json AS sector_props,
          s.schema_name AS sector_schema_name,
          s.schema_version AS sector_schema_version,
          s.tx_from AS sector_updated_at,
          s.properties_json->>'name' AS sector_label,
          COALESCE(s.properties_json->>'name', 'Unknown Equity') AS sector,
          pa.properties_json AS position_asset_edge_props,
          pa.relation_schema_name AS position_asset_edge_schema_name,
          pa.relation_schema_version AS position_asset_edge_schema_version,
          pa.relation_schema_name AS position_asset_edge_relation_schema_name,
          pa.relation_schema_version AS position_asset_edge_relation_schema_version,
          pa.tx_from AS position_asset_edge_updated_at,
          ase.properties_json AS asset_sector_edge_props,
          ase.relation_schema_name AS asset_sector_edge_schema_name,
          ase.relation_schema_version AS asset_sector_edge_schema_version,
          ase.relation_schema_name AS asset_sector_edge_relation_schema_name,
          ase.relation_schema_version AS asset_sector_edge_relation_schema_version,
          ase.tx_from AS asset_sector_edge_updated_at
        FROM objs p
        LEFT JOIN rels pa
          ON pa.source_object_uid = p.object_uid
         AND pa.relation_type = 'references_asset'
        LEFT JOIN objs a
          ON a.object_uid = pa.target_object_uid
         AND a.object_type = 'Asset'
        LEFT JOIN rels ase
          ON ase.source_object_uid = a.object_uid
         AND ase.relation_type = 'belongs_to_sector'
        LEFT JOIN objs s
          ON s.object_uid = ase.target_object_uid
         AND s.object_type = 'Sector'
        WHERE p.object_type = 'Position'
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_ontology_position_read_model_position
        ON ontology_current_position_risk_read_model(position_id)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_ontology_position_read_model_filters
        ON ontology_current_position_risk_read_model(upper(ticker), lower(asset), lower(sector))
        """
    )
    op.execute(
        """
        CREATE INDEX idx_ontology_position_read_model_risk
        ON ontology_current_position_risk_read_model(risk_score_value DESC NULLS LAST, position_id)
        """
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW ontology_current_position_signal_evidence_read_model AS
        WITH objs AS (
          SELECT *
          FROM ontology_object_versions
          WHERE tx_to IS NULL
            AND valid_from <= clock_timestamp()
            AND (valid_to IS NULL OR valid_to > clock_timestamp())
        ),
        rels AS (
          SELECT *
          FROM ontology_relation_versions
          WHERE tx_to IS NULL
            AND valid_from <= clock_timestamp()
            AND (valid_to IS NULL OR valid_to > clock_timestamp())
        )
        SELECT
          ps.source_object_uid AS position_id,
          s.object_uid AS signal_id,
          COALESCE(s.properties_json->>'name', s.business_key, s.object_uid) AS signal_label,
          s.properties_json AS signal_props,
          s.schema_name AS signal_schema_name,
          s.schema_version AS signal_schema_version,
          s.tx_from AS signal_updated_at,
          ps.properties_json AS edge_props,
          ps.relation_schema_name AS edge_schema_name,
          ps.relation_schema_version AS edge_schema_version,
          ps.relation_schema_name AS relation_schema_name,
          ps.relation_schema_version AS relation_schema_version,
          ps.tx_from AS edge_updated_at
        FROM rels ps
        JOIN objs s
          ON s.object_uid = ps.target_object_uid
         AND s.object_type = 'Signal'
        WHERE ps.relation_type = 'exposed_to_signal'
        """
    )
    op.execute(
        """
        CREATE INDEX idx_ontology_signal_evidence_read_model_position
        ON ontology_current_position_signal_evidence_read_model(position_id, signal_id)
        """
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW ontology_current_position_thesis_context_read_model AS
        WITH objs AS (
          SELECT *
          FROM ontology_object_versions
          WHERE tx_to IS NULL
            AND valid_from <= clock_timestamp()
            AND (valid_to IS NULL OR valid_to > clock_timestamp())
        ),
        rels AS (
          SELECT *
          FROM ontology_relation_versions
          WHERE tx_to IS NULL
            AND valid_from <= clock_timestamp()
            AND (valid_to IS NULL OR valid_to > clock_timestamp())
        ),
        theses AS (
          SELECT
            ht.source_object_uid AS position_id,
            ht.target_object_uid AS thesis_id,
            ht.properties_json AS source_edge_props,
            ht.relation_schema_name AS source_edge_schema_name,
            ht.relation_schema_version AS source_edge_schema_version,
            ht.tx_from AS source_edge_updated_at,
            t.object_uid AS target_id,
            t.object_type AS target_type,
            t.properties_json AS target_props,
            t.schema_name AS target_schema_name,
            t.schema_version AS target_schema_version,
            t.tx_from AS target_updated_at,
            COALESCE(t.properties_json->>'ticker', t.business_key, t.object_uid) AS target_label,
            'thesis'::text AS context_type
          FROM rels ht
          JOIN objs t ON t.object_uid = ht.target_object_uid AND t.object_type = 'Thesis'
          WHERE ht.relation_type = 'has_thesis'
        )
        SELECT * FROM theses
        UNION ALL
        SELECT
          th.position_id,
          th.thesis_id,
          eb.properties_json,
          eb.relation_schema_name,
          eb.relation_schema_version,
          eb.tx_from,
          e.object_uid,
          e.object_type,
          e.properties_json,
          e.schema_name,
          e.schema_version,
          e.tx_from,
          COALESCE(e.properties_json->>'evaluated_at', e.business_key, e.object_uid),
          'evaluation'::text
        FROM theses th
        JOIN rels eb ON eb.source_object_uid = th.thesis_id AND eb.relation_type = 'evaluated_by'
        JOIN objs e ON e.object_uid = eb.target_object_uid AND e.object_type = 'Evaluation'
        UNION ALL
        SELECT
          th.position_id,
          th.thesis_id,
          hc.properties_json,
          hc.relation_schema_name,
          hc.relation_schema_version,
          hc.tx_from,
          c.object_uid,
          c.object_type,
          c.properties_json,
          c.schema_name,
          c.schema_version,
          c.tx_from,
          COALESCE(c.properties_json->>'name', c.business_key, c.object_uid),
          'catalyst'::text
        FROM theses th
        JOIN rels hc ON hc.source_object_uid = th.thesis_id AND hc.relation_type = 'has_catalyst'
        JOIN objs c ON c.object_uid = hc.target_object_uid AND c.object_type = 'Catalyst'
        """
    )
    op.execute(
        """
        CREATE INDEX idx_ontology_thesis_context_read_model_position
        ON ontology_current_position_thesis_context_read_model(position_id, context_type, target_id)
        """
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW ontology_current_decision_lineage_read_model AS
        SELECT
          r.relation_uid,
          r.relation_type,
          r.source_object_uid,
          so.object_type AS source_object_type,
          so.business_key AS source_business_key,
          r.target_object_uid,
          ta.object_type AS target_object_type,
          ta.business_key AS target_business_key,
          r.properties_json,
          r.provenance_event_id,
          r.action_run_id,
          r.approval_id,
          r.valid_from,
          r.valid_to,
          r.tx_from,
          r.tx_to,
          r.temporal_confidence
        FROM ontology_relation_versions r
        LEFT JOIN ontology_object_versions so
          ON so.object_uid = r.source_object_uid
         AND so.tx_to IS NULL
         AND so.valid_from <= clock_timestamp()
         AND (so.valid_to IS NULL OR so.valid_to > clock_timestamp())
        LEFT JOIN ontology_object_versions ta
          ON ta.object_uid = r.target_object_uid
         AND ta.tx_to IS NULL
         AND ta.valid_from <= clock_timestamp()
         AND (ta.valid_to IS NULL OR ta.valid_to > clock_timestamp())
        WHERE r.tx_to IS NULL
          AND r.valid_from <= clock_timestamp()
          AND (r.valid_to IS NULL OR r.valid_to > clock_timestamp())
          AND r.relation_type IN (
            'workflow_run_produces_artifact',
            'workflow_artifact_proposes_approval',
            'report_run_produces_recommendation',
            'recommendation_supported_by_source_record',
            'recommendation_uses_risk_metric',
            'recommendation_uses_scenario',
            'trade_proposal_derives_from_recommendation',
            'trade_proposal_requires_approval',
            'approval_applies_action_run',
            'action_run_produces_executed_action',
            'executed_action_mutates_object_version',
            'action_run_mutates_object_version',
            'audit_event_observes_action_run'
          )
        """
    )
    op.execute(
        """
        CREATE INDEX idx_ontology_decision_lineage_read_model_source
        ON ontology_current_decision_lineage_read_model(source_object_uid, relation_type)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_ontology_decision_lineage_read_model_target
        ON ontology_current_decision_lineage_read_model(target_object_uid, relation_type)
        """
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW ontology_current_source_status_read_model AS
        SELECT DISTINCT ON (source_name)
          source_name,
          status,
          quality,
          as_of,
          load_time,
          provenance_event_id
        FROM source_record_versions
        WHERE tx_to IS NULL
          AND valid_from <= clock_timestamp()
          AND (valid_to IS NULL OR valid_to > clock_timestamp())
        ORDER BY source_name, load_time DESC
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_ontology_source_status_read_model_source
        ON ontology_current_source_status_read_model(source_name)
        """
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW ontology_current_computed_snapshot_read_model AS
        SELECT
          snapshot_key,
          payload_hash,
          payload_json,
          artifact_uri,
          as_of,
          load_time,
          status,
          quality,
          error,
          source_record_ids,
          provenance_event_id,
          valid_from,
          valid_to,
          tx_from,
          tx_to
        FROM computed_snapshot_versions
        WHERE tx_to IS NULL
          AND valid_from <= clock_timestamp()
          AND (valid_to IS NULL OR valid_to > clock_timestamp())
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_ontology_computed_snapshot_read_model_key
        ON ontology_current_computed_snapshot_read_model(snapshot_key)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_ontology_computed_snapshot_read_model_status
        ON ontology_current_computed_snapshot_read_model(status, quality)
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        raise RuntimeError("Temporal ontology read models require PostgreSQL.")

    op.execute("DROP MATERIALIZED VIEW IF EXISTS ontology_current_computed_snapshot_read_model")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ontology_current_source_status_read_model")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ontology_current_decision_lineage_read_model")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ontology_current_position_thesis_context_read_model")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ontology_current_position_signal_evidence_read_model")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ontology_current_position_risk_read_model")
