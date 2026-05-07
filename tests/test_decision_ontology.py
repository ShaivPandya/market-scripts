from __future__ import annotations

from datetime import UTC, datetime

from ontology.decision_writeback import DecisionOntologyWriteback
from ontology.models import OntologyNode
from ontology.object_service import OntologyObjectService, source_record_object_uid_for
from ontology.schemas.identity import (
    audit_event_id,
    executed_action_id,
    object_version_ref_id,
    recommendation_id,
    source_record_object_id,
)
from ontology.schemas.objects import (
    AuditEventV1,
    ExecutedActionV1,
    ObjectVersionRefV1,
    RecommendationV1,
    SourceRecordV1,
)
from ontology.schemas.registry import normalize_node
from ontology.schemas.relations import get_relation_definition
from ontology.temporal_repository import ObjectVersionWrite, RelationVersionWrite


class _FakeTemporalRepo:
    def __init__(self):
        self.object_writes: list[ObjectVersionWrite] = []
        self.relation_writes: list[RelationVersionWrite] = []

    def write_object_version(self, write: ObjectVersionWrite):
        self.object_writes.append(write)
        return {
            "version_id": f"version-{len(self.object_writes)}",
            "object_uid": write.object_uid,
            "object_type": write.object_type,
            "business_key": write.business_key,
            "schema_name": write.schema_name,
            "schema_version": write.schema_version,
            "properties_json": write.properties,
            "valid_from": datetime(2026, 5, 4, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 4, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }

    def write_relation_version(self, write: RelationVersionWrite):
        self.relation_writes.append(write)
        return {
            "version_id": f"relation-{len(self.relation_writes)}",
            "relation_uid": write.relation_uid,
            "source_object_uid": write.source_object_uid,
            "target_object_uid": write.target_object_uid,
            "relation_type": write.relation_type,
            "relation_schema_name": write.relation_schema_name,
            "relation_schema_version": write.relation_schema_version,
            "properties_json": write.properties,
            "valid_from": datetime(2026, 5, 4, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 4, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }


def test_decision_object_schemas_have_stable_identities():
    source = normalize_node(
        OntologyNode(
            id=source_record_object_id("report:daily:payload"),
            type="SourceRecord",
            label="report source",
            properties=SourceRecordV1(
                source_record_id="report:daily:payload",
                vendor="github_actions",
                source_name="daily_report_sync",
                dataset="report_sync",
                record_kind="report_payload",
                record_key_hash="abc",
                payload_hash="def",
            ).model_dump(mode="json"),
            schema_name="SourceRecord",
            schema_version=1,
        ),
        allow_legacy=False,
    )
    version_ref = normalize_node(
        OntologyNode(
            id=object_version_ref_id("action_item:1:version-1"),
            type="ObjectVersionRef",
            label="version ref",
            properties=ObjectVersionRefV1(
                ref_id="action_item:1:version-1",
                object_uid="action_item:1",
                version_id="version-1",
            ).model_dump(mode="json"),
            schema_name="ObjectVersionRef",
            schema_version=1,
        ),
        allow_legacy=False,
    )
    executed = normalize_node(
        OntologyNode(
            id=executed_action_id("1:2:create_action_item"),
            type="ExecutedAction",
            label="executed action",
            properties=ExecutedActionV1(
                executed_action_id="1:2:create_action_item",
                action_id="create_action_item",
            ).model_dump(mode="json"),
            schema_name="ExecutedAction",
            schema_version=1,
        ),
        allow_legacy=False,
    )
    audit = normalize_node(
        OntologyNode(
            id=audit_event_id("evt-1"),
            type="AuditEvent",
            label="audit event",
            properties=AuditEventV1(
                event_id="evt-1",
                action_name="approval.applied",
                action_category="approval",
                status="succeeded",
            ).model_dump(mode="json"),
            schema_name="AuditEvent",
            schema_version=1,
        ),
        allow_legacy=False,
    )
    recommendation = normalize_node(
        OntologyNode(
            id=recommendation_id("daily:2026-05-02:buy:MU"),
            type="Recommendation",
            label="recommendation",
            properties=RecommendationV1(
                recommendation_id="daily:2026-05-02:buy:MU",
                report_type="daily",
                as_of="2026-05-02",
                action="buy",
                ticker="MU",
                instrument="MU",
                decision_state="proposed",
                approval_required=True,
            ).model_dump(mode="json"),
            schema_name="Recommendation",
            schema_version=1,
        ),
        allow_legacy=False,
    )

    assert source.id == "source_record:report_daily_payload"
    assert version_ref.id == "object_version_ref:action_item_1_version_1"
    assert executed.id == "executed_action:1_2_create_action_item"
    assert audit.id == "audit_event:evt_1"
    assert recommendation.id == "recommendation:daily_2026_05_02_buy_mu"


def test_source_record_identity_canonicalizes_logical_prefixed_ids():
    repo = _FakeTemporalRepo()
    service = OntologyObjectService(repository=repo)
    logical_id = "source_record:portfolio:portfolio_position:d1370b6e76212e53"
    canonical_uid = "source_record:source_record_portfolio_portfolio_position_d1370b6e76212e53"

    assert source_record_object_uid_for(logical_id) == canonical_uid
    assert source_record_object_uid_for(canonical_uid) == canonical_uid

    row = service.write_object(
        "SourceRecord",
        logical_id,
        SourceRecordV1(
            source_record_id=logical_id,
            vendor="portfolio",
            source_name="portfolio",
            dataset="portfolio",
            record_kind="portfolio_position",
            record_key_hash="abc",
            payload_hash="def",
        ).model_dump(mode="json"),
        "2026-05-04T00:00:00+00:00",
        provenance="pv:source-record-test",
    )

    assert row["object_uid"] == canonical_uid
    assert row["business_key"] == logical_id
    assert row["properties_json"]["source_record_id"] == logical_id


def test_audit_event_identity_accepts_prefixed_business_key():
    repo = _FakeTemporalRepo()
    service = OntologyObjectService(repository=repo)
    event_uid = audit_event_id("approval.created:abc")

    row = service.write_object(
        "AuditEvent",
        event_uid,
        AuditEventV1(
            event_id=event_uid,
            action_name="approval.created",
            action_category="approval",
            status="succeeded",
        ).model_dump(mode="json"),
        "2026-05-04T00:00:00+00:00",
        actor={"actor_type": "system", "actor_id": "test"},
        provenance="pv:test-audit",
    )

    assert audit_event_id(event_uid) == event_uid
    assert row["object_uid"] == event_uid


def test_decision_relation_registry_models_lineage_edges():
    expected = {
        "workflow_artifact_proposes_approval": ("WorkflowArtifact", "Approval"),
        "recommendation_supported_by_source_record": ("Recommendation", "SourceRecord"),
        "recommendation_uses_risk_metric": ("Recommendation", "RiskMetric"),
        "recommendation_uses_scenario": ("Recommendation", "Scenario"),
        "trade_proposal_targets_asset": ("TradeProposal", "Asset"),
        "trade_proposal_requires_approval": ("TradeProposal", "Approval"),
        "recommendation_supported_by_evidence": ("Recommendation", "Evidence"),
        "recommendation_contradicted_by_evidence": ("Recommendation", "Evidence"),
        "evidence_cites_citation": ("Evidence", "Citation"),
        "recommendation_has_policy_gate_result": ("Recommendation", "PolicyGateResult"),
        "recommendation_has_trade_proposal": ("Recommendation", "TradeProposal"),
        "recommendation_uses_position_risk_snapshot": ("Recommendation", "PositionRiskSnapshot"),
        "recommendation_uses_portfolio_risk_snapshot": ("Recommendation", "PortfolioRiskSnapshot"),
        "computed_snapshot_materializes_object_version": ("ComputedSnapshotRef", "ObjectVersionRef"),
        "market_regime_has_factor_score": ("MarketRegimeSnapshot", "SignalFactorScore"),
        "market_regime_has_forward_outlook": ("MarketRegimeSnapshot", "ForwardOutlook"),
        "market_regime_has_episode": ("MarketRegimeSnapshot", "RegimeEpisode"),
        "factor_score_uses_computed_snapshot": ("SignalFactorScore", "ComputedSnapshotRef"),
        "document_artifact_materializes_research_object": ("DocumentArtifact", "EquityOverview"),
        "equity_overview_has_financial_profile": ("EquityOverview", "CompanyFinancialProfile"),
        "thesis_document_has_section": ("ThesisDocument", "ThesisSection"),
        "action_run_produces_executed_action": ("ActionRun", "ExecutedAction"),
        "executed_action_mutates_object_version": ("ExecutedAction", "ObjectVersionRef"),
        "source_record_materializes_object_version": ("SourceRecord", "ObjectVersionRef"),
        "audit_event_observes_action_run": ("AuditEvent", "ActionRun"),
    }

    for relation_type, endpoint_types in expected.items():
        definition = get_relation_definition(relation_type)
        assert (definition.source_type, definition.target_type) == endpoint_types


def test_decision_writeback_records_report_recommendation_lineage(monkeypatch):
    monkeypatch.setenv("ONTOLOGY_SHADOW_WRITES", "true")
    repo = _FakeTemporalRepo()
    service = DecisionOntologyWriteback(OntologyObjectService(repository=repo))

    rows = service.record_report_output(
        report_type="daily",
        payload={
            "as_of": "2026-05-02",
            "artifact_paths": {"recommendations_json": "/tmp/recs.json"},
            "risk_metrics": [{"metric_id": "mu_var", "metric": "var", "scope_type": "asset", "scope_id": "MU"}],
            "scenarios": [{"scenario_id": "mu_stress", "name": "MU stress", "scenario_type": "stress"}],
        },
        report_run={
            "report_id": "daily:2026-05-02",
            "report_type": "daily",
            "as_of": "2026-05-02",
            "status": "completed",
            "input_hash": "input-hash",
        },
        persisted_recommendations=[
            {
                "approval_id": 10,
                "record": {
                    "report_type": "daily",
                    "as_of": "2026-05-02",
                    "idempotency_key": "daily:2026-05-02:mu",
                    "action": "buy",
                    "ticker": "MU",
                    "instrument": "MU",
                    "account_id": "acct-1",
                    "portfolio_id": "portfolio-1",
                    "risk_metric_ids": ["mu_var"],
                    "scenario_ids": ["mu_stress"],
                    "rationale": "Validated setup.",
                    "confidence": 0.7,
                    "source_quality": "ok",
                    "policy_gate_result_id": 3,
                    "policy_gate_result": {"decision": "warn", "warnings": [{"code": "missing_constraint"}]},
                    "risk_snapshot_id": "mu-risk-2026-05-02",
                    "portfolio_risk_snapshot_id": "portfolio-risk-2026-05-02",
                    "risk_score": 0.42,
                    "risk_confidence": 0.8,
                    "risk_quality": "ok",
                    "risk_source_status": {
                        "quality": "ok",
                        "computed_at": "2026-05-02T12:00:00+00:00",
                        "average_risk_score": 0.31,
                        "max_risk_score": 0.42,
                        "position_count": 1,
                    },
                    "risk_bindings": [{"risk_snapshot_id": "mu-risk-2026-05-02"}],
                    "evidence": [
                        {
                            "source": "filing",
                            "summary": "HBM demand supports the proposed action.",
                            "url": "https://example.test/mu-filing",
                        }
                    ],
                    "disconfirming_evidence": [
                        {
                            "source": "liquidity",
                            "summary": "Liquidity remains mixed.",
                            "citation": {"url": "https://example.test/liquidity"},
                        }
                    ],
                },
            }
        ],
        provenance="pv:test-report",
    )

    object_types = {write.object_type for write in repo.object_writes}
    relation_types = {write.relation_type for write in repo.relation_writes}
    assert rows
    assert {
        "ReportRun",
        "SourceRecord",
        "RiskMetric",
        "Scenario",
        "Recommendation",
        "PolicyGateResult",
        "TradeProposal",
        "Evidence",
        "Citation",
        "PositionRiskSnapshot",
        "PortfolioRiskSnapshot",
    } <= object_types
    assert {
        "report_run_produces_recommendation",
        "recommendation_supported_by_source_record",
        "recommendation_supported_by_evidence",
        "recommendation_contradicted_by_evidence",
        "evidence_cites_citation",
        "recommendation_targets_account",
        "recommendation_targets_portfolio",
        "recommendation_uses_risk_metric",
        "recommendation_uses_scenario",
        "recommendation_uses_position_risk_snapshot",
        "recommendation_uses_portfolio_risk_snapshot",
        "policy_gate_evaluates_recommendation",
        "recommendation_has_policy_gate_result",
        "recommendation_has_trade_proposal",
        "trade_proposal_derives_from_recommendation",
        "trade_proposal_requires_approval",
        "approval_targets_trade_proposal",
    } <= relation_types


def test_decision_writeback_records_workflow_artifact_and_executed_action(monkeypatch):
    monkeypatch.setenv("ONTOLOGY_SHADOW_WRITES", "true")
    repo = _FakeTemporalRepo()
    service = DecisionOntologyWriteback(OntologyObjectService(repository=repo))

    service.record_workflow_artifact_proposal(
        run_id="run-1",
        artifact_key="action_items",
        artifact_index=0,
        artifact_value={"description": "Review MU"},
        approval_id=7,
        action_id="create_action_item",
        artifact_id="artifact-1",
        provenance="pv:artifact",
    )
    service.apply_approved_decision(
        approval_id=7,
        action_run_id=8,
        action_id="create_action_item",
        output={"id": 1},
        mutated_versions=[
            {
                "object_uid": "action_item:1",
                "object_type": "ActionItem",
                "_meta": {"temporal": {"version_id": "version-1"}},
            }
        ],
        provenance="pv:action_run:8",
    )

    object_types = {write.object_type for write in repo.object_writes}
    relation_types = {write.relation_type for write in repo.relation_writes}
    assert {"WorkflowArtifact", "ExecutedAction", "ObjectVersionRef"} <= object_types
    assert {
        "workflow_run_produces_artifact",
        "workflow_artifact_proposes_approval",
        "approval_targets_workflow_artifact",
        "action_run_produces_executed_action",
        "approval_applies_action_run",
        "executed_action_mutates_object_version",
    } <= relation_types
