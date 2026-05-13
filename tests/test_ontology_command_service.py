from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ontology.command_service import (
    OntologyCommandConflict,
    OntologyCommandContext,
    OntologyCommandService,
    OntologyCommandValidationError,
)
from ontology.object_service import OntologyObjectService
from ontology.policy import admin_actor
from ontology.schemas.identity import document_artifact_id
from ontology.temporal_repository import ObjectVersionWrite, RelationVersionWrite

SAMPLE_NEWS_DIGEST = """# Newsletter Digest - May 1, 2026

*Generated: 2026-05-01*

## liquidity_path

- [MULTI-SIGNAL] Japan launches FX intervention for the first time since 2024 - (Bloomberg) - [body content]
  - *MOF/BOJ-coordinated dollar-selling intervention with explicit final warning rhetoric.*
"""


class FakeObjectService:
    def __init__(self):
        self.objects: dict[str, dict[str, Any]] = {}
        self.relations: list[dict[str, Any]] = []

    def write_object(self, object_type, business_key, properties, valid_from, **kwargs):
        object_uid = str(business_key)
        if not object_uid.startswith(
            (
                "account:",
                "approval:",
                "audit_event:",
                "executed_decision_record:",
                "document_artifact:",
                "instrument:",
                "issuer:",
                "management_quality_accomplishment:",
                "management_quality_assessment:",
                "management_quality_scorecard_row:",
                "management_quality_setback:",
                "portfolio:",
                "position:",
                "recommendation:",
                "thesis:",
            )
        ):
            object_uid = f"{object_type.lower()}:{object_uid}"
        row = {
            "object_uid": object_uid,
            "object_type": object_type,
            "properties": dict(properties),
            "_meta": {"temporal": {"version_id": f"version:{len(self.objects) + 1}", "valid_from": str(valid_from)}},
        }
        self.objects[object_uid] = row
        return row

    def write_relation(self, source_uid, target_uid, relation_type, properties, valid_from, **kwargs):
        row = {
            "relation_uid": f"{relation_type}:{source_uid}:{target_uid}",
            "source_object_uid": source_uid,
            "target_object_uid": target_uid,
            "relation_type": relation_type,
            "properties": dict(properties or {}),
            "_meta": {"temporal": {"valid_from": str(valid_from)}},
        }
        self.relations.append(row)
        return row

    def get_object(self, object_uid, **kwargs):
        return self.objects.get(str(object_uid))

    def query_objects(self, object_type=None, filters=None, **kwargs):
        rows = [row for row in self.objects.values() if object_type is None or row["object_type"] == object_type]
        for key, value in (filters or {}).items():
            rows = [row for row in rows if row["properties"].get(key) == value]
        return rows


class NormalizingTemporalRepo:
    def __init__(self):
        self.objects: dict[str, dict[str, Any]] = {}
        self.relations: list[dict[str, Any]] = []
        self.version = 0

    def write_object_version(self, write: ObjectVersionWrite):
        self.version += 1
        row = {
            "version_id": f"version:{self.version}",
            "object_uid": write.object_uid,
            "object_type": write.object_type,
            "business_key": write.business_key,
            "schema_name": write.schema_name,
            "schema_version": write.schema_version,
            "properties_json": dict(write.properties),
            "valid_from": datetime(2026, 5, 6, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 6, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }
        self.objects[write.object_uid] = row
        return row

    def write_relation_version(self, write: RelationVersionWrite):
        row = {
            "version_id": f"relation:{len(self.relations) + 1}",
            "relation_uid": write.relation_uid,
            "source_object_uid": write.source_object_uid,
            "target_object_uid": write.target_object_uid,
            "relation_type": write.relation_type,
            "relation_schema_name": write.relation_schema_name,
            "relation_schema_version": write.relation_schema_version,
            "properties_json": dict(write.properties),
            "valid_from": datetime(2026, 5, 6, tzinfo=UTC),
            "valid_to": None,
            "tx_from": datetime(2026, 5, 6, tzinfo=UTC),
            "tx_to": None,
            "temporal_confidence": write.temporal_confidence,
        }
        self.relations.append(row)
        return row

    def get_object(self, object_uid, **kwargs):
        return self.objects.get(str(object_uid))

    def query_objects(self, object_type=None, filters=None, **kwargs):
        include_history = bool(kwargs.get("include_history"))
        rows = [
            row
            for row in self.objects.values()
            if (object_type is None or row["object_type"] == object_type)
            and (include_history or row.get("tx_to") is None)
        ]
        for key, value in (filters or {}).items():
            rows = [row for row in rows if row["properties_json"].get(key) == value]
        return rows

    def query_relations(self, relation_type=None, **kwargs):
        include_history = bool(kwargs.get("include_history"))
        source_uid = kwargs.get("source_object_uid")
        target_uid = kwargs.get("target_object_uid")
        rows = [
            row
            for row in self.relations
            if (relation_type is None or row["relation_type"] == relation_type)
            and (include_history or row.get("tx_to") is None)
            and (not source_uid or row["source_object_uid"] == source_uid)
            and (not target_uid or row["target_object_uid"] == target_uid)
        ]
        return rows

    def expire_object_versions(self, object_uid, **kwargs):
        row = self.objects.get(str(object_uid))
        if not row or row.get("tx_to") is not None:
            return 0
        row["tx_to"] = kwargs.get("tx_to") or datetime(2026, 5, 6, tzinfo=UTC)
        return 1

    def expire_relation_versions(self, relation_uid, **kwargs):
        count = 0
        for row in self.relations:
            if row.get("relation_uid") == relation_uid and row.get("tx_to") is None:
                row["tx_to"] = kwargs.get("tx_to") or datetime(2026, 5, 6, tzinfo=UTC)
                count += 1
        return count


def _isolate_news_digest_store(monkeypatch, tmp_path):
    import portfolio.news_digests as digests

    base = tmp_path / "news_digests"
    monkeypatch.setattr(digests, "DIGESTS_DIR", base)
    monkeypatch.setattr(digests, "MANIFEST_PATH", base / "manifest.json")
    monkeypatch.setattr(digests, "FILES_DIR", base / "files")
    monkeypatch.setattr(digests, "DIGESTS_GCS_PREFIX", "test/news_digests")
    monkeypatch.setattr(digests, "MANIFEST_GCS_KEY", "test/news_digests/manifest.json")
    monkeypatch.setattr(digests, "FILES_GCS_PREFIX", "test/news_digests/files")
    monkeypatch.setenv("STATE_STORAGE_BACKEND", "local")
    return digests


def test_propose_and_apply_position_update_writes_only_ontology_objects():
    service = OntologyCommandService(FakeObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(
        actor=admin_actor(source="test"),
        source_type="test",
        source_id="unit",
    )

    approval = service.propose_action(
        "update_portfolio_positions",
        {"positions": [{"ticker": "MU", "asset": "equity", "direction": "long", "shares": 10}]},
        context,
        reason="unit",
    )
    assert approval["id"].startswith("approval:")
    assert approval["status"] == "pending"

    applied = service.resolve_approval(approval["id"], "approved", "apply", context)
    assert applied["application_status"] == "applied"
    assert "position:MU" in service.objects.objects  # type: ignore[attr-defined]
    assert any(rel["relation_type"] == "position_references_instrument" for rel in service.objects.relations)  # type: ignore[attr-defined]
    assert any(rel["relation_type"] == "executed_decision_applies_approval" for rel in service.objects.relations)  # type: ignore[attr-defined]


def test_command_service_refreshes_read_model_after_position_approval_apply(monkeypatch):
    import ontology.domain_write_service as domain_write_service
    import ontology.read_model as read_model

    refresh_calls: list[str] = []

    class _Repo:
        def refresh(self):
            refresh_calls.append("refresh")

    monkeypatch.setattr(read_model, "TemporalReadModelRepository", _Repo)

    service = OntologyCommandService(FakeObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "update_portfolio_positions",
        {"positions": [{"ticker": "META", "asset": "equity", "direction": "long", "shares": 1}]},
        context,
        reason="unit",
    )
    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    assert applied["application_status"] == "applied"
    assert "position:META" in service.objects.objects  # type: ignore[attr-defined]
    assert refresh_calls == ["refresh", "refresh"]


def test_position_update_apply_preserves_negative_quantity():
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "update_portfolio_positions",
        {
            "positions": [
                {
                    "ticker": "SPY",
                    "asset": "equity",
                    "direction": "short",
                    "shares": -12,
                    "quantity": -12,
                }
            ]
        },
        context,
        reason="signed regular position",
    )

    applied = service.resolve_approval(approval["id"], "approved", "apply", context)
    position = repo.objects["position:SPY"]["properties_json"]

    assert applied["application_status"] == "applied"
    assert approval["proposed_change"]["positions"][0]["shares"] == -12
    assert position["shares"] == -12
    assert position["quantity"] == -12


def test_position_update_apply_accepts_reviewed_valuation_fields():
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "update_portfolio_positions",
        {
            "positions": [
                {
                    "ticker": "APO",
                    "asset": "equity",
                    "shares": 25,
                    "quantity": 25,
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 3,
                    "cost_basis": None,
                    "price_symbol": "APO",
                    "country": None,
                    "currency": None,
                    "exchange": None,
                    "base_currency": None,
                    "fx_rate_as_of": None,
                    "notional_base": None,
                    "cost_basis_base": None,
                    "fx_rate_to_base": None,
                    "instrument_type": "security",
                    "valuation_status": None,
                    "contract_multiplier": 1,
                }
            ]
        },
        context,
        reason="unit",
    )

    assert approval["proposed_change"]["position_changes"]
    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    assert applied["application_status"] == "applied"
    assert "position:APO" in repo.objects


def test_position_replacement_apply_expires_removed_positions():
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    initial = service.propose_action(
        "update_portfolio_positions",
        {
            "positions": [
                {"ticker": "MU", "asset": "equity", "direction": "long", "shares": 10},
                {"ticker": "OKLO", "asset": "equity", "direction": "short", "shares": 5},
            ]
        },
        context,
        reason="initial book",
    )
    service.resolve_approval(initial["id"], "approved", "apply", context)

    replacement = service.propose_action(
        "update_portfolio_positions",
        {"positions": [{"ticker": "MU", "asset": "equity", "direction": "long", "shares": 10}]},
        context,
        reason="remove OKLO",
    )
    applied = service.resolve_approval(replacement["id"], "approved", "apply", context)

    active_positions = service.objects.query_objects("Position")
    assert applied["application_status"] == "applied"
    assert [row["object_uid"] for row in active_positions] == ["position:MU"]
    assert repo.objects["position:OKLO"]["tx_to"] is not None


def test_hedge_update_apply_accepts_enriched_payload_fields():
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "update_hedge_positions",
        {
            "positions": [
                {
                    "ticker": "SPY",
                    "direction": "short",
                    "asset": "equity",
                    "shares": 5,
                    "quantity": 5,
                    "instrument_type": "security",
                    "price_symbol": "SPY",
                    "contrarian": False,
                    "conviction": 3,
                    "currency": None,
                    "country": None,
                    "exchange": None,
                    "base_currency": None,
                    "fx_rate_to_base": None,
                    "fx_rate_as_of": None,
                    "cost_basis_base": None,
                    "notional_base": None,
                    "valuation_status": None,
                }
            ]
        },
        context,
        reason="unit",
    )

    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    assert approval["entity_type"] == "hedge_positions"
    assert applied["application_status"] == "applied"
    assert "hedge_position:SPY" in repo.objects


def test_hedge_update_apply_preserves_negative_quantity():
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "update_hedge_positions",
        {
            "positions": [
                {
                    "ticker": "SPY",
                    "direction": "short",
                    "shares": -12,
                    "quantity": -12,
                }
            ]
        },
        context,
        reason="signed hedge",
    )

    applied = service.resolve_approval(approval["id"], "approved", "apply", context)
    hedge = repo.objects["hedge_position:SPY"]["properties_json"]

    assert applied["application_status"] == "applied"
    assert approval["proposed_change"]["positions"][0]["shares"] == -12
    assert hedge["shares"] == -12
    assert hedge["quantity"] == -12


def test_news_digest_create_approval_persists_digest_store_and_projection(monkeypatch, tmp_path):
    digests = _isolate_news_digest_store(monkeypatch, tmp_path)
    monkeypatch.setattr("api.routers.portfolio_news._index_digest_best_effort", lambda _detail: None)

    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "create_portfolio_news_digest",
        {"content": SAMPLE_NEWS_DIGEST, "filename": "05012026_digest.md"},
        context,
        reason="upload digest",
    )
    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    listed = digests.list_digests()
    digest_id = listed["items"][0]["id"]
    detail = digests.get_digest(digest_id)
    doc_uid = document_artifact_id("news_digest", digest_id)
    doc = repo.objects[doc_uid]["properties_json"]

    assert approval["entity_type"] == "news_digest_create"
    assert applied["application_status"] == "applied"
    assert listed["counts"]["digests"] == 1
    assert detail["content"] == SAMPLE_NEWS_DIGEST
    assert digest_id == "2026-05-01-newsletter-digest-may-1-2026"
    assert doc["document_id"] == digest_id
    assert doc["status"] == "active"
    assert "05012026_digest" not in doc_uid
    assert not doc_uid.startswith("document_artifact:news_digest:news_digest")


def test_news_digest_delete_approval_removes_digest_store_and_marks_projection(monkeypatch, tmp_path):
    digests = _isolate_news_digest_store(monkeypatch, tmp_path)
    monkeypatch.setattr("api.routers.portfolio_news._index_digest_best_effort", lambda _detail: None)
    monkeypatch.setattr("api.routers.portfolio_news._delete_digest_index_best_effort", lambda _digest_id: None)

    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    create = service.propose_action(
        "create_portfolio_news_digest",
        {"content": SAMPLE_NEWS_DIGEST, "filename": "05012026_digest.md"},
        context,
        reason="upload digest",
    )
    service.resolve_approval(create["id"], "approved", "apply", context)
    digest_id = digests.list_digests()["items"][0]["id"]

    delete = service.propose_action(
        "delete_portfolio_news_digest",
        {"digest_id": digest_id},
        context,
        reason="delete digest",
    )
    applied = service.resolve_approval(delete["id"], "approved", "apply", context)
    doc = repo.objects[document_artifact_id("news_digest", digest_id)]["properties_json"]

    assert delete["entity_type"] == "news_digest_delete"
    assert applied["application_status"] == "applied"
    assert digests.list_digests()["counts"]["digests"] == 0
    with pytest.raises(FileNotFoundError):
        digests.get_digest(digest_id)
    assert doc["document_id"] == digest_id
    assert doc["status"] == "deleted"


@pytest.mark.parametrize(
    ("action_id", "payload", "expected_type"),
    [
        ("update_catalyst_status", {"catalyst_id": 1, "status": "played_out"}, "Catalyst"),
        ("update_kill_condition_status", {"kill_condition_id": 1, "status": "triggered"}, "KillCondition"),
        ("update_thesis_claim", {"claim_id": 1, "status": "supported", "confidence": 0.8}, "ThesisClaim"),
    ],
)
def test_id_only_ontology_update_approvals_apply_without_ticker_context(action_id, payload, expected_type):
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(action_id, payload, context, reason="unit")
    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    assert applied["application_status"] == "applied"
    assert any(row["object_type"] == expected_type for row in repo.objects.values())


def test_create_catalyst_approval_defaults_to_pending_status():
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "create_catalyst",
        {"ticker": "META", "description": "AI ads: Monetization improves", "category": "fundamental"},
        context,
        reason="Create catalyst",
    )
    applied = service.resolve_approval(approval["id"], "approved", "apply", context)
    catalyst = next(row for row in repo.objects.values() if row["object_type"] == "Catalyst")

    assert applied["application_status"] == "applied"
    assert catalyst["properties_json"]["status"] == "pending"


def test_create_recommendation_approval_applies_with_real_schema_normalization():
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="workflow", source_id="daily")

    approval = service.propose_action(
        "create_recommendation",
        {
            "record": {
                "action": "rebalance",
                "instrument": "hedge_overlay",
                "report_type": "daily",
                "as_of": "2026-05-06",
                "confidence": 0.65,
                "horizon": "1 trading day",
                "rationale": "Rebalance hedge overlay.",
                "critical_data_quality": "ok",
                "idempotency_key": "daily:2026-05-06:hedge-overlay",
            }
        },
        context,
        reason="Daily recommendation for hedge_overlay",
    )

    applied = service.resolve_approval(approval["id"], "approved", "approved", context)

    assert applied["application_status"] == "applied"
    assert "recommendation:daily_2026_05_06_hedge_overlay" in repo.objects
    assert any(row["object_type"] == "ActionRun" for row in repo.objects.values())
    assert any(row["object_type"] == "ExecutedDecisionRecord" for row in repo.objects.values())


@pytest.mark.parametrize(
    ("action_id", "payload", "expected_uid"),
    [
        (
            "create_action_item",
            {
                "description": "Review daily report flag for OKLO 2026-05-06",
                "action_type": "review",
                "ticker": "OKLO",
                "urgency": "normal",
            },
            "action_item:review_daily_report_flag_for_oklo_2026_05_06",
        ),
        (
            "create_watch_trigger",
            {
                "condition": "Watch OKLO breadth reversal",
                "trigger_type": "custom",
                "ticker": "OKLO",
            },
            "watch_trigger:watch_oklo_breadth_reversal",
        ),
        (
            "create_research_note",
            {
                "title": "OKLO daily report flag",
                "ticker": "OKLO",
                "note": "Review daily report flag for OKLO.",
                "document_id": "daily:2026-05-06:oklo-flag",
            },
            "document_artifact:research_note:daily_2026_05_06_oklo_flag",
        ),
    ],
)
def test_research_object_approvals_apply_with_schema_canonical_ids(action_id, payload, expected_uid):
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="workflow", source_id="daily")

    approval = service.propose_action(action_id, payload, context, reason="Apply research object")
    applied = service.resolve_approval(approval["id"], "approved", "approved", context)

    assert applied["application_status"] == "applied"
    assert expected_uid in repo.objects


def test_unsupported_action_is_rejected_before_any_write():
    fake = FakeObjectService()
    service = OntologyCommandService(fake)  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    with pytest.raises(OntologyCommandValidationError):
        service.propose_action("unregistered_write", {}, context)
    assert fake.objects == {}


def test_restaged_approval_uses_distinct_uid_and_survives_original_rejection(monkeypatch):
    import portfolio.action_registry as action_registry

    monkeypatch.setattr(action_registry, "compute_action_base_state_hash", lambda _action_id, _payload: "base")
    service = OntologyCommandService(FakeObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")
    payload = {"item_id": 1, "resolution_note": "Done"}

    original = service.propose_action("complete_action_item", payload, context, reason="Complete item")
    replacement = service.propose_action(
        "complete_action_item",
        payload,
        context,
        reason="Restage item",
        supersedes_approval_id=original["id"],
    )

    assert replacement["id"] != original["id"]
    assert replacement["supersedes_approval_id"] == original["id"]

    rejected = service.resolve_approval(original["id"], "rejected", "Superseded", context)

    assert rejected["status"] == "rejected"
    assert service.get_approval(replacement["id"], actor=context.actor)["status"] == "pending"


def test_action_item_status_proposal_accepts_ontology_uid_and_keeps_item_context(monkeypatch):
    import ontology.runtime_read_service as runtime_read_service

    class _Reads:
        def get(self, object_uid: str):
            assert object_uid == "action_item:5"
            return {
                "id": "action_item:5",
                "ticker": "MU",
                "description": "Review MU thesis",
                "action_type": "review",
                "urgency": "normal",
                "status": "open",
            }

    monkeypatch.setattr(runtime_read_service, "OntologyRuntimeReadService", _Reads)
    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "complete_action_item",
        {"item_id": "action_item:5", "resolution_note": "Done"},
        context,
        reason="Complete item",
    )

    assert approval["entity_type"] == "action_item_status"
    assert approval["ticker"] == "MU"
    assert approval["target_object_uid"] == "action_item:5"
    assert approval["target_object_type"] == "ActionItem"
    assert approval["proposed_change"] == {
        "item_id": "action_item:5",
        "resolution_note": "Done",
        "ticker": "MU",
        "description": "Review MU thesis",
        "action_type": "review",
        "urgency": "normal",
    }


def test_approve_rejects_stale_ontology_base_state(monkeypatch):
    import portfolio.action_registry as action_registry

    current_hash = {"value": "old"}
    monkeypatch.setattr(
        action_registry,
        "compute_action_base_state_hash",
        lambda _action_id, _payload: current_hash["value"],
    )
    service = OntologyCommandService(FakeObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "complete_action_item",
        {"item_id": 1, "resolution_note": "Done"},
        context,
        reason="Complete item",
    )
    current_hash["value"] = "new"

    with pytest.raises(OntologyCommandConflict, match="base state changed"):
        service.resolve_approval(approval["id"], "approved", "Apply", context)

    assert service.get_approval(approval["id"], actor=context.actor)["status"] == "pending"


def test_apply_failure_keeps_ontology_approval_retryable(monkeypatch):
    service = OntologyCommandService(FakeObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "create_action_item",
        {"ticker": "MU", "description": "Review MU thesis", "action_type": "review"},
        context,
        reason="Create action item",
    )

    def fail_apply(*args, **kwargs):
        raise RuntimeError("cannot apply")

    monkeypatch.setattr(service, "_write_action_targets", fail_apply)

    with pytest.raises(OntologyCommandConflict, match="cannot apply"):
        service.resolve_approval(approval["id"], "approved", "Apply", context)

    failed = service.get_approval(approval["id"], actor=context.actor)
    assert failed["status"] == "pending"
    assert failed["resolution_state"] == "pending"
    assert failed["application_status"] == "failed"
    assert failed["application_state"] == "failed"
    assert failed["application_attempts"] == 1
    assert failed["application_error"] == "cannot apply"


def test_audit_write_failure_does_not_break_ontology_rejection():
    class AuditFailObjectService(FakeObjectService):
        def write_object(self, object_type, business_key, properties, valid_from, **kwargs):
            if object_type == "AuditEvent":
                raise RuntimeError("audit down")
            return super().write_object(object_type, business_key, properties, valid_from, **kwargs)

    service = OntologyCommandService(AuditFailObjectService())  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "create_action_item",
        {"ticker": "MU", "description": "Review MU thesis", "action_type": "review"},
        context,
        reason="Create action item",
    )
    rejected = service.resolve_approval(approval["id"], "rejected", "Skip", context)

    assert rejected["status"] == "rejected"
    assert rejected["application_status"] == "not_applicable"


def test_save_thesis_content_writes_native_markdown_entities_and_relations(monkeypatch, tmp_path):
    import portfolio.thesis_content as thesis_content

    indexed: list[dict[str, Any]] = []
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_content, "THESES_DIR", thesis_dir)
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: indexed.append(kwargs))

    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")
    content = """# META

## Thesis
- AI ad tools improve monetization.

## Key Catalysts
- **AI capex ramp:** Llama and ad-ranking investments convert into revenue growth.

## Risk Factors
- **Ad deceleration:** Reels and AI ad load stop improving.

## Thesis Claims
- **AI capex remains durable:** Infrastructure spend creates monetization leverage.
  - Status: active
  - Catalysts: AI capex ramp
  - Kill conditions: Ad deceleration
"""

    approval = service.propose_action(
        "save_thesis_content",
        {"ticker": "meta", "content": content},
        context,
        reason="unit",
    )

    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    assert applied["application_status"] == "applied"
    assert (thesis_dir / "META.md").read_text(encoding="utf-8").endswith("\n")
    object_types = {row["object_type"] for row in repo.objects.values()}
    assert {"ThesisDocument", "ThesisSection", "Catalyst", "KillCondition", "ThesisClaim"} <= object_types
    relation_types = {row["relation_type"] for row in repo.relations}
    assert "has_catalyst" in relation_types
    assert "thesis_has_kill_condition" in relation_types
    assert "thesis_has_claim" in relation_types
    assert "claim_links_catalyst" in relation_types
    assert "claim_links_kill_condition" in relation_types
    assert indexed and indexed[0]["doc_type"] == "thesis"


def test_save_management_quality_content_writes_ontology_children_and_markdown(monkeypatch, tmp_path):
    import portfolio.management_quality_content as management_quality_content

    indexed: list[dict[str, Any]] = []
    mgmt_dir = tmp_path / "investment_management_quality"
    mgmt_dir.mkdir()
    monkeypatch.setattr(management_quality_content, "MANAGEMENT_QUALITY_DIR", mgmt_dir)
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: indexed.append(kwargs))

    fake = FakeObjectService()
    service = OntologyCommandService(fake)  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")

    approval = service.propose_action(
        "save_management_quality_content",
        {
            "ticker": "mu",
            "content": """# MU Management Quality

## Executive Summary
- **Overall Rating**: Strong
- **Bottom Line**: Good operator.
- **Owner Mindset**: Strong - Disciplined capital allocation.
- **Business Value Understanding**: Mixed - Some gaps.
- **Follow-through / Character**: Strong - Targets met.

## Management Scorecard
| Question | Rating | Evidence |
|----------|--------|----------|
| Do managers think and act like owners? | Strong | Buybacks were disciplined. |

## Most Impressive Accomplishments
- **HBM ramp (2025)**: Executed well.

## Biggest Setbacks and Responses
- **Inventory cycle (2023)**: Downturn. **Response**: Mixed - Costs were reset.
""",
        },
        context,
        reason="unit",
    )

    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    assert applied["application_status"] == "applied"
    assert (mgmt_dir / "MU.md").read_text(encoding="utf-8").endswith("\n")
    object_types = {row["object_type"] for row in fake.objects.values()}
    assert "ManagementQualityAssessment" in object_types
    assert "ManagementQualityScorecardRow" in object_types
    assert "ManagementQualityAccomplishment" in object_types
    assert "ManagementQualitySetback" in object_types
    assert any(rel["relation_type"] == "management_quality_assesses_issuer" for rel in fake.relations)
    assert any(rel["relation_type"] == "research_object_uses_document" for rel in fake.relations)
    assert indexed and indexed[0]["doc_type"] == "management_quality"


def test_save_overview_content_writes_typed_research_objects_and_chunk_lineage(monkeypatch, tmp_path):
    import portfolio.overview_content as overview_content

    indexed: list[dict[str, Any]] = []
    overview_dir = tmp_path / "investment_overviews"
    overview_dir.mkdir()
    monkeypatch.setattr(overview_content, "OVERVIEWS_DIR", overview_dir)
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: indexed.append(kwargs))

    repo = NormalizingTemporalRepo()
    service = OntologyCommandService(OntologyObjectService(repository=repo))  # type: ignore[arg-type]
    context = OntologyCommandContext(actor=admin_actor(source="test"), source_type="test", source_id="unit")
    content = """# MU Overview

## Financials
- **3-Year Avg. YoY Revenue Growth**: +12% driven by memory recovery.
- **3-Year Avg. YoY EPS Growth**: +9% through cycle.
- **Debt**: manageable ladder.
| Tranche | Rate | Maturity |
|---------|------|----------|
| 2030 notes | 5.0% | 2030 |
- **Reinvestment Costs**: elevated HBM capex.

## Sensitivity to Extrinsic Factors
| Factor | Sensitivity | Capacity |
|--------|-------------|----------|
| Memory pricing | High | Medium |

## Porter's Five Forces
- **Supplier Power - Medium**: Equipment suppliers remain important.

## Supply Outlook
- **HBM capacity**: Supply remains constrained.

## Demand Outlook
- **AI servers**: Strong demand is visible.

### Supply Chain

#### Key Suppliers
| Entity | Relationship | Exposure | Notes |
|--------|--------------|----------|-------|
| ASML | Lithography equipment | Material capex supplier | EUV tools. |

#### Key Customers
| Entity | Relationship | Exposure | Notes |
|--------|--------------|----------|-------|
| Nvidia | HBM customer | Significant | AI accelerator demand. |
"""

    approval = service.propose_action(
        "save_overview_content", {"ticker": "mu", "content": content}, context, reason="unit"
    )
    applied = service.resolve_approval(approval["id"], "approved", "apply", context)

    object_types = {row["object_type"] for row in repo.objects.values()}
    relation_types = {row["relation_type"] for row in repo.relations}
    assert applied["application_status"] == "applied"
    assert (overview_dir / "MU.md").exists()
    assert {
        "DocumentArtifact",
        "EquityOverview",
        "CompanyFinancialProfile",
        "ExtrinsicSensitivity",
        "IndustryForceAssessment",
        "SupplyDemandOutlook",
        "SupplyChainRelationship",
    } <= object_types
    assert {
        "document_artifact_materializes_research_object",
        "equity_overview_covers_issuer",
        "equity_overview_covers_instrument",
        "equity_overview_has_financial_profile",
        "equity_overview_has_extrinsic_sensitivity",
        "equity_overview_has_industry_force",
        "equity_overview_has_supply_demand_outlook",
        "equity_overview_has_supply_chain_relationship",
    } <= relation_types
    assert indexed
    assert indexed[0]["doc_type"] == "overview"
    assert indexed[0]["object_uid"].startswith("equity_overview:")
    assert indexed[0]["object_version_id"]
