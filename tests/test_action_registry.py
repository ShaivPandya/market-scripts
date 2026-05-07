from __future__ import annotations

import json
import os

import pytest

import portfolio.core_db as core_db
import portfolio.news_digests as digests
import portfolio.portfolio_db as portfolio_db
import portfolio.thesis_db as thesis_db
import portfolio.valuation as valuation
from portfolio.action_registry import (
    ActionAuthorizationError,
    ActionContext,
    ActionValidationError,
    CreateActionItemInput,
    DomainAction,
    execute_action,
    propose_action,
    register_action_schema_version,
    register_action_upgrade_adapter,
)


def _approve_action(action_id: str, payload: dict, *, reason: str = "test approval") -> dict:
    approval = propose_action(
        action_id,
        payload,
        ActionContext(actor_type="user", source_type="user", source_id=f"test:{action_id}"),
        reason=reason,
    )
    core_db.resolve_approval(approval["id"], "approved", "Approved in test")
    runs = core_db.get_action_runs(action_id, approval_id=approval["id"])
    assert runs, f"missing child action run for {action_id}"
    return json.loads(runs[-1]["output_json"])


@pytest.fixture(autouse=True)
def _temp_action_state(tmp_path, monkeypatch):
    for module, name in (
        (core_db, "core.db"),
        (portfolio_db, "portfolio.db"),
        (thesis_db, "thesis.db"),
    ):
        conn = getattr(module, "_conn", None)
        if conn:
            conn.close()
        monkeypatch.setattr(module, "DB_PATH", tmp_path / name)
        monkeypatch.setattr(module, "_conn", None)

    import api.cache as cache
    import portfolio.portfolio_dashboard as dashboard

    monkeypatch.setattr(cache, "invalidate_all", lambda: None)
    monkeypatch.setattr(dashboard, "reload_portfolio", lambda: None)
    monkeypatch.setattr(
        valuation,
        "_cached_yfinance_metadata",
        lambda symbol: {"currency": "USD", "country": "United States", "exchange": None},
    )
    monkeypatch.setattr(
        valuation,
        "fx_rate_to_base",
        lambda currency, base_currency="USD": {"rate": 1.0, "as_of": "2026-05-05"},
    )
    base = tmp_path / "news_digests"
    monkeypatch.setattr(digests, "DIGESTS_DIR", base)
    monkeypatch.setattr(digests, "MANIFEST_PATH", base / "manifest.json")
    monkeypatch.setattr(digests, "FILES_DIR", base / "files")
    monkeypatch.setattr(digests, "DIGESTS_GCS_PREFIX", "test/news_digests")
    monkeypatch.setattr(digests, "MANIFEST_GCS_KEY", "test/news_digests/manifest.json")
    monkeypatch.setattr(digests, "FILES_GCS_PREFIX", "test/news_digests/files")
    os.environ["STATE_STORAGE_BACKEND"] = "local"
    yield
    for module in (core_db, portfolio_db, thesis_db):
        conn = getattr(module, "_conn", None)
        if conn:
            conn.close()
        monkeypatch.setattr(module, "_conn", None)


def test_user_direct_portfolio_action_is_denied_and_approval_apply_writes_positions():
    payload = {
        "positions": [
            {
                "ticker": "mu",
                "asset": "equity",
                "direction": "long",
                "contrarian": False,
                "conviction": 4,
                "cost_basis": 100,
                "shares": 12,
            }
        ]
    }
    with pytest.raises(ActionAuthorizationError):
        execute_action(
            "update_portfolio_positions",
            payload,
            ActionContext(actor_type="user", source_type="api", source_id="test"),
        )
    assert portfolio_db.get_positions() == []

    result = _approve_action("update_portfolio_positions", payload)

    assert result == {"status": "ok", "count": 1}
    assert portfolio_db.get_positions() == [
        {
            "ticker": "MU",
            "asset": "equity",
            "direction": "long",
            "contrarian": 0,
            "conviction": 4,
            "cost_basis": 100.0,
            "shares": 12.0,
            "quantity": 12.0,
            "instrument_type": "security",
            "price_symbol": "MU",
            "contract_multiplier": 1.0,
            "fx_base_currency": None,
            "fx_quote_currency": None,
            "currency": "USD",
            "country": "United States",
            "exchange": None,
            "base_currency": "USD",
            "fx_rate_to_base": 1.0,
            "fx_rate_as_of": "2026-05-05",
            "cost_basis_base": 100.0,
            "notional_base": 1200.0,
            "valuation_status": "ok",
            "role": "position",
        }
    ]
    runs = core_db.get_action_runs("update_portfolio_positions")
    assert len(runs) == 2
    assert runs[-1]["status"] == "succeeded"
    assert runs[-1]["approval_id"] is not None
    events = [event["event_type"] for event in core_db.get_action_events(runs[-1]["id"])]
    assert "validated" in events
    assert "mutation_completed" in events
    assert "callback_completed" in events
    audit = core_db.get_audit_events(action_category="domain_action", limit=10)
    assert [row["action_name"] for row in audit].count("domain.action.started") >= 1
    succeeded = [
        row
        for row in audit
        if row["action_name"] == "domain.action.succeeded"
        and row["object_refs"]
        and row["object_refs"][0] == {"type": "domain_action", "id": "update_portfolio_positions"}
    ][0]
    assert succeeded["object_refs"][0] == {"type": "domain_action", "id": "update_portfolio_positions"}
    assert succeeded["after_summary"]["status"] == "ok"


def test_portfolio_action_accepts_continuous_future_positions():
    result = _approve_action(
        "update_portfolio_positions",
        {
            "positions": [
                {
                    "ticker": "CL=F",
                    "instrument_type": "future",
                    "direction": "short",
                    "contrarian": False,
                    "conviction": 3,
                    "cost_basis": 75,
                    "quantity": 1,
                }
            ]
        },
    )

    assert result == {"status": "ok", "count": 1}
    position = portfolio_db.get_positions()[0]
    assert position["ticker"] == "CL=F"
    assert position["instrument_type"] == "future"
    assert position["asset"] == "commodity"
    assert position["quantity"] == 1.0
    assert position["contract_multiplier"] == 1000.0


def test_portfolio_action_accepts_spot_fx_positions():
    result = _approve_action(
        "update_portfolio_positions",
        {
            "positions": [
                {
                    "ticker": "EUR-USD",
                    "instrument_type": "spot_fx",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 3,
                    "cost_basis": 1.08,
                    "quantity": 50_000,
                }
            ]
        },
    )

    assert result == {"status": "ok", "count": 1}
    position = portfolio_db.get_positions()[0]
    assert position["ticker"] == "EURUSD=X"
    assert position["instrument_type"] == "spot_fx"
    assert position["asset"] == "fx"
    assert position["fx_base_currency"] == "EUR"
    assert position["fx_quote_currency"] == "USD"
    assert position["notional_base"] == pytest.approx(54_000.0)


def test_portfolio_update_approval_payload_lists_position_changes():
    portfolio_db.save_positions(
        [
            {
                "ticker": "MU",
                "asset": "equity",
                "direction": "long",
                "contrarian": False,
                "conviction": 3,
                "cost_basis": 100,
                "shares": 10,
            },
            {
                "ticker": "NVDA",
                "asset": "equity",
                "direction": "long",
                "contrarian": False,
                "conviction": 4,
                "cost_basis": 200,
                "shares": 2,
            },
        ],
        role="position",
    )

    approval = propose_action(
        "update_portfolio_positions",
        {
            "positions": [
                {
                    "ticker": "MU",
                    "asset": "equity",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 3,
                    "cost_basis": 100,
                    "shares": 15,
                },
                {
                    "ticker": "CRWD",
                    "asset": "equity",
                    "direction": "short",
                    "contrarian": True,
                    "conviction": 2,
                    "cost_basis": 300,
                    "shares": 3,
                },
            ]
        },
        ActionContext(actor_type="user", source_type="user", source_id="test"),
        reason="rebalance test",
    )

    change = approval["proposed_change"]
    assert [item["ticker"] for item in change["position_changes"]] == ["MU", "CRWD", "NVDA"]
    assert change["position_change_summary"] == {"before_count": 2, "after_count": 2}

    mu_change = change["position_changes"][0]
    assert mu_change["change_type"] == "updated"
    assert mu_change["fields"] == [{"field": "quantity", "before": 10.0, "after": 15.0}]

    crwd_change = change["position_changes"][1]
    assert crwd_change["change_type"] == "added"
    assert crwd_change["before"] is None
    assert crwd_change["after"]["contrarian"] is True

    nvda_change = change["position_changes"][2]
    assert nvda_change["change_type"] == "removed"
    assert nvda_change["before"]["ticker"] == "NVDA"
    assert nvda_change["after"] is None


def test_execute_action_shadow_writes_ontology_versions(monkeypatch):
    captured: dict[str, list[dict]] = {"objects": [], "relations": []}

    class FakeObjectService:
        def write_object(self, object_type, business_key, properties, valid_from, **kwargs):
            version_id = f"version-{len(captured['objects']) + 1}"
            object_uid = f"{object_type}:{business_key}"
            row = {
                "version_id": version_id,
                "object_uid": object_uid,
                "object_type": object_type,
                "business_key": business_key,
                "properties": dict(properties),
                "_meta": {"temporal": {"object_uid": object_uid, "version_id": version_id}},
            }
            captured["objects"].append({**row, "kwargs": kwargs, "valid_from": valid_from})
            return row

        def write_relation(self, source_uid, target_uid, relation_type, properties, valid_from, **kwargs):
            captured["relations"].append(
                {
                    "source_uid": source_uid,
                    "target_uid": target_uid,
                    "relation_type": relation_type,
                    "properties": dict(properties),
                    "valid_from": valid_from,
                    "kwargs": kwargs,
                }
            )
            return {
                "relation_uid": f"{relation_type}:{source_uid}->{target_uid}",
                "_meta": {"temporal": {"version_id": f"relation-{len(captured['relations'])}"}},
            }

    import ontology.domain_write_service as domain_write_service

    monkeypatch.setenv("ONTOLOGY_SHADOW_WRITES", "true")
    monkeypatch.setattr(domain_write_service, "OntologyObjectService", FakeObjectService)

    result = _approve_action(
        "create_action_item",
        {"description": "Review MU", "action_type": "review", "ticker": "MU", "urgency": "high"},
    )

    assert result["description"] == "Review MU"
    object_types = [row["object_type"] for row in captured["objects"]]
    assert "Approval" in object_types
    assert "ActionRun" in object_types
    assert "ActionItem" in object_types
    action_item = next(row for row in captured["objects"] if row["object_type"] == "ActionItem")
    assert action_item["properties"]["ticker"] == "MU"
    assert action_item["kwargs"]["action_run_id"] is not None
    assert action_item["kwargs"]["input_hash"]
    assert captured["relations"][0]["relation_type"] == "action_run_mutates_object_version"
    action_run = core_db.get_action_runs("create_action_item")[0]
    events = [event["event_type"] for event in core_db.get_action_events(action_run["id"])]
    assert "ontology_versions_written" in events


def test_propose_action_shadow_writes_pending_approval(monkeypatch):
    captured: list[dict] = []

    class FakeObjectService:
        def write_object(self, object_type, business_key, properties, valid_from, **kwargs):
            captured.append(
                {
                    "object_type": object_type,
                    "business_key": business_key,
                    "properties": dict(properties),
                    "valid_from": valid_from,
                    "kwargs": kwargs,
                    "_meta": {"temporal": {"version_id": "approval-version"}},
                }
            )
            return captured[-1]

    import ontology.domain_write_service as domain_write_service

    monkeypatch.setenv("ONTOLOGY_SHADOW_WRITES", "true")
    monkeypatch.setattr(domain_write_service, "OntologyObjectService", FakeObjectService)

    approval = propose_action(
        "create_action_item",
        {"description": "Review CRWD", "action_type": "review", "ticker": "CRWD"},
        ActionContext(actor_type="agent", source_type="agent", source_id="agent-1"),
        reason="needs review",
    )

    assert approval["entity_type"] == "action_item"
    assert captured[0]["object_type"] == "Approval"
    assert captured[0]["properties"]["action_id"] == "create_action_item"
    assert captured[0]["properties"]["ticker"] == "CRWD"
    events = [event["event_type"] for event in core_db.get_action_events(1)]
    assert "ontology_approval_version_written" in events


def test_execute_action_denies_agent_direct_mutation_and_audits_failure():
    with pytest.raises(ActionAuthorizationError):
        execute_action(
            "update_portfolio_positions",
            {
                "positions": [
                    {"ticker": "MU", "asset": "equity", "direction": "long", "contrarian": False, "conviction": 3}
                ]
            },
            ActionContext(actor_type="agent", source_type="agent"),
        )

    assert portfolio_db.get_positions() == []
    runs = core_db.get_action_runs("update_portfolio_positions")
    assert runs[0]["status"] == "failed"
    assert "authorization_denied" in [event["event_type"] for event in core_db.get_action_events(runs[0]["id"])]
    denied = core_db.get_audit_events(action_name="domain.action.denied")
    assert denied[0]["status"] == "denied"
    assert denied[0]["object_refs"][0]["id"] == "update_portfolio_positions"


def test_execute_portfolio_action_validation_failure_is_audited():
    with pytest.raises(ActionValidationError, match="At least one position"):
        execute_action(
            "update_portfolio_positions",
            {"positions": []},
            ActionContext(actor_type="user", source_type="api", source_id="test"),
        )

    runs = core_db.get_action_runs("update_portfolio_positions")
    assert runs[0]["status"] == "failed"
    assert "At least one position" in runs[0]["error"]


def test_action_backed_approval_applies_registered_action():
    approval = propose_action(
        "update_hedge_positions",
        {"positions": [{"ticker": "SPY", "direction": "short", "shares": 5}]},
        ActionContext(actor_type="workflow", source_type="workflow", source_id="run-1"),
        reason="Workflow hedge proposal",
    )

    assert approval["entity_type"] == "hedge_positions"
    assert approval["action_id"] == "update_hedge_positions"

    resolved = core_db.resolve_approval(approval["id"], "approved", "Approved in test")

    assert resolved["status"] == "approved"
    assert portfolio_db.get_hedge_positions()[0]["ticker"] == "SPY"
    child_runs = core_db.get_action_runs("update_hedge_positions", approval_id=approval["id"])
    assert len(child_runs) == 1
    assert child_runs[0]["status"] == "succeeded"
    assert child_runs[0]["parent_action_run_id"] is not None
    audit_names = {event["action_name"] for event in core_db.get_audit_events(limit=50)}
    assert "approval.created" in audit_names
    assert "approval.apply.started" in audit_names
    assert "approval.applied" in audit_names


def test_recommendation_approval_normalizes_apply_required_fields():
    result = _approve_action(
        "create_recommendation",
        {
            "record": {
                "report_type": "daily",
                "as_of": "2026-05-06",
                "action": "rebalance",
                "instrument": "hedge_overlay",
                "horizon": "1 trading day",
                "rationale": "Rebalance hedge overlay.",
                "confidence": 0.65,
                "critical_data_quality": "ok",
                "idempotency_key": "daily:2026-05-06:hedge-overlay",
            }
        },
    )

    assert result["stance"] == "Neutral / Watchful"
    assert core_db.get_recommendations(report_type="daily")[0]["stance"] == "Neutral / Watchful"


def test_v1_portfolio_approval_upgrades_and_applies_after_schema_bump():
    approval = core_db.create_pending_approval(
        entity_type="portfolio_positions",
        proposed_change={
            "positions": [
                {
                    "ticker": "MU",
                    "asset": "equity",
                    "direction": "long",
                    "contrarian": False,
                    "conviction": 3,
                    "cost_basis": 100,
                    "shares": 5,
                }
            ]
        },
        reason="legacy approval",
        source_type="workflow",
        source_id="legacy-run",
        action_id="update_portfolio_positions",
        action_schema_name="update_portfolio_positions",
        action_schema_version=1,
    )

    resolved = core_db.resolve_approval(approval["id"], "approved", "Approved in test")

    assert resolved["application_status"] == "applied"
    position = portfolio_db.get_positions()[0]
    assert position["ticker"] == "MU"
    assert position["quantity"] == 5.0
    assert position["instrument_type"] == "security"
    assert position["contract_multiplier"] == 1.0
    child_run = core_db.get_action_runs("update_portfolio_positions", approval_id=approval["id"])[0]
    assert child_run["action_schema_version"] == 1


def test_thesis_status_action_noops_same_status_without_history_row():
    thesis_db.upsert_thesis_meta("MU", status="active")
    assert len(thesis_db.get_status_history("MU")) == 1

    result = _approve_action(
        "change_thesis_status",
        {"ticker": "MU", "status": "active", "reason": "No change"},
    )

    assert result["changed"] is False
    assert len(thesis_db.get_status_history("MU")) == 1


def test_process_actions_write_entities_and_run_markdown_callbacks(monkeypatch):
    synced: list[str] = []

    monkeypatch.setattr(
        "portfolio.thesis_sync.sync_markdown_from_entities",
        lambda ticker: synced.append(ticker),
    )

    catalyst = _approve_action(
        "create_catalyst",
        {"ticker": "mu", "description": "HBM ramp", "category": "fundamental"},
    )
    updated_catalyst = _approve_action(
        "update_catalyst_status",
        {"catalyst_id": catalyst["id"], "status": "played_out", "evidence": "Confirmed"},
    )
    kill_condition = _approve_action(
        "create_kill_condition",
        {"ticker": "mu", "condition": "Demand rolls", "metric": "orders"},
    )
    updated_kill_condition = _approve_action(
        "update_kill_condition_status",
        {"kill_condition_id": kill_condition["id"], "status": "triggered"},
    )

    assert updated_catalyst["status"] == "played_out"
    assert updated_kill_condition["status"] == "triggered"
    assert core_db.get_catalysts("MU")[0]["description"] == "HBM ramp"
    assert core_db.get_kill_conditions("MU")[0]["condition"] == "Demand rolls"
    assert synced == ["MU", "MU", "MU", "MU"]


def test_action_item_and_watch_trigger_actions_cover_lifecycle():
    action = _approve_action(
        "create_action_item",
        {"description": "Review MU", "action_type": "review", "ticker": "mu", "urgency": "high"},
    )
    completed = _approve_action(
        "complete_action_item",
        {"item_id": action["id"], "resolution_note": "Done"},
    )
    dismiss_source = _approve_action("create_action_item", {"description": "Dismiss me"})
    dismissed = _approve_action(
        "dismiss_action_item",
        {"item_id": dismiss_source["id"]},
    )

    trigger = _approve_action(
        "create_watch_trigger",
        {"condition": "MU > 150", "trigger_type": "price_level", "ticker": "mu"},
    )
    fired = _approve_action(
        "fire_watch_trigger",
        {"trigger_id": trigger["id"], "result": {"price": 151}, "evidence": "Price crossed trigger"},
    )
    cancel_source = _approve_action("create_watch_trigger", {"condition": "Cancel me", "trigger_type": "custom"})
    cancelled = _approve_action(
        "cancel_watch_trigger",
        {"trigger_id": cancel_source["id"]},
    )

    assert completed["status"] == "completed"
    assert completed["resolution_note"] == "Done"
    assert dismissed["status"] == "dismissed"
    assert fired["status"] == "fired"
    assert fired["last_result"] == {"price": 151}
    assert cancelled["status"] == "cancelled"

    run = core_db.get_action_runs("create_action_item")[0]
    trace = core_db.get_provenance_trace(action_run_id=run["id"])
    assert any(
        link["source_ref_type"] == "action_run"
        and link["source_ref_id"] == str(run["id"])
        and link["target_ref_type"] == "action_item"
        and link["target_ref_id"] == str(action["id"])
        and link["link_type"] == "produced"
        for link in trace["links"]
    )
    assert any(link["target_ref_type"] == "audit_event" for link in trace["links"])


def test_legacy_approval_backed_actions_replay_through_registry(monkeypatch):
    monkeypatch.setattr("api.routers.portfolio_news._delete_digest_index_best_effort", lambda _digest_id: None)

    digest = digests.save_digest("# Digest\n\n## Movers\n- MU update\n", filename="05012026_digest.md")
    context = ActionContext(actor_type="workflow", source_type="workflow", source_id="run-legacy-actions")

    evaluation_approval = propose_action(
        "save_evaluation",
        {
            "ticker": "MU",
            "thesis_status": "active",
            "technical_read": "Constructive",
            "fundamental_read": "Stable",
            "action": "hold",
            "confidence": "high",
            "key_developments": ["HBM demand remains strong"],
        },
        context,
        reason="workflow evaluation",
    )
    delete_approval = propose_action(
        "delete_portfolio_news_digest",
        {"digest_id": digest["id"]},
        context,
        reason="remove stale digest",
    )

    core_db.resolve_approval(evaluation_approval["id"], "approved", "Approved in test")
    core_db.resolve_approval(delete_approval["id"], "approved", "Approved in test")

    assert thesis_db.get_evaluations("MU", limit=1)[0]["thesis_status"] == "active"
    with pytest.raises(FileNotFoundError):
        digests.get_digest(digest["id"])
    assert core_db.get_action_runs("save_evaluation", approval_id=evaluation_approval["id"])[0]["status"] == "succeeded"
    assert (
        core_db.get_action_runs("delete_portfolio_news_digest", approval_id=delete_approval["id"])[0]["status"]
        == "succeeded"
    )


def test_pending_approval_replays_old_action_schema_after_current_schema_evolves(monkeypatch):
    from typing import Literal

    import portfolio.action_registry as registry

    approval = propose_action(
        "create_action_item",
        {"description": "Review MU", "action_type": "review", "ticker": "mu", "urgency": "high"},
        ActionContext(actor_type="workflow", source_type="workflow", source_id="run-old-schema"),
        reason="old schema test",
    )

    class CreateActionItemInputV2(CreateActionItemInput):
        schema_version: Literal[2] = 2
        source: Literal["approval"] = "approval"

    old_action = registry.get_action("create_action_item")
    monkeypatch.setitem(
        registry._ACTIONS,
        "create_action_item",
        DomainAction(
            action_id=old_action.action_id,
            input_model=CreateActionItemInputV2,
            handler=old_action.handler,
            schema_version=2,
            execute_actor_types=old_action.execute_actor_types,
            propose_actor_types=old_action.propose_actor_types,
            approval_entity_type=old_action.approval_entity_type,
            approval_payload=old_action.approval_payload,
            approval_ticker=old_action.approval_ticker,
        ),
    )
    register_action_schema_version("create_action_item", 1, CreateActionItemInput)
    register_action_schema_version("create_action_item", 2, CreateActionItemInputV2)
    register_action_upgrade_adapter(
        "create_action_item",
        1,
        2,
        lambda payload: {**payload, "source": "approval"},
    )

    resolved = core_db.resolve_approval(approval["id"], "approved", "Approved in test")

    assert resolved["application_status"] == "applied"
    action = core_db.get_action_items()[0]
    assert action["description"] == "Review MU"
    child_run = core_db.get_action_runs("create_action_item", approval_id=approval["id"])[0]
    assert child_run["action_schema_version"] == 1


def test_thesis_claim_actions_normalize_sources_and_not_found_errors(monkeypatch):
    monkeypatch.setattr("portfolio.thesis_sync.sync_markdown_from_entities", lambda _ticker: None)
    created = _approve_action(
        "create_thesis_claim",
        {
            "ticker": "mu",
            "claim": "HBM stays tight",
            "source_requirements": ["earnings"],
            "confidence": 0.6,
        },
    )

    updated = _approve_action(
        "update_thesis_claim",
        {"claim_id": created["id"], "status": "supported", "confidence": 0.8},
    )

    assert updated["status"] == "supported"
    assert updated["source_requirements"][0]["description"] == "earnings"
    with pytest.raises(ActionValidationError, match="confidence"):
        execute_action(
            "create_thesis_claim",
            {"ticker": "MU", "claim": "Bad", "confidence": 2},
            ActionContext(actor_type="user", source_type="api", source_id="test"),
        )


def test_save_thesis_content_action_writes_meta_and_runs_callbacks(monkeypatch, tmp_path):
    import portfolio.thesis_content as thesis_content

    indexed: list[tuple[str, str]] = []
    synced: list[str] = []
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_content, "THESES_DIR", thesis_dir)
    monkeypatch.setattr(
        "api.retrieval.index_document",
        lambda **kwargs: indexed.append((kwargs["ticker"], kwargs["content"])),
    )
    monkeypatch.setattr(
        "portfolio.thesis_sync.sync_entities_from_markdown",
        lambda ticker: synced.append(ticker),
    )

    result = _approve_action(
        "save_thesis_content",
        {"ticker": "mu", "content": "# MU\n\n## Thesis\n- Good"},
    )

    assert result == {"status": "ok", "ticker": "MU", "content": "# MU\n\n## Thesis\n- Good"}
    assert (thesis_dir / "MU.md").read_text(encoding="utf-8").endswith("\n")
    assert thesis_db.get_thesis_meta("MU")["status"] == "active"
    assert indexed == [("MU", "# MU\n\n## Thesis\n- Good")]
    assert synced == ["MU"]


def test_save_management_quality_content_action_writes_and_indexes(monkeypatch, tmp_path):
    import portfolio.management_quality_content as management_quality_content

    indexed: list[dict] = []
    mgmt_dir = tmp_path / "investment_management_quality"
    mgmt_dir.mkdir()
    monkeypatch.setattr(management_quality_content, "MANAGEMENT_QUALITY_DIR", mgmt_dir)
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: indexed.append(kwargs))

    result = _approve_action(
        "save_management_quality_content",
        {"ticker": "mu", "content": "# MU Management Quality\n\n## Executive Summary\n- **Overall Rating**: Strong"},
    )

    assert result["status"] == "ok"
    assert result["ticker"] == "MU"
    assert (mgmt_dir / "MU.md").read_text(encoding="utf-8").endswith("\n")
    assert indexed[0]["doc_type"] == "management_quality"
    assert indexed[0]["ticker"] == "MU"
    assert indexed[0]["doc_id"] == "management_quality-MU"


def test_resolve_approval_action_applies_action_backed_approval_without_duplicate_top_level_run():
    approval = propose_action(
        "create_action_item",
        {"description": "Review MU", "ticker": "MU"},
        ActionContext(actor_type="workflow", source_type="workflow", source_id="run-1"),
        reason="Workflow generated",
    )

    resolved = execute_action(
        "resolve_approval",
        {"approval_id": approval["id"], "status": "approved", "note": "Apply"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output

    assert resolved["status"] == "approved"
    assert resolved["application_status"] == "applied"
    assert core_db.get_action_items(ticker="MU")[0]["description"] == "Review MU"
    resolve_runs = core_db.get_action_runs("resolve_approval")
    child_runs = core_db.get_action_runs("create_action_item", approval_id=approval["id"])
    assert len(resolve_runs) == 1
    assert len(child_runs) == 1
    assert child_runs[0]["parent_action_run_id"] == resolve_runs[0]["id"]
