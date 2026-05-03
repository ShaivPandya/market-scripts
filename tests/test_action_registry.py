from __future__ import annotations

import pytest

import portfolio.core_db as core_db
import portfolio.portfolio_db as portfolio_db
import portfolio.thesis_db as thesis_db
from portfolio.action_registry import (
    ActionAuthorizationError,
    ActionContext,
    ActionValidationError,
    execute_action,
    propose_action,
)


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
    yield
    for module in (core_db, portfolio_db, thesis_db):
        conn = getattr(module, "_conn", None)
        if conn:
            conn.close()
        monkeypatch.setattr(module, "_conn", None)


def test_execute_portfolio_action_writes_positions_and_audit_events():
    result = execute_action(
        "update_portfolio_positions",
        {
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
        },
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    )

    assert result.output == {"status": "ok", "count": 1}
    assert portfolio_db.get_positions() == [
        {
            "ticker": "MU",
            "asset": "equity",
            "direction": "long",
            "contrarian": 0,
            "conviction": 4,
            "cost_basis": 100.0,
            "shares": 12.0,
            "role": "position",
        }
    ]
    runs = core_db.get_action_runs("update_portfolio_positions")
    assert len(runs) == 1
    assert runs[0]["status"] == "succeeded"
    events = [event["event_type"] for event in core_db.get_action_events(runs[0]["id"])]
    assert "validated" in events
    assert "mutation_completed" in events
    assert "callback_completed" in events
    audit = core_db.get_audit_events(action_category="domain_action", limit=10)
    assert [row["action_name"] for row in audit].count("domain.action.started") == 1
    succeeded = [row for row in audit if row["action_name"] == "domain.action.succeeded"][0]
    assert succeeded["object_refs"][0] == {"type": "domain_action", "id": "update_portfolio_positions"}
    assert succeeded["after_summary"]["status"] == "ok"


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

    resolved = core_db.resolve_approval(approval["id"], "approved")

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


def test_thesis_status_action_noops_same_status_without_history_row():
    thesis_db.upsert_thesis_meta("MU", status="active")
    assert len(thesis_db.get_status_history("MU")) == 1

    result = execute_action(
        "change_thesis_status",
        {"ticker": "MU", "status": "active", "reason": "No change"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    )

    assert result.output["changed"] is False
    assert len(thesis_db.get_status_history("MU")) == 1


def test_process_actions_write_entities_and_run_markdown_callbacks(monkeypatch):
    synced: list[str] = []

    monkeypatch.setattr(
        "portfolio.thesis_sync.sync_markdown_from_entities",
        lambda ticker: synced.append(ticker),
    )

    catalyst = execute_action(
        "create_catalyst",
        {"ticker": "mu", "description": "HBM ramp", "category": "fundamental"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output
    updated_catalyst = execute_action(
        "update_catalyst_status",
        {"catalyst_id": catalyst["id"], "status": "played_out", "evidence": "Confirmed"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output
    kill_condition = execute_action(
        "create_kill_condition",
        {"ticker": "mu", "condition": "Demand rolls", "metric": "orders"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output
    updated_kill_condition = execute_action(
        "update_kill_condition_status",
        {"kill_condition_id": kill_condition["id"], "status": "triggered"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output

    assert updated_catalyst["status"] == "played_out"
    assert updated_kill_condition["status"] == "triggered"
    assert core_db.get_catalysts("MU")[0]["description"] == "HBM ramp"
    assert core_db.get_kill_conditions("MU")[0]["condition"] == "Demand rolls"
    assert synced == ["MU", "MU", "MU", "MU"]


def test_action_item_and_watch_trigger_actions_cover_lifecycle():
    action = execute_action(
        "create_action_item",
        {"description": "Review MU", "action_type": "review", "ticker": "mu", "urgency": "high"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output
    completed = execute_action(
        "complete_action_item",
        {"item_id": action["id"], "resolution_note": "Done"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output
    dismissed = execute_action(
        "dismiss_action_item",
        {
            "item_id": execute_action(
                "create_action_item",
                {"description": "Dismiss me"},
                ActionContext(actor_type="user", source_type="api", source_id="test"),
            ).output["id"]
        },
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output

    trigger = execute_action(
        "create_watch_trigger",
        {"condition": "MU > 150", "trigger_type": "price_level", "ticker": "mu"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output
    fired = execute_action(
        "fire_watch_trigger",
        {"trigger_id": trigger["id"], "result": {"price": 151}, "evidence": "Breakout"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output
    cancelled = execute_action(
        "cancel_watch_trigger",
        {
            "trigger_id": execute_action(
                "create_watch_trigger",
                {"condition": "Cancel me", "trigger_type": "custom"},
                ActionContext(actor_type="user", source_type="api", source_id="test"),
            ).output["id"]
        },
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output

    assert completed["status"] == "completed"
    assert completed["resolution_note"] == "Done"
    assert dismissed["status"] == "dismissed"
    assert fired["status"] == "fired"
    assert fired["last_result"] == {"price": 151}
    assert cancelled["status"] == "cancelled"


def test_thesis_claim_actions_normalize_sources_and_not_found_errors(monkeypatch):
    monkeypatch.setattr("portfolio.thesis_sync.sync_markdown_from_entities", lambda _ticker: None)
    created = execute_action(
        "create_thesis_claim",
        {
            "ticker": "mu",
            "claim": "HBM stays tight",
            "source_requirements": ["earnings"],
            "confidence": 0.6,
        },
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output

    updated = execute_action(
        "update_thesis_claim",
        {"claim_id": created["id"], "status": "supported", "confidence": 0.8},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output

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

    result = execute_action(
        "save_thesis_content",
        {"ticker": "mu", "content": "# MU\n\n## Thesis\n- Good"},
        ActionContext(actor_type="user", source_type="api", source_id="test"),
    ).output

    assert result == {"status": "ok", "ticker": "MU", "content": "# MU\n\n## Thesis\n- Good"}
    assert (thesis_dir / "MU.md").read_text(encoding="utf-8").endswith("\n")
    assert thesis_db.get_thesis_meta("MU")["status"] == "active"
    assert indexed == [("MU", "# MU\n\n## Thesis\n- Good")]
    assert synced == ["MU"]


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
