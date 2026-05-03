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
