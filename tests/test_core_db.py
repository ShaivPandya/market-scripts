"""Tests for portfolio/core_db.py -- CRUD for all 7 entity types."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

import portfolio.core_db as core_db


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Point core_db at a temporary database for every test."""
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "test_core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    yield
    # Clean up
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "_conn", None)


# ---------------------------------------------------------------------------
# Catalysts
# ---------------------------------------------------------------------------


class TestCatalysts:
    def test_create_and_get(self):
        cat = core_db.create_catalyst("MU", "HBM3 ramp", "fundamental")
        assert cat["ticker"] == "MU"
        assert cat["description"] == "HBM3 ramp"
        assert cat["status"] == "pending"
        assert cat["category"] == "fundamental"
        assert cat["id"] is not None

        cats = core_db.get_catalysts("MU")
        assert len(cats) == 1
        assert cats[0]["description"] == "HBM3 ramp"

    def test_ticker_uppercased(self):
        cat = core_db.create_catalyst("mu", "test")
        assert cat["ticker"] == "MU"
        cats = core_db.get_catalysts("mu")
        assert len(cats) == 1

    def test_update_status(self):
        cat = core_db.create_catalyst("CRWD", "ARR acceleration")
        updated = core_db.update_catalyst_status(cat["id"], "played_out", "Beat expectations")
        assert updated["status"] == "played_out"
        assert updated["evidence"] == "Beat expectations"

    def test_update_nonexistent_raises(self):
        with pytest.raises(ValueError, match="No catalyst"):
            core_db.update_catalyst_status(9999, "failed")


# ---------------------------------------------------------------------------
# Kill Conditions
# ---------------------------------------------------------------------------


class TestKillConditions:
    def test_create_and_get(self):
        kc = core_db.create_kill_condition("MU", "DRAM price drops >30%", metric="DRAM ASP", threshold="-30%")
        assert kc["ticker"] == "MU"
        assert kc["condition"] == "DRAM price drops >30%"
        assert kc["status"] == "active"

        kcs = core_db.get_kill_conditions("MU")
        assert len(kcs) == 1

    def test_trigger_sets_timestamp(self):
        kc = core_db.create_kill_condition("MU", "Revenue miss >10%")
        updated = core_db.update_kill_condition_status(kc["id"], "triggered")
        assert updated["status"] == "triggered"
        assert updated["triggered_at"] is not None

    def test_retire(self):
        kc = core_db.create_kill_condition("MU", "Old risk")
        updated = core_db.update_kill_condition_status(kc["id"], "retired")
        assert updated["status"] == "retired"


# ---------------------------------------------------------------------------
# Workflow Runs
# ---------------------------------------------------------------------------


class TestWorkflowRuns:
    def test_create_and_complete(self):
        run = core_db.create_workflow_run("morning_brief")
        assert run["status"] == "running"
        assert run["run_id"]

        completed = core_db.complete_workflow_run(
            run["run_id"],
            synthesis="Market is risk-on.",
            artifacts={"action_items": []},
            tool_sections=[{"tool": "get_signal_aggregator", "data": {}}],
        )
        assert completed["status"] == "completed"
        assert completed["synthesis"] == "Market is risk-on."
        assert completed["artifacts"] == {"action_items": []}

    def test_create_and_fail(self):
        run = core_db.create_workflow_run("thesis_review", ticker="MU")
        failed = core_db.fail_workflow_run(run["run_id"], "Tool timeout")
        assert failed["status"] == "failed"
        assert failed["error"] == "Tool timeout"

    def test_list_and_filter(self):
        core_db.create_workflow_run("morning_brief")
        core_db.create_workflow_run("thesis_review", ticker="MU")
        core_db.create_workflow_run("thesis_review", ticker="CRWD")

        all_runs = core_db.get_workflow_runs()
        assert len(all_runs) == 3

        mu_runs = core_db.get_workflow_runs(ticker="MU")
        assert len(mu_runs) == 1

        brief_runs = core_db.get_workflow_runs(workflow_name="morning_brief")
        assert len(brief_runs) == 1

    def test_get_single_run(self):
        run = core_db.create_workflow_run("morning_brief")
        fetched = core_db.get_workflow_run(run["run_id"])
        assert fetched is not None
        assert fetched["workflow_name"] == "morning_brief"

        assert core_db.get_workflow_run("nonexistent") is None


# ---------------------------------------------------------------------------
# Action Items
# ---------------------------------------------------------------------------


class TestActionItems:
    def test_create_and_list(self):
        item = core_db.create_action_item("Review MU position", "review", ticker="MU", urgency="high")
        assert item["status"] == "open"
        assert item["urgency"] == "high"

        items = core_db.get_action_items(status="open")
        assert len(items) == 1

    def test_complete(self):
        item = core_db.create_action_item("Check CRWD thesis", "review", ticker="CRWD")
        completed = core_db.complete_action_item(item["id"], "Thesis intact")
        assert completed["status"] == "completed"
        assert completed["resolution_note"] == "Thesis intact"

    def test_dismiss(self):
        item = core_db.create_action_item("Low priority task", "other")
        dismissed = core_db.dismiss_action_item(item["id"])
        assert dismissed["status"] == "dismissed"

    def test_urgency_ordering(self):
        core_db.create_action_item("Low", "other", urgency="low")
        core_db.create_action_item("Urgent", "review", urgency="urgent")
        core_db.create_action_item("Normal", "review", urgency="normal")

        items = core_db.get_action_items()
        urgencies = [i["urgency"] for i in items]
        assert urgencies[0] == "urgent"
        assert urgencies[-1] == "low"


# ---------------------------------------------------------------------------
# Watch Triggers
# ---------------------------------------------------------------------------


class TestWatchTriggers:
    def test_create_and_list(self):
        trigger = core_db.create_watch_trigger("MU breaks $140", "price_level", ticker="MU")
        assert trigger["status"] == "active"

        triggers = core_db.get_watch_triggers(status="active")
        assert len(triggers) == 1

    def test_fire(self):
        trigger = core_db.create_watch_trigger("VIX > 30", "macro")
        fired = core_db.fire_watch_trigger(trigger["id"])
        assert fired["status"] == "fired"
        assert fired["fired_at"] is not None

    def test_cancel(self):
        trigger = core_db.create_watch_trigger("Obsolete trigger", "custom")
        cancelled = core_db.cancel_watch_trigger(trigger["id"])
        assert cancelled["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Research Notes
# ---------------------------------------------------------------------------


class TestResearchNotes:
    def test_create_and_list(self):
        note = core_db.create_research_note("Earnings Summary", "MU beat by 5%", ticker="MU", note_type="earnings")
        assert note["ticker"] == "MU"
        assert note["note_type"] == "earnings"

        notes = core_db.get_research_notes(ticker="MU")
        assert len(notes) == 1

    def test_list_all(self):
        core_db.create_research_note("Note 1", "Content 1", ticker="MU")
        core_db.create_research_note("Note 2", "Content 2", ticker="CRWD")
        core_db.create_research_note("Note 3", "Content 3")

        all_notes = core_db.get_research_notes()
        assert len(all_notes) == 3


# ---------------------------------------------------------------------------
# Pending Approvals
# ---------------------------------------------------------------------------


class TestPendingApprovals:
    def test_create_and_list(self):
        approval = core_db.create_pending_approval(
            entity_type="thesis_status",
            proposed_change={"ticker": "MU", "new_status": "under_review", "reason": "Weak earnings"},
            ticker="MU",
            reason="Agent detected thesis pressure",
        )
        assert approval["status"] == "pending"
        assert approval["proposed_change"]["new_status"] == "under_review"

        approvals = core_db.get_pending_approvals()
        assert len(approvals) == 1
        assert approvals[0]["proposed_change"]["new_status"] == "under_review"

    def test_resolve_reject(self):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Test", "action_type": "review"},
        )
        resolved = core_db.resolve_approval(approval["id"], "rejected", "Not needed")
        assert resolved["status"] == "rejected"
        assert resolved["resolved_note"] == "Not needed"

    def test_resolve_already_resolved_raises(self):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Test", "action_type": "review"},
        )
        core_db.resolve_approval(approval["id"], "rejected")
        with pytest.raises(ValueError, match="already"):
            core_db.resolve_approval(approval["id"], "approved")

    def test_approve_creates_action_item(self):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
        )
        core_db.resolve_approval(approval["id"], "approved")

        items = core_db.get_action_items(ticker="MU")
        assert len(items) == 1
        assert items[0]["description"] == "Review MU thesis"

    def test_approve_creates_watch_trigger(self):
        approval = core_db.create_pending_approval(
            entity_type="watch_trigger",
            proposed_change={"condition": "VIX > 25", "trigger_type": "macro"},
        )
        core_db.resolve_approval(approval["id"], "approved")

        triggers = core_db.get_watch_triggers(status="active")
        assert len(triggers) == 1
        assert triggers[0]["condition"] == "VIX > 25"

    def test_filter_by_ticker(self):
        core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "A"},
            ticker="MU",
        )
        core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "B"},
            ticker="CRWD",
        )

        mu_approvals = core_db.get_pending_approvals(ticker="MU")
        assert len(mu_approvals) == 1
        assert mu_approvals[0]["ticker"] == "MU"
