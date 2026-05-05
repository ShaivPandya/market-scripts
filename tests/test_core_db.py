"""Tests for portfolio/core_db.py -- CRUD for all 7 entity types."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
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
# Investment Ideas
# ---------------------------------------------------------------------------


class TestInvestmentIdeas:
    def test_create_update_list_and_archive(self):
        idea = core_db.create_investment_idea(
            "aapl",
            company_name="Apple",
            user_notes="High-quality compounder under review.",
            tags=["quality", "mega-cap"],
        )

        assert idea["ticker"] == "AAPL"
        assert idea["status"] == "watching"
        assert idea["tags"] == ["quality", "mega-cap"]

        updated = core_db.update_investment_idea(
            idea["id"],
            status="researching",
            user_notes="Waiting for valuation and management evidence.",
            tags=["quality"],
        )
        assert updated["status"] == "researching"
        assert updated["user_notes"].startswith("Waiting")
        assert updated["tags"] == ["quality"]

        ideas = core_db.list_investment_ideas()
        assert [row["ticker"] for row in ideas] == ["AAPL"]

        archived = core_db.archive_investment_idea(idea["id"])
        assert archived["status"] == "archived"
        assert core_db.list_investment_ideas() == []
        assert core_db.list_investment_ideas(include_archived=True)[0]["ticker"] == "AAPL"

    def test_evaluation_persistence_and_acceptance_links_recommendation(self):
        idea = core_db.create_investment_idea("GOOG", company_name="Alphabet", user_notes="Review search and cloud.")
        result = {
            "action": "watch",
            "recommendation_status": "review_required",
            "score": 64,
            "confidence": 0.52,
            "thesis_statement": "Alphabet needs more valuation evidence.",
            "rationale": "Evidence is incomplete.",
            "factor_scores": {"business_quality": {"score": 75, "status": "strong"}},
            "missing_information": [{"field": "valuation", "severity": "critical", "reason": "No downside case."}],
            "data_quality": {"critical_data_quality": "degraded", "source_quality": "degraded"},
            "evidence": [{"source": "notes", "summary": "User wants review."}],
            "disconfirming_evidence": [{"source": "data_gap", "summary": "No valuation."}],
            "portfolio_fit": {"status": "needs_review"},
            "recommendation_record": {"action": "watch"},
        }

        evaluation = core_db.create_idea_evaluation(idea["id"], result, job_id="job-1")

        assert evaluation["ticker"] == "GOOG"
        assert evaluation["missing_information"][0]["field"] == "valuation"
        refreshed = core_db.get_investment_idea(idea["id"])
        assert refreshed["latest_evaluation_id"] == evaluation["id"]
        assert refreshed["latest_job_id"] == "job-1"
        assert refreshed["status"] == "ready_for_review"

        accepted = core_db.mark_idea_evaluation_accepted(
            evaluation["id"], recommendation_id=123, action_approval_id=456
        )

        assert accepted["recommendation_id"] == 123
        assert accepted["action_approval_id"] == 456
        assert accepted["accepted_at"] is not None
        assert core_db.get_investment_idea(idea["id"])["status"] == "accepted"

    def test_comparison_run_persists_rankings_and_lists_latest_first(self):
        first = core_db.create_investment_idea("MSFT", company_name="Microsoft")
        second = core_db.create_investment_idea("NVDA", company_name="Nvidia")
        first_eval = core_db.create_idea_evaluation(
            first["id"],
            {
                "action": "watch",
                "score": 70,
                "confidence": 0.61,
                "rationale": "Good but not top ranked.",
                "recommendation_record": {"action": "watch"},
            },
            job_id="comparison-job-1",
        )
        second_eval = core_db.create_idea_evaluation(
            second["id"],
            {
                "action": "buy",
                "score": 84,
                "confidence": 0.82,
                "rationale": "Best fresh setup.",
                "recommendation_record": {"action": "buy"},
            },
            job_id="comparison-job-1",
        )

        old_run = core_db.create_idea_comparison_run(
            job_id="comparison-job-0",
            scope_statuses=["watching"],
            summary="Older run.",
            rankings=[
                {
                    "idea_id": first["id"],
                    "evaluation_id": first_eval["id"],
                    "ticker": "MSFT",
                    "rank": 1,
                    "action": "watch",
                    "score": 70,
                    "confidence": 0.61,
                    "confidence_level": "medium",
                    "rationale": "Older ranking.",
                }
            ],
            raw_result={"summary": "Older run."},
            run_id="older-run",
        )
        latest = core_db.create_idea_comparison_run(
            job_id="comparison-job-1",
            scope_statuses=["watching", "ready_for_review"],
            summary="Fresh comparative ranking.",
            rankings=[
                {
                    "idea_id": second["id"],
                    "evaluation_id": second_eval["id"],
                    "ticker": "NVDA",
                    "rank": 1,
                    "action": "buy",
                    "score": 84,
                    "confidence": 0.82,
                    "confidence_level": "high",
                    "rationale": "Best fresh setup.",
                },
                {
                    "idea_id": first["id"],
                    "evaluation_id": first_eval["id"],
                    "ticker": "MSFT",
                    "rank": 2,
                    "action": "watch",
                    "score": 70,
                    "confidence": 0.61,
                    "confidence_level": "medium",
                    "rationale": "Good but lower confidence.",
                },
            ],
            raw_result={"summary": "Fresh comparative ranking."},
            run_id="latest-run",
        )

        loaded = core_db.get_idea_comparison_run(latest["run_id"])
        assert loaded["summary"] == "Fresh comparative ranking."
        assert loaded["scope_statuses"] == ["watching", "ready_for_review"]
        assert [row["ticker"] for row in loaded["rankings"]] == ["NVDA", "MSFT"]
        assert loaded["rankings"][0]["confidence_level"] == "high"

        runs = core_db.list_idea_comparison_runs(limit=2)
        assert [run["run_id"] for run in runs] == [latest["run_id"], old_run["run_id"]]


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
        assert resolved["application_status"] == "not_applicable"
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
        updated = core_db.get_pending_approval(approval["id"])
        assert updated["status"] == "approved"
        assert updated["application_status"] == "applied"
        assert updated["application_attempts"] == 1
        assert updated["application_error"] is None

    def test_approve_failure_rolls_back_side_effect_and_can_retry(self, monkeypatch):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
        )
        original = core_db._APPROVAL_SIDE_EFFECT_HANDLERS["action_item"]

        def fail_after_insert(conn, current, change, callbacks):
            original(conn, current, change, callbacks)
            raise RuntimeError("forced apply failure")

        monkeypatch.setitem(core_db._APPROVAL_SIDE_EFFECT_HANDLERS, "action_item", fail_after_insert)

        with pytest.raises(core_db.ApprovalApplicationError, match="forced apply failure"):
            core_db.resolve_approval(approval["id"], "approved")

        assert core_db.get_action_items(ticker="MU") == []
        failed = core_db.get_pending_approval(approval["id"])
        assert failed["status"] == "pending"
        assert failed["application_status"] == "failed"
        assert failed["application_attempts"] == 1
        assert "forced apply failure" in failed["application_error"]

        monkeypatch.setitem(core_db._APPROVAL_SIDE_EFFECT_HANDLERS, "action_item", original)
        resolved = core_db.resolve_approval(approval["id"], "approved")

        items = core_db.get_action_items(ticker="MU")
        assert len(items) == 1
        assert items[0]["description"] == "Review MU thesis"
        assert resolved["status"] == "approved"
        assert resolved["application_status"] == "applied"
        assert resolved["application_attempts"] == 2

    def test_approve_already_applied_is_idempotent(self):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
        )
        first = core_db.resolve_approval(approval["id"], "approved")
        second = core_db.resolve_approval(approval["id"], "approved")

        assert first["status"] == "approved"
        assert second["status"] == "approved"
        assert second["application_status"] == "applied"
        assert second["application_attempts"] == 1
        assert len(core_db.get_action_items(ticker="MU")) == 1

    def test_reject_after_failed_apply_marks_not_applicable(self, monkeypatch):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
        )

        def fail_apply(conn, current, change, callbacks):
            raise RuntimeError("cannot apply")

        monkeypatch.setitem(core_db._APPROVAL_SIDE_EFFECT_HANDLERS, "action_item", fail_apply)
        with pytest.raises(core_db.ApprovalApplicationError):
            core_db.resolve_approval(approval["id"], "approved")

        rejected = core_db.resolve_approval(approval["id"], "rejected", "Skip it")
        assert rejected["status"] == "rejected"
        assert rejected["application_status"] == "not_applicable"
        assert rejected["resolved_note"] == "Skip it"

    def test_approve_endpoint_surfaces_application_failure_as_conflict(self, monkeypatch):
        from api.exceptions import ConflictError
        from api.routers.approvals import approve_item

        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
        )

        def fail_apply(conn, current, change, callbacks):
            raise RuntimeError("cannot apply")

        monkeypatch.setitem(core_db._APPROVAL_SIDE_EFFECT_HANDLERS, "action_item", fail_apply)
        from api.routers.approvals import ResolveRequest

        with pytest.raises(ConflictError) as exc:
            approve_item(approval["id"], ResolveRequest(note="Apply in test"))

        assert exc.value.status_code == 409
        assert "cannot apply" in exc.value.message

    def test_unknown_entity_type_fails_retryably(self):
        approval = core_db.create_pending_approval(
            entity_type="unknown_entity",
            proposed_change={"description": "No handler"},
        )

        with pytest.raises(core_db.ApprovalApplicationError, match="Unsupported approval entity_type"):
            core_db.resolve_approval(approval["id"], "approved")

        failed = core_db.get_pending_approval(approval["id"])
        assert failed["status"] == "pending"
        assert failed["application_status"] == "failed"
        assert failed["application_attempts"] == 1

    def test_malformed_proposed_change_fails_retryably(self):
        conn = core_db._get_conn()
        with core_db._lock:
            cur = conn.execute(
                "INSERT INTO pending_approvals (entity_type, proposed_change, source_type, status, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("action_item", "{not-json", "workflow", "pending", "2026-05-03T00:00:00+00:00"),
            )
            conn.commit()

        with pytest.raises(core_db.ApprovalApplicationError, match="proposed_change"):
            core_db.resolve_approval(cur.lastrowid, "approved")

        failed = core_db.get_pending_approval(cur.lastrowid)
        assert failed["status"] == "pending"
        assert failed["application_status"] == "failed"

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


class TestPendingApprovalContracts:
    def test_create_pending_approval_persists_action_schema_metadata_and_audit(self):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
            reason="Workflow suggested review",
            source_type="workflow",
            source_id="run-approval",
            action_id="create_action_item",
            action_schema_name="create_action_item",
            action_schema_version=3,
            action_input_hash="abc123",
            request_schema_name="ActionRequest",
            request_schema_version=2,
        )

        saved = core_db.get_pending_approval(approval["id"])
        assert saved is not None
        assert saved["action_id"] == "create_action_item"
        assert saved["action_schema_name"] == "create_action_item"
        assert saved["action_schema_version"] == 3
        assert saved["action_input_hash"] == "abc123"
        assert saved["request_schema_name"] == "ActionRequest"
        assert saved["request_schema_version"] == 2

        event = core_db.get_audit_events(action_name="approval.created", limit=5)[0]
        assert event["status"] == "pending"
        assert event["after_summary"]["action_id"] == "create_action_item"
        assert event["source_lineage"]["source_type"] == "workflow"
        assert event["source_lineage"]["source_id"] == "run-approval"
        assert event["source_lineage"]["action_input_hash"] == "abc123"

    def test_action_backed_approval_precondition_failure_is_retryable_and_non_mutating(self):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"item_id": 9999, "resolution_note": "Done"},
            source_type="workflow",
            source_id="run-precondition",
            action_id="complete_action_item",
            action_schema_name="complete_action_item",
            action_schema_version=1,
        )

        with pytest.raises(core_db.ApprovalApplicationError, match="not found"):
            core_db.resolve_approval(approval["id"], "approved")

        failed = core_db.get_pending_approval(approval["id"])
        assert failed["status"] == "pending"
        assert failed["application_status"] == "failed"
        assert failed["application_attempts"] == 1
        assert core_db.get_action_items() == []

        child_runs = core_db.get_action_runs(action_id="complete_action_item", approval_id=approval["id"])
        assert len(child_runs) == 1
        assert child_runs[0]["status"] == "failed"

    def test_approval_apply_success_creates_child_action_run_parent_link_and_audit_chain(self):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
            source_type="workflow",
            source_id="run-success",
            action_id="create_action_item",
            action_schema_name="create_action_item",
            action_schema_version=1,
        )

        resolved = core_db.resolve_approval(approval["id"], "approved", "Apply it")

        assert resolved["status"] == "approved"
        assert resolved["application_status"] == "applied"
        assert core_db.get_action_items(ticker="MU")[0]["description"] == "Review MU thesis"

        resolve_run = core_db.get_action_runs(action_id="resolve_approval", approval_id=approval["id"])[0]
        child_run = core_db.get_action_runs(action_id="create_action_item", approval_id=approval["id"])[0]
        assert child_run["status"] == "succeeded"
        assert child_run["parent_action_run_id"] == resolve_run["id"]

        audit_names = {row["action_name"] for row in core_db.get_audit_events(limit=20)}
        assert {"approval.apply.started", "approval.applied", "approval.resolved"} <= audit_names

    def test_create_pending_approval_once_dedupes_same_workflow_payload_per_source_id_only(self):
        first = core_db.create_pending_approval_once(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
            source_type="workflow",
            source_id="run-1",
            action_id="create_action_item",
        )
        second = core_db.create_pending_approval_once(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
            source_type="workflow",
            source_id="run-1",
            action_id="create_action_item",
        )
        third = core_db.create_pending_approval_once(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
            source_type="workflow",
            source_id="run-2",
            action_id="create_action_item",
        )

        assert first["id"] == second["id"]
        assert third["id"] != first["id"]
        assert len(core_db.get_pending_approvals(status=None)) == 2

    def test_fresh_application_lease_blocks_duplicate_apply_but_stale_lease_recovers(self):
        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
            source_type="workflow",
            source_id="run-lease",
            action_id="create_action_item",
            action_schema_name="create_action_item",
            action_schema_version=1,
        )
        conn = core_db._get_conn()
        with core_db._lock:
            conn.execute(
                "UPDATE pending_approvals SET application_status = 'applying', application_started_at = ?, "
                "application_attempts = 1 WHERE id = ?",
                (datetime.now(UTC).isoformat(), approval["id"]),
            )
            conn.commit()

        with pytest.raises(ValueError, match="already in progress"):
            core_db.apply_approval_resolution(approval["id"], "approved")

        stale_started = (datetime.now(UTC) - core_db.APPROVAL_APPLICATION_LEASE - timedelta(minutes=1)).isoformat()
        with core_db._lock:
            conn.execute(
                "UPDATE pending_approvals SET application_status = 'applying', application_started_at = ? WHERE id = ?",
                (stale_started, approval["id"]),
            )
            conn.commit()

        resolved = core_db.apply_approval_resolution(approval["id"], "approved")
        assert resolved["status"] == "approved"
        assert resolved["application_status"] == "applied"
        assert core_db.get_action_items(ticker="MU")[0]["description"] == "Review MU thesis"

    def test_old_approval_action_schema_replays_after_schema_evolution_through_core_db_resolution(self, monkeypatch):
        from typing import Literal

        import portfolio.action_registry as registry
        from portfolio.action_registry import (
            CreateActionItemInput,
            DomainAction,
            register_action_schema_version,
            register_action_upgrade_adapter,
        )

        approval = core_db.create_pending_approval(
            entity_type="action_item",
            proposed_change={"description": "Review MU thesis", "action_type": "review", "ticker": "MU"},
            ticker="MU",
            source_type="workflow",
            source_id="run-old-schema",
            action_id="create_action_item",
            action_schema_name="create_action_item",
            action_schema_version=1,
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

        resolved = core_db.resolve_approval(approval["id"], "approved")

        assert resolved["application_status"] == "applied"
        assert core_db.get_action_items(ticker="MU")[0]["description"] == "Review MU thesis"
