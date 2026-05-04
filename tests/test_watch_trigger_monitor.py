from __future__ import annotations

import os

import pytest

import portfolio.core_db as core_db
import portfolio.news_digests as digests
import portfolio.thesis_db as thesis_db


@pytest.fixture
def temp_monitor_state(tmp_path, monkeypatch):
    if core_db._conn:
        core_db._conn.close()
    if thesis_db._conn:
        thesis_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    monkeypatch.setattr(thesis_db, "DB_PATH", tmp_path / "thesis.db")
    monkeypatch.setattr(thesis_db, "_conn", None)

    base = tmp_path / "news_digests"
    monkeypatch.setattr(digests, "DIGESTS_DIR", base)
    monkeypatch.setattr(digests, "MANIFEST_PATH", base / "manifest.json")
    monkeypatch.setattr(digests, "FILES_DIR", base / "files")
    monkeypatch.setattr(digests, "DIGESTS_GCS_PREFIX", "test/news_digests")
    monkeypatch.setattr(digests, "MANIFEST_GCS_KEY", "test/news_digests/manifest.json")
    monkeypatch.setattr(digests, "FILES_GCS_PREFIX", "test/news_digests/files")
    os.environ["STATE_STORAGE_BACKEND"] = "local"
    yield
    if core_db._conn:
        core_db._conn.close()
    if thesis_db._conn:
        thesis_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)
    monkeypatch.setattr(thesis_db, "_conn", None)


def _digest_context(headline: str, notes: list[str] | None = None) -> dict:
    return {
        "window_days": 7,
        "cutoff_date": "2026-05-01",
        "fallback_used": False,
        "counts": {"digests": 1, "stories": 1},
        "digests": [
            {
                "id": "2026-05-02-test-digest",
                "title": "Test Digest",
                "generated_date": "2026-05-02",
                "sections": [
                    {
                        "name": "earnings",
                        "stories": [{"headline": headline, "notes": notes or []}],
                    }
                ],
            }
        ],
    }


def _verified_search(query: str) -> dict:
    return {
        "query": query,
        "summary": "Verified source summary.",
        "citations": [{"title": "Company release", "url": "https://example.com/release"}],
        "citation_count": 1,
    }


def test_approved_watch_trigger_preserves_definition(temp_monitor_state):
    definition = {"type": "price_level", "ticker": "MU", "operator": ">=", "threshold": 150}
    approval = core_db.create_pending_approval(
        entity_type="watch_trigger",
        ticker="MU",
        proposed_change={
            "condition": "MU >= 150",
            "trigger_type": "price_level",
            "definition": definition,
        },
    )

    core_db.resolve_approval(approval["id"], "approved", "Apply trigger")

    trigger = core_db.get_watch_triggers(status="active", ticker="MU")[0]
    assert trigger["definition_json"] == definition
    assert trigger["definition"] == definition


def test_infers_safe_price_definition_and_fires(monkeypatch, temp_monitor_state):
    from api import watch_trigger_monitor

    monkeypatch.setattr(watch_trigger_monitor, "_latest_price", lambda _ticker: {"value": 151.0, "as_of": "2026-05-02"})
    monkeypatch.setattr(
        watch_trigger_monitor,
        "_load_news_context",
        lambda _days: {"fallback_used": False, "counts": {"digests": 0, "stories": 0}, "digests": []},
    )
    core_db.create_watch_trigger("MU >= 150", "price_level", ticker="MU")

    result = watch_trigger_monitor.run_watch_trigger_monitor()

    assert result["fired"] == 1
    assert core_db.get_watch_triggers(status="fired", ticker="MU") == []
    approvals = core_db.get_pending_approvals(status="pending")
    assert {approval["action_id"] for approval in approvals} >= {
        "update_watch_trigger_definition",
        "fire_watch_trigger",
        "create_action_item",
    }
    definition_approval = [
        approval for approval in approvals if approval["action_id"] == "update_watch_trigger_definition"
    ][0]
    fire_approval = [approval for approval in approvals if approval["action_id"] == "fire_watch_trigger"][0]
    assert definition_approval["proposed_change"]["definition"]["type"] == "price_level"
    assert fire_approval["proposed_change"]["result"]["inferred_definition"]["threshold"] == 150.0
    assert (
        len([a for a in core_db.get_pending_approvals(status="pending") if a["action_id"] == "create_action_item"]) == 1
    )


def test_deterministic_fire_adds_news_enrichment_and_dedupes(monkeypatch, temp_monitor_state):
    from api import watch_trigger_monitor

    monkeypatch.setattr(watch_trigger_monitor, "_latest_price", lambda _ticker: {"value": 151.0, "as_of": "2026-05-02"})
    monkeypatch.setattr(
        watch_trigger_monitor,
        "_load_news_context",
        lambda _days: _digest_context(
            "Micron raises HBM guidance after AI demand improves - (Reuters)",
            ["Micron management said HBM demand supports stronger guidance."],
        ),
    )
    monkeypatch.setattr(watch_trigger_monitor, "_search_web", _verified_search)
    core_db.create_watch_trigger(
        "MU >= 150",
        "price_level",
        ticker="MU",
        definition={
            "type": "price_level",
            "ticker": "MU",
            "operator": ">=",
            "threshold": 150,
            "entities": ["Micron"],
            "topics": ["HBM", "guidance"],
            "materiality_threshold": 0.65,
        },
    )

    first = watch_trigger_monitor.run_watch_trigger_monitor()
    second = watch_trigger_monitor.run_watch_trigger_monitor()

    assert first["fired"] == 1
    assert second["checked"] == 1
    fire_approvals = [
        a for a in core_db.get_pending_approvals(status="pending") if a["action_id"] == "fire_watch_trigger"
    ]
    assert len(fire_approvals) == 1
    fire_result = fire_approvals[0]["proposed_change"]["result"]
    assert fire_result["news_enrichment"]["matches"][0]["sources"] == ["Reuters"]
    assert fire_result["news_enrichment"]["verifications"][0]["citations"][0]["url"]
    assert (
        len([a for a in core_db.get_pending_approvals(status="pending") if a["action_id"] == "create_action_item"]) == 1
    )
    assert (
        len([a for a in core_db.get_pending_approvals(status="pending") if a["action_id"] == "create_research_note"])
        == 1
    )


def test_gated_news_trigger_creates_review_only(monkeypatch, temp_monitor_state):
    from api import watch_trigger_monitor

    catalyst = core_db.create_catalyst("MU", "HBM ramp: AI memory demand accelerates")
    kill_condition = core_db.create_kill_condition("MU", "HBM miss: AI demand weakens")
    claim = core_db.create_thesis_claim(
        {
            "ticker": "MU",
            "claim": "Micron can sustain HBM-driven growth.",
            "expected_evidence": "HBM demand supports stronger guidance.",
            "disconfirming_evidence": "HBM demand weakens.",
            "linked_catalyst_ids": [catalyst["id"]],
            "linked_kill_condition_ids": [kill_condition["id"]],
        }
    )
    monkeypatch.setattr(
        watch_trigger_monitor,
        "_load_news_context",
        lambda _days: _digest_context(
            "Micron raises HBM guidance after AI demand improves - (Reuters)",
            ["HBM demand supports stronger guidance."],
        ),
    )
    monkeypatch.setattr(watch_trigger_monitor, "_search_web", _verified_search)
    core_db.create_watch_trigger(
        "Micron HBM guidance news",
        "fundamental_news",
        ticker="MU",
        definition={
            "type": "fundamental_news",
            "ticker": "MU",
            "entities": ["Micron"],
            "topics": ["HBM", "guidance"],
            "lookback_days": 7,
            "materiality_threshold": 0.65,
            "linked_claim_ids": [claim["id"]],
            "linked_catalyst_ids": [catalyst["id"]],
            "linked_kill_condition_ids": [kill_condition["id"]],
            "source_requirements": {"min_sources": 1, "primary_source_required": False},
        },
    )

    result = watch_trigger_monitor.run_watch_trigger_monitor()

    assert result["fired"] == 1
    action_approval = [
        approval
        for approval in core_db.get_pending_approvals(status="pending")
        if approval["action_id"] == "create_action_item"
    ][0]
    assert action_approval["proposed_change"]["description"].startswith("Needs review:")
    assert core_db.get_recommendations() == []
    assert core_db.get_thesis_claim(claim["id"])["status"] == "active"
    assert core_db.get_catalysts("MU")[0]["status"] == "pending"
    assert core_db.get_kill_conditions("MU")[0]["status"] == "active"


def test_news_trigger_uses_web_fallback_when_digest_missing(monkeypatch, temp_monitor_state):
    from api import watch_trigger_monitor

    monkeypatch.setattr(
        watch_trigger_monitor,
        "_load_news_context",
        lambda _days: {"fallback_used": False, "counts": {"digests": 0, "stories": 0}, "digests": []},
    )
    monkeypatch.setattr(watch_trigger_monitor, "_search_web", _verified_search)
    core_db.create_watch_trigger(
        "Micron material HBM news",
        "news_event",
        ticker="MU",
        definition={
            "type": "news_event",
            "ticker": "MU",
            "entities": ["Micron"],
            "topics": ["HBM"],
            "source_requirements": {"min_sources": 1, "primary_source_required": True},
        },
    )

    result = watch_trigger_monitor.run_watch_trigger_monitor()

    assert result["fired"] == 1
    fire_approval = [
        approval
        for approval in core_db.get_pending_approvals(status="pending")
        if approval["action_id"] == "fire_watch_trigger"
    ][0]
    assert fire_approval["proposed_change"]["result"]["news"]["web_fallback_used"] is True
    assert (
        len([a for a in core_db.get_pending_approvals(status="pending") if a["action_id"] == "create_action_item"]) == 1
    )


def test_news_trigger_does_not_fire_without_source_provenance(monkeypatch, temp_monitor_state):
    from api import watch_trigger_monitor

    monkeypatch.setattr(
        watch_trigger_monitor,
        "_load_news_context",
        lambda _days: _digest_context(
            "Micron raises HBM guidance after AI demand improves",
            ["HBM demand supports stronger guidance."],
        ),
    )
    monkeypatch.setattr(
        watch_trigger_monitor,
        "_search_web",
        lambda query: {"query": query, "summary": "No cited source.", "citations": [], "citation_count": 0},
    )
    core_db.create_watch_trigger(
        "Micron HBM guidance news",
        "fundamental_news",
        ticker="MU",
        definition={
            "type": "fundamental_news",
            "ticker": "MU",
            "entities": ["Micron"],
            "topics": ["HBM", "guidance"],
            "source_requirements": {"min_sources": 1, "primary_source_required": False},
        },
    )

    result = watch_trigger_monitor.run_watch_trigger_monitor()

    trigger = core_db.get_watch_triggers(status="active", ticker="MU")[0]
    assert result["fired"] == 0
    check_approval = [
        approval
        for approval in core_db.get_pending_approvals(status="pending")
        if approval["action_id"] == "update_watch_trigger_check"
    ][0]
    core_db.resolve_approval(check_approval["id"], "approved", "Apply check result")
    trigger = core_db.get_watch_triggers(status="active", ticker="MU")[0]
    assert "source requirements" in trigger["last_evidence"]
    assert core_db.get_action_items(status="open", ticker="MU") == []
