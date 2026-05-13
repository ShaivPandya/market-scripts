from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from api import agent_tools
from api.routers import agent as agent_router
from ontology import action_registry
from ontology.action_registry import ActionValidationError, iter_tool_exposures
from ontology.policy import admin_actor, agent_actor

PROPOSAL_TOOL_NAMES = {
    "propose_thesis_status_change",
    "propose_action_item",
    "propose_catalyst_status_change",
    "propose_kill_condition_status_change",
    "propose_watch_trigger",
    "propose_portfolio_positions_update",
    "propose_hedge_positions_update",
    "propose_thesis_content_update",
    "propose_catalyst",
    "propose_kill_condition",
    "propose_news_digest_delete",
}


def test_get_sector_metrics_tool_uses_snapshot_when_required(monkeypatch):
    monkeypatch.setattr(agent_tools, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_tools, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "api.snapshot_store.get_snapshot_response",
        lambda key: {"weights_df": [{"Sector": "Technology"}], "_meta": {"snapshot": {"key": key}}},
    )
    monkeypatch.setattr("api.snapshot_store.snapshots_required", lambda: True)
    monkeypatch.setattr(
        "equities.sector_metrics.sector_metrics.get_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live compute should not run")),
    )

    payload = json.loads(agent_tools.execute_tool("get_sector_metrics", {"_force_refresh": True}))

    assert payload["weights_df"][0]["Sector"] == "Technology"
    assert payload["_meta"]["snapshot"]["key"] == "sector_metrics:sp500:2y"


def test_get_liquidity_tool_uses_snapshot_when_required(monkeypatch):
    monkeypatch.setattr(agent_tools, "get_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_tools, "set_cached", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "api.snapshot_store.get_snapshot_response",
        lambda key: {
            "composite_score": -0.56,
            "regime": "tight",
            "df_weekly": [{"should": "be dropped"}],
            "composite_series": [{"should": "be dropped"}],
            "_meta": {"snapshot": {"key": key}},
        },
    )
    monkeypatch.setattr("api.snapshot_store.snapshots_required", lambda: True)
    monkeypatch.setattr(
        "macro.liquidity.liquidity.get_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live liquidity compute should not run")),
    )

    payload = json.loads(agent_tools.execute_tool("get_liquidity", {"_force_refresh": True}))

    assert payload["composite_score"] == -0.56
    assert payload["regime"] == "tight"
    assert "df_weekly" not in payload
    assert "composite_series" not in payload
    assert payload["_meta"]["snapshot"]["key"] == "liquidity:current:v1"


def test_agent_tools_return_first_class_portfolio_risk(tmp_path, monkeypatch):
    from api import position_risk_store
    from api.position_risk_store import write_portfolio_risk_snapshot

    monkeypatch.setattr(position_risk_store, "_SQLITE_PATH", tmp_path / "position_risk.sqlite3")
    write_portfolio_risk_snapshot(
        {
            "result_id": "portfolio-risk:agent",
            "as_of": "2099-01-01",
            "computed_at": "2099-01-01T22:00:00+00:00",
            "average_risk_score": 0.44,
            "max_risk_score": 0.8,
            "risk_score": 0.44,
            "risk_level": "high",
            "confidence": 0.9,
            "quality": "ok",
            "position_count": 1,
            "risk_buckets": {"high": 1, "medium": 0, "low": 0},
            "source_status": {},
            "degraded_modules": [],
            "input_snapshots": {},
            "position_snapshot_ids": {"MU": "position-risk:MU:agent"},
        }
    )

    payload = json.loads(agent_tools.execute_tool("get_portfolio_risk", {}))

    assert payload["result_id"] == "portfolio-risk:agent"
    assert payload["position_snapshot_ids"]["MU"] == "position-risk:MU:agent"


def _proposal_tool_cases(digest_id: str) -> dict[str, tuple[dict, str, str]]:
    return {
        "propose_thesis_status_change": (
            {"ticker": "mu", "new_status": "under_review", "reason": "Fresh evidence"},
            "change_thesis_status",
            "thesis_status",
        ),
        "propose_action_item": (
            {
                "ticker": "mu",
                "description": "Review sizing",
                "action_type": "review",
                "urgency": "high",
                "reason": "Risk changed",
            },
            "create_action_item",
            "action_item",
        ),
        "propose_catalyst_status_change": (
            {
                "ticker": "mu",
                "catalyst_id": 12,
                "new_status": "played_out",
                "evidence": "Management confirmed it",
                "reason": "Update catalyst status",
            },
            "update_catalyst_status",
            "catalyst_status",
        ),
        "propose_kill_condition_status_change": (
            {
                "ticker": "mu",
                "kill_condition_id": 34,
                "new_status": "triggered",
                "reason": "Condition occurred",
            },
            "update_kill_condition_status",
            "kill_condition_status",
        ),
        "propose_watch_trigger": (
            {
                "ticker": "mu",
                "condition": "MU breaks below 90",
                "trigger_type": "price_level",
                "expires_at": "2026-06-01T00:00:00Z",
                "definition": {"operator": "<", "value": 90},
                "reason": "Watch downside risk",
            },
            "create_watch_trigger",
            "watch_trigger",
        ),
        "propose_portfolio_positions_update": (
            {
                "positions": [
                    {
                        "ticker": "mu",
                        "asset": "equity",
                        "direction": "long",
                        "contrarian": False,
                        "conviction": 4,
                    }
                ],
                "reason": "Resize portfolio",
            },
            "update_portfolio_positions",
            "portfolio_positions",
        ),
        "propose_hedge_positions_update": (
            {"positions": [{"ticker": "spy", "direction": "short", "shares": 5}], "reason": "Add hedge"},
            "update_hedge_positions",
            "hedge_positions",
        ),
        "propose_thesis_content_update": (
            {"ticker": "mu", "content": "# MU\n\nUpdated thesis", "reason": "Refresh thesis"},
            "save_thesis_content",
            "thesis_content",
        ),
        "propose_catalyst": (
            {
                "ticker": "mu",
                "description": "HBM ramp",
                "category": "fundamental",
                "target_date": "2026-06-30",
                "reason": "Track material catalyst",
            },
            "create_catalyst",
            "catalyst",
        ),
        "propose_kill_condition": (
            {
                "ticker": "mu",
                "condition": "Gross margin breaks below 40%",
                "metric": "gross_margin",
                "threshold": "40%",
                "reason": "Track thesis risk",
            },
            "create_kill_condition",
            "kill_condition",
        ),
        "propose_news_digest_delete": (
            {"digest_id": digest_id, "reason": "Remove stale digest"},
            "delete_portfolio_news_digest",
            "news_digest_delete",
        ),
    }


@pytest.fixture
def agent_proposal_state(tmp_path, monkeypatch):
    import portfolio.core_db as core_db
    import portfolio.news_digests as digests

    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)

    base = tmp_path / "news_digests"
    monkeypatch.setattr(digests, "DIGESTS_DIR", base)
    monkeypatch.setattr(digests, "MANIFEST_PATH", base / "manifest.json")
    monkeypatch.setattr(digests, "FILES_DIR", base / "files")
    monkeypatch.setattr(digests, "DIGESTS_GCS_PREFIX", "test/news_digests")
    monkeypatch.setattr(digests, "MANIFEST_GCS_KEY", "test/news_digests/manifest.json")
    monkeypatch.setattr(digests, "FILES_GCS_PREFIX", "test/news_digests/files")
    monkeypatch.setenv("STATE_STORAGE_BACKEND", "local")

    yield SimpleNamespace(core_db=core_db, digests=digests)

    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)


def test_cached_singleflight_fetches_once(monkeypatch):
    from cachetools import TTLCache

    import api.cache as cache_mod

    monkeypatch.setattr(cache_mod, "_DISK_CACHE_ENABLED", False)

    calls = 0
    calls_lock = threading.Lock()

    def loader():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return {"value": 1}

    cache_token = TTLCache(maxsize=8, ttl=60)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(agent_tools._cached_singleflight, cache_token, "k", loader) for _ in range(4)]
        results = [f.result() for f in futs]

    assert calls == 1
    assert all(v[0]["value"] == 1 for v in results)
    assert {"miss_fetch", "miss_wait"} & {v[1] for v in results}


def test_fetch_with_cache_force_refresh_bypasses_cached_value(monkeypatch):
    from cachetools import TTLCache

    import api.cache as cache_mod

    monkeypatch.setattr(cache_mod, "_DISK_CACHE_ENABLED", False)

    calls = 0

    def loader():
        nonlocal calls
        calls += 1
        return {"value": calls}

    cache_token = TTLCache(maxsize=8, ttl=60)
    first, first_meta = agent_tools._fetch_with_cache(cache_token, "k", loader)
    second, second_meta = agent_tools._fetch_with_cache(cache_token, "k", loader)
    refreshed, refreshed_meta = agent_tools._fetch_with_cache(cache_token, "k", loader, force_refresh=True)

    assert first["value"] == 1
    assert second["value"] == 1
    assert refreshed["value"] == 2
    assert first_meta["cache"] == "miss_fetch"
    assert second_meta["cache"] == "hit"
    assert refreshed_meta["cache"] == "refresh"
    assert isinstance(refreshed_meta["data_age_seconds"], int)
    assert refreshed_meta["stale"] is False
    assert refreshed_meta["fresh"] is True
    assert refreshed_meta["refreshed"] is True


def test_execute_tool_outputs_valid_json_when_compacted(monkeypatch):
    huge_payload = {
        "rows": [{"i": i, "text": "x" * 120} for i in range(3000)],
        "nested": {"values": list(range(1000))},
    }
    monkeypatch.setattr("api.agent_tools._dispatch", lambda _name, _args: (huge_payload, {"cache": "miss_fetch"}))

    raw = agent_tools.execute_tool("get_workspace", {})
    payload = json.loads(raw)

    assert isinstance(payload, dict)
    assert "_meta" in payload
    assert payload["_meta"]["tool"] == "get_workspace"
    assert payload["_meta"]["output_chars"] <= payload["_meta"]["max_chars"]


def test_agent_capability_registry_covers_user_facing_app_surface():
    names = {cap.name for cap in agent_tools.AGENT_CAPABILITIES}
    expected = {
        "get_commodity_research",
        "get_commodities",
        "get_commodities_curve",
        "get_portfolio_news",
        "get_country_dashboard",
        "get_index_dashboard",
        "get_fx_dashboard",
        "run_fx_model",
        "get_financials",
        "get_dcf_historical",
        "run_dcf_valuation",
        "run_chart",
        "run_ratio_chart",
        "run_quality_screen",
        "run_short_screen",
        "run_long_screen",
        "run_fundamental_momentum",
        "run_portfolio_analyzer",
        "run_portfolio_sizer",
        "run_hedging_tool",
        "get_workspace",
        "search_agent_capabilities",
    }

    assert expected <= names
    assert len(names) == len(agent_tools.AGENT_CAPABILITIES)
    assert all(cap.category and cap.access_mode and cap.aliases for cap in agent_tools.AGENT_CAPABILITIES)
    assert all(cap.schema_safe for cap in agent_tools.AGENT_CAPABILITIES)


def test_agent_tool_exposures_have_complete_governance_metadata():
    for tool in iter_tool_exposures(agent_exposed_only=True):
        assert tool.required_scopes
        assert tool.account_scope == "default-account"
        assert tool.portfolio_scope == "default-portfolio"
        assert tool.data_sensitivity in {
            "public_market",
            "portfolio_private",
            "research_private",
            "account_private",
            "operational_private",
        }
        assert tool.provider_egress in {
            "external_allowed",
            "external_allowed_raw_private",
            "external_blocked",
            "local_only",
        }
        assert tool.timeout_s > 0
        assert int(tool.retry_policy["max_attempts"]) >= 1
        assert tool.token_budget is not None and tool.token_budget > 0
        assert tool.cost_budget_usd is not None and tool.cost_budget_usd >= 0
        assert tool.rate_limit.get("label")
        assert tool.audit_level in {"standard", "enhanced", "financial_critical"}
        assert tool.failure_mode in {"fail_closed", "partial_allowed"}
        assert tool.lifecycle_state in {"draft", "enabled", "deprecated", "disabled"}


def test_agent_capability_registry_does_not_expose_direct_mutations():
    names = {cap.name for cap in agent_tools.AGENT_CAPABILITIES}

    forbidden_direct_tools = {
        "update_portfolio_positions",
        "update_hedge_positions",
        "save_thesis",
        "delete_portfolio_news_digest",
        "approve_item",
        "reject_item",
        "bulk_approve",
        "bulk_reject",
    }
    assert names.isdisjoint(forbidden_direct_tools)
    assert PROPOSAL_TOOL_NAMES <= names


def test_execute_tool_rejects_non_exposed_direct_mutation():
    payload = json.loads(
        agent_tools.execute_tool(
            "update_portfolio_positions",
            {
                "positions": [
                    {"ticker": "MU", "asset": "equity", "direction": "long", "contrarian": False, "conviction": 3}
                ]
            },
        )
    )

    assert "not exposed to the agent" in payload["error"]
    assert payload["_meta"]["status"] == "blocked"


def test_disabled_tools_are_not_exposed_and_return_blocked(monkeypatch):
    original = action_registry.get_tool_exposure("get_liquidity")
    monkeypatch.setitem(action_registry._TOOL_EXPOSURES, "get_liquidity", replace(original, lifecycle_state="disabled"))

    assert "get_liquidity" not in {tool.tool_name for tool in iter_tool_exposures(agent_exposed_only=True)}
    assert action_registry.is_agent_tool_exposed("get_liquidity") is False
    payload = json.loads(agent_tools.execute_tool("get_liquidity", {}))

    assert payload["_meta"]["status"] == "blocked"
    assert "not exposed to the agent" in payload["error"]


def test_deprecated_tools_remain_exposed(monkeypatch):
    original = action_registry.get_tool_exposure("get_liquidity")
    monkeypatch.setitem(
        action_registry._TOOL_EXPOSURES, "get_liquidity", replace(original, lifecycle_state="deprecated")
    )

    assert "get_liquidity" in {tool.tool_name for tool in iter_tool_exposures(agent_exposed_only=True)}
    assert action_registry.is_agent_tool_exposed("get_liquidity") is True


def test_agent_dispatch_proposal_tools_use_canonical_helper(monkeypatch):
    calls: list[tuple[str, dict, object]] = []

    def fake_propose_action_from_tool(tool_name, raw_input, context):
        calls.append((tool_name, raw_input, context))
        return {"id": len(calls), "entity_type": f"{tool_name}_entity", "ticker": "MU"}

    monkeypatch.setattr(agent_tools, "propose_action_from_tool", fake_propose_action_from_tool)

    for tool_name in sorted(PROPOSAL_TOOL_NAMES):
        payload, meta = agent_tools._dispatch(tool_name, {"reason": "test"})

        assert meta == {"cache": "n/a"}
        assert payload["status"] == "pending_approval_created"
        assert payload["approval_id"] == len(calls)
        assert payload["entity_type"] == f"{tool_name}_entity"

    assert [call[0] for call in calls] == sorted(PROPOSAL_TOOL_NAMES)
    assert all(call[2].actor_type == "agent" for call in calls)
    assert all(call[2].source_type == "agent" for call in calls)


def test_agent_proposal_tools_create_action_backed_approvals(agent_proposal_state):
    core_db = agent_proposal_state.core_db
    digest = agent_proposal_state.digests.save_digest(
        "# Digest\n\n## Movers\n- MU update\n", filename="05012026_digest.md"
    )

    for tool_name, (args, expected_action_id, expected_entity_type) in _proposal_tool_cases(digest["id"]).items():
        payload, meta = agent_tools._dispatch(tool_name, args)
        approval = core_db.get_pending_approval(payload["approval_id"])

        assert meta == {"cache": "n/a"}
        assert payload["status"] == "pending_approval_created"
        assert payload["entity_type"] == expected_entity_type
        assert approval["status"] == "pending"
        assert approval["application_status"] == "pending"
        assert approval["entity_type"] == expected_entity_type
        assert approval["action_id"] == expected_action_id
        assert approval["action_schema_name"] == expected_action_id
        expected_schema_version = (
            3
            if expected_action_id == "update_portfolio_positions"
            else 2
            if expected_action_id == "update_hedge_positions"
            else 1
        )
        assert approval["action_schema_version"] == expected_schema_version
        assert approval["action_input_hash"]
        assert approval["source_type"] == "agent"
        assert approval["source_id"] == "admin"

        proposal_runs = core_db.get_action_runs(f"{expected_action_id}:propose")
        assert proposal_runs
        assert proposal_runs[-1]["status"] == "succeeded"


@pytest.mark.parametrize(
    ("tool_name", "args", "match"),
    [
        ("propose_action_item", {"description": "Review", "reason": "Missing type"}, "action_type"),
        (
            "propose_thesis_status_change",
            {"ticker": "MU", "new_status": "paused", "reason": "Bad status"},
            "Invalid status",
        ),
        (
            "propose_catalyst",
            {"ticker": "bad ticker", "description": "HBM ramp", "reason": "Bad ticker"},
            "Invalid ticker format",
        ),
        (
            "propose_news_digest_delete",
            {"digest_id": "2026-05-01-missing", "reason": "Not present"},
            "Unknown news digest id",
        ),
    ],
)
def test_agent_proposal_tools_reject_invalid_input(agent_proposal_state, tool_name, args, match):
    with pytest.raises(ActionValidationError, match=match):
        agent_tools._dispatch(tool_name, args)

    assert agent_proposal_state.core_db.get_pending_approvals(status=None) == []


def test_agent_proposal_source_lineage_preserves_actor_metadata(agent_proposal_state):
    core_db = agent_proposal_state.core_db
    actor = agent_actor(admin_actor("alice"))

    payload, _meta = agent_tools._dispatch(
        "propose_action_item",
        {
            "ticker": "mu",
            "description": "Review sizing",
            "action_type": "review",
            "reason": "Risk changed",
        },
        actor=actor,
    )

    approval = core_db.get_pending_approval(payload["approval_id"])
    assert approval["source_type"] == "agent"
    assert approval["source_id"] == "alice"

    proposal_run = core_db.get_action_runs("create_action_item:propose")[0]
    assert proposal_run["actor_type"] == "agent"
    assert proposal_run["actor_id"] == "agent:alice"
    assert proposal_run["source_type"] == "agent"
    assert proposal_run["source_id"] == "alice"

    approval_audit = core_db.get_audit_events(action_name="approval.created", limit=5)[0]
    assert approval_audit["source_lineage"]["source_type"] == "agent"
    assert approval_audit["source_lineage"]["source_id"] == "alice"
    assert approval_audit["source_lineage"]["action_input_hash"] == approval["action_input_hash"]

    domain_audit = core_db.get_audit_events(action_name="domain.action.succeeded", limit=5)[0]
    assert domain_audit["source_lineage"]["source_type"] == "agent"
    assert domain_audit["source_lineage"]["source_id"] == "alice"
    assert domain_audit["metadata"]["action_id"] == "create_action_item"


def test_agent_provider_tool_definitions_are_in_parity():
    anthropic_names = {tool["name"] for tool in agent_router.ANTHROPIC_TOOL_DEFINITIONS}
    openai_names = {tool["name"] for tool in agent_router.OPENAI_TOOL_DEFINITIONS}
    registry_names = {cap.name for cap in agent_tools.AGENT_CAPABILITIES}

    assert anthropic_names == registry_names
    assert openai_names == registry_names
    assert all("input_schema" in tool for tool in agent_router.ANTHROPIC_TOOL_DEFINITIONS)
    assert all(tool["type"] == "function" and "parameters" in tool for tool in agent_router.OPENAI_TOOL_DEFINITIONS)


def test_agent_selector_finds_new_full_app_capabilities():
    cases = {
        "show me the commodity proxy screener": "get_commodity_research",
        "pull the latest portfolio news digests": "get_portfolio_news",
        "run a DCF valuation for NVDA": "run_dcf_valuation",
        "get SONO financials": "get_financials",
        "run the FX model for EURUSD": "run_fx_model",
        "run a short screener": "run_short_screen",
        "chart MU over 2Y": "run_chart",
        "update my portfolio positions": "propose_portfolio_positions_update",
        "Have any of the catalysts for Apollo played out?": "search_web",
    }

    for prompt, expected_tool in cases.items():
        selected = agent_router._select_tool_names(prompt)
        assert agent_router._is_data_seeking(prompt)
        assert expected_tool in selected
        assert "search_agent_capabilities" in selected


def test_agent_selector_includes_catalyst_proposal_for_thesis_catalyst_creation():
    selected = agent_router._select_tool_names(
        "Take the catalysts from the META thesis and generate action items to create the catalysts"
    )

    assert "get_thesis" in selected
    assert "get_catalysts" in selected
    assert "propose_catalyst" in selected
    assert "search_agent_capabilities" in selected


def test_agent_capability_search_returns_fallback_matches():
    result = agent_tools.search_agent_capabilities("commodity proxy screener", top_k=5)
    names = [row["name"] for row in result["matches"]]

    assert "get_commodity_research" in names


def test_agent_capabilities_endpoint(auth_client):
    resp = auth_client.get("/api/v1/agent/capabilities")

    assert resp.status_code == 200
    payload = resp.json()
    names = {row["name"] for row in payload["capabilities"]}
    assert payload["count"] == len(payload["capabilities"])
    assert "get_commodity_research" in names
    assert "run_fx_model" in names


def test_get_catalysts_tool_returns_object_payload(monkeypatch):
    from ontology.runtime_read_service import OntologyRuntimeReadService

    rows = [
        {
            "id": 1,
            "ticker": "ZZZZ",
            "description": "Origination recovery",
            "status": "pending",
        }
    ]
    seen = {}

    def fake_catalysts(self, ticker, status=None, limit=100):
        seen["ticker"] = ticker
        return rows

    monkeypatch.setattr(OntologyRuntimeReadService, "catalysts", fake_catalysts)

    payload = json.loads(agent_tools.execute_tool("get_catalysts", {"ticker": "zzzz"}))

    assert payload.get("error") is None
    assert payload["ticker"] == "ZZZZ"
    assert payload["catalysts"] == rows
    assert payload["count"] == 1
    assert payload["_meta"]["status"] == "ok"
    assert seen["ticker"] == "ZZZZ"


def test_get_catalysts_tool_empty_result_is_not_blocked(monkeypatch):
    from ontology.runtime_read_service import OntologyRuntimeReadService

    monkeypatch.setattr(OntologyRuntimeReadService, "catalysts", lambda self, ticker, status=None, limit=100: [])

    payload = json.loads(agent_tools.execute_tool("get_catalysts", {"ticker": "ZZZZ"}))

    assert payload.get("error") is None
    assert payload["ticker"] == "ZZZZ"
    assert payload["catalysts"] == []
    assert payload["count"] == 0
    assert payload["_meta"]["status"] == "ok"


def test_get_thesis_tool_reads_state_storage(monkeypatch, tmp_path):
    from ontology.runtime_read_service import OntologyRuntimeReadService

    seen: dict[str, object] = {}

    monkeypatch.setattr(agent_tools, "_THESES_DIR", tmp_path / "investment_theses")
    monkeypatch.setattr(
        OntologyRuntimeReadService, "thesis", lambda self, ticker: {"ticker": ticker, "status": "active"}
    )

    def fake_exists(local_path, gcs_key):
        seen["exists_path"] = local_path
        seen["exists_key"] = gcs_key
        return True

    def fake_read(local_path, gcs_key, *, encoding="utf-8"):
        seen["read_path"] = local_path
        seen["read_key"] = gcs_key
        seen["encoding"] = encoding
        return "# META\n\n## Key Catalysts\n- **AI ads:** Monetization improves.\n"

    monkeypatch.setattr("api.state_storage.exists_text", fake_exists)
    monkeypatch.setattr("api.state_storage.read_text", fake_read)

    payload = json.loads(agent_tools.execute_tool("get_thesis", {"ticker": "meta"}))

    assert payload.get("error") is None
    assert payload["ticker"] == "META"
    assert "AI ads" in payload["content"]
    assert payload["source_key"] == "live/theses/META.md"
    assert seen["exists_key"] == "live/theses/META.md"
    assert seen["read_key"] == "live/theses/META.md"


def test_get_catalysts_tool_adds_markdown_fallback_candidates(monkeypatch, tmp_path):
    from ontology.runtime_read_service import OntologyRuntimeReadService

    monkeypatch.setattr(agent_tools, "_THESES_DIR", tmp_path / "investment_theses")
    monkeypatch.setattr(OntologyRuntimeReadService, "catalysts", lambda self, ticker, status=None, limit=100: [])
    monkeypatch.setattr("api.state_storage.exists_text", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "api.state_storage.read_text",
        lambda *_args, **_kwargs: (
            "# META\n\n"
            "## Key Catalysts\n"
            "- **AI ads:** Llama and ranking tools improve monetization.\n"
            "- **Regulatory approval:** EU signs off on data usage.\n"
        ),
    )

    payload = json.loads(agent_tools.execute_tool("get_catalysts", {"ticker": "META"}))

    assert payload.get("error") is None
    assert payload["count"] == 2
    assert [row["persisted"] for row in payload["catalysts"]] == [False, False]
    assert payload["catalysts"][0]["id"] == "thesis_markdown:META:catalyst:1"
    assert payload["catalysts"][0]["source"] == "thesis_markdown"
    assert payload["catalysts"][0]["provenance"]["section"] == "Key Catalysts"
    assert payload["catalysts"][1]["category"] == "regulatory"


def test_get_catalysts_tool_dedupes_structured_rows_against_markdown(monkeypatch, tmp_path):
    from ontology.runtime_read_service import OntologyRuntimeReadService

    structured = [
        {
            "id": 7,
            "ticker": "META",
            "description": "AI ads: Llama and ranking tools improve monetization.",
            "status": "pending",
        }
    ]
    monkeypatch.setattr(agent_tools, "_THESES_DIR", tmp_path / "investment_theses")
    monkeypatch.setattr(
        OntologyRuntimeReadService, "catalysts", lambda self, ticker, status=None, limit=100: structured
    )
    monkeypatch.setattr("api.state_storage.exists_text", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "api.state_storage.read_text",
        lambda *_args, **_kwargs: (
            "# META\n\n"
            "## Key Catalysts\n"
            "- **AI ads:** Llama and ranking tools improve monetization.\n"
            "- **Reality Labs discipline:** Capex gets tied to clearer return thresholds.\n"
        ),
    )

    payload = json.loads(agent_tools.execute_tool("get_catalysts", {"ticker": "META"}))

    assert payload["count"] == 2
    assert payload["catalysts"][0] == structured[0]
    assert payload["catalysts"][1]["description"].startswith("Reality Labs discipline:")
    assert payload["catalysts"][1]["persisted"] is False


def test_get_dossier_tool_adds_markdown_catalyst_fallback(monkeypatch, tmp_path):
    import api.routers.dossier as dossier_router

    monkeypatch.setattr(agent_tools, "_THESES_DIR", tmp_path / "investment_theses")
    monkeypatch.setattr(dossier_router, "get_dossier", lambda ticker: {"ticker": ticker, "catalysts": []})
    monkeypatch.setattr("api.state_storage.exists_text", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "api.state_storage.read_text",
        lambda *_args, **_kwargs: "# META\n\n## Key Catalysts\n- **AI ads:** Monetization improves.\n",
    )

    payload, meta = agent_tools._dispatch("get_dossier", {"ticker": "meta"})

    assert meta == {"cache": "n/a"}
    assert payload["ticker"] == "META"
    assert payload["catalysts"][0]["description"] == "AI ads: Monetization improves."
    assert payload["catalysts"][0]["persisted"] is False


@pytest.mark.parametrize(
    ("tool_name", "args", "method_name", "collection_key"),
    [
        ("get_kill_conditions", {"ticker": "APO"}, "kill_conditions", "kill_conditions"),
        ("get_action_items", {"ticker": "APO"}, "action_items", "action_items"),
        ("get_watch_triggers", {"ticker": "APO"}, "watch_triggers", "watch_triggers"),
        ("get_pending_approvals", {"ticker": "APO"}, "approvals", "pending_approvals"),
        ("get_workflow_history", {"ticker": "APO"}, "workflow_runs", "workflow_runs"),
    ],
)
def test_control_plane_list_tools_return_object_payloads(monkeypatch, tool_name, args, method_name, collection_key):
    from ontology.runtime_read_service import OntologyRuntimeReadService

    rows = [{"id": f"{collection_key}:1", "ticker": "APO"}]

    def fake_reader(self, *reader_args, **reader_kwargs):
        return rows

    monkeypatch.setattr(OntologyRuntimeReadService, method_name, fake_reader)

    payload = json.loads(agent_tools.execute_tool(tool_name, args))

    assert payload.get("error") is None
    assert payload[collection_key] == rows
    assert payload["count"] == 1
    assert payload["_meta"]["status"] == "ok"
    assert payload.get("type") != "tool_output_validation"


def test_get_portfolio_tool_includes_full_position_context_and_short_semantics(monkeypatch):
    dates = pd.date_range("2026-04-24", periods=2, freq="D")
    raw = {
        "positions": {
            "OKLO": pd.Series([10.0, 9.0], index=dates),
            "CRWD": pd.Series([400.0, 315.0], index=dates),
        },
        "analytics": {
            "per_position": {
                "OKLO": {
                    "current_price": 9.0,
                    "unrealized_pnl_pct": 10.0,
                    "unrealized_pnl_dollar": 1.0,
                    "weekly_return_pct": 10.0,
                    "monthly_return_pct": 8.0,
                    "weekly_contribution_pct": 2.5,
                    "monthly_contribution_pct": 2.0,
                    "current_notional": 900.0,
                    "weight": 0.25,
                    "drawdown_from_52w_pct": 0.0,
                },
                "CRWD": {
                    "current_price": 315.0,
                    "unrealized_pnl_pct": 5.0,
                    "weekly_return_pct": 5.0,
                    "monthly_return_pct": 4.0,
                    "weekly_contribution_pct": 1.0,
                    "monthly_contribution_pct": 0.8,
                    "current_notional": 3150.0,
                    "weight": 0.2,
                    "drawdown_from_52w_pct": 0.0,
                },
            },
            "portfolio": {"weekly_portfolio_return_pct": 3.5},
        },
        "timeframe": "Daily",
        "timestamp": datetime(2026, 4, 30, tzinfo=UTC),
    }
    positions = [
        {
            "ticker": "OKLO",
            "asset": "equity",
            "direction": "short",
            "contrarian": True,
            "conviction": 4,
            "cost_basis": 10.0,
            "shares": 100.0,
            "role": "position",
        },
        {
            "ticker": "CRWD",
            "asset": "equity",
            "direction": "long",
            "contrarian": False,
            "conviction": 3,
            "cost_basis": 300.0,
            "shares": 10.0,
            "role": "position",
        },
    ]

    monkeypatch.setitem(
        sys.modules, "portfolio.portfolio_dashboard", SimpleNamespace(get_data=lambda timeframe="Daily": raw)
    )
    monkeypatch.setitem(
        sys.modules,
        "portfolio.portfolio_db",
        SimpleNamespace(get_positions=lambda include_hedges=False: positions),
    )
    monkeypatch.setattr("api.agent_tools.get_cached", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("api.agent_tools.set_cached", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("api.portfolio_settings.get_portfolio_book_size", lambda: 10_000.0)

    payload = json.loads(agent_tools.execute_tool("get_portfolio", {}))
    rows = {row["ticker"]: row for row in payload["positions"]}
    oklo = rows["OKLO"]

    assert payload["summary"]["book_size"] == 10_000.0
    assert oklo["direction"] == "short"
    assert oklo["cost_basis"] == 10.0
    assert oklo["shares"] == 100.0
    assert oklo["quantity"] == 100.0
    assert oklo["conviction"] == 4
    assert oklo["asset"] == "equity"
    assert oklo["contrarian"] is True
    assert oklo["role"] == "position"
    assert "first_date" not in oklo
    assert "first_price" not in oklo
    assert "raw_price_return_pct" not in oklo
    assert oklo["weekly_return_pct"] == 10.0
    assert oklo["current_notional"] == 900.0
    assert oklo["weight"] == 0.09
    assert oklo["weight_of_book"] == 0.09
    assert oklo["gross_position_share"] == 0.25
    assert payload["semantics"]["entry_history_available"] is False
    assert "not first entry price" in payload["semantics"]["cost_basis"]
    assert payload["semantics"]["short_price_declines_are_favorable"] is True
    assert "book size" in payload["semantics"]["weight"]

    crwd = rows["CRWD"]
    assert crwd["cost_basis"] == 300.0
    assert crwd["current_price"] == 315.0
    assert crwd["weight"] == 0.315
    assert crwd["weight_of_book"] == 0.315
    assert crwd["gross_position_share"] == 0.2
    assert "first_date" not in crwd
    assert "first_price" not in crwd
    assert "raw_price_return_pct" not in crwd


def test_sentiment_snapshot_picks_latest_by_date():
    today = datetime.now(UTC).date().isoformat()
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()

    surveys = {
        "aaii": [
            {"date": today, "bull": 40.0, "bear": 30.0, "neutral": 30.0, "spread": 10.0},
            {"date": yesterday, "bull": 20.0, "bear": 50.0, "neutral": 30.0, "spread": -30.0},
        ],
        "naaim": [
            {"date": yesterday, "exposure": 40.0},
            {"date": today, "exposure": 55.0},
        ],
        "errors": {},
    }
    volatility = [
        {"date": yesterday, "vix": 18.2, "vxn": 20.1, "vvix": 93.0},
        {"date": today, "vix": 16.2, "vxn": 18.4, "vvix": 88.0},
    ]
    put_call = {"equity": {"ratio": 1.02, "calls": 1000, "puts": 1020, "as_of": today}}

    snapshot = agent_tools._build_agent_sentiment_snapshot(put_call, surveys, volatility)

    assert snapshot["latest"]["surveys"]["aaii"]["date"] == today
    assert snapshot["latest"]["surveys"]["naaim"]["date"] == today
    assert snapshot["latest"]["volatility"]["date"] == today
    assert snapshot["quality"]["ok"] is True


def test_sentiment_snapshot_fail_closed_on_stale_or_inconsistent_inputs():
    stale = (datetime.now(UTC).date() - timedelta(days=60)).isoformat()
    surveys = {
        "aaii": [{"date": stale, "bull": 60.0, "bear": 30.0, "neutral": 30.0, "spread": 30.0}],
        "naaim": [{"date": stale, "exposure": 100.0}],
        "errors": {"naaim": "timeout"},
    }
    volatility = [{"date": stale, "vix": 19.0, "vxn": 22.0, "vvix": 95.0}]
    put_call = {"equity": {"ratio": 1.11, "calls": 1000, "puts": 1110, "as_of": stale}}

    snapshot = agent_tools._build_agent_sentiment_snapshot(put_call, surveys, volatility)

    assert snapshot["quality"]["ok"] is False
    assert snapshot["quality"]["allow_sentiment_conclusion"] is False
    issues = " | ".join(snapshot["quality"]["issues"]).lower()
    assert "stale" in issues
    assert "inconsistent" in issues or "source error" in issues


def test_sentiment_snapshot_parity_from_normalized_shape():
    today = datetime.now(UTC).date().isoformat()
    two_days_ago = (datetime.now(UTC).date() - timedelta(days=2)).isoformat()
    surveys = {
        "aaii": [
            {"date": two_days_ago, "bull": 35.0, "bear": 45.0, "neutral": 20.0, "spread": -10.0},
            {"date": today, "bull": 45.0, "bear": 30.0, "neutral": 25.0, "spread": 15.0},
        ],
        "naaim": [
            {"date": today, "exposure": 72.0},
            {"date": two_days_ago, "exposure": 51.0},
        ],
        "errors": {},
    }
    volatility = [
        {"date": two_days_ago, "vix": 17.0, "vxn": 20.0, "vvix": 92.0},
        {"date": today, "vix": 15.5, "vxn": 18.8, "vvix": 87.2},
    ]
    put_call = {
        "equity": {"ratio": 0.98, "calls": 2200, "puts": 2150, "as_of": today},
        "spy": {"ratio": 1.11, "calls": 900, "puts": 999, "as_of": today},
    }

    snapshot = agent_tools._build_agent_sentiment_snapshot(put_call, surveys, volatility)

    assert snapshot["latest"]["put_call"]["equity"]["ratio"] == 0.98
    assert snapshot["latest"]["surveys"]["aaii"]["spread"] == 15.0
    assert snapshot["latest"]["surveys"]["naaim"]["exposure"] == 72.0
    assert snapshot["latest"]["volatility"]["vix"] == 15.5


def test_run_search_web_uses_unrestricted_web_search(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    calls: list[dict] = []

    def fake_call_llm_text(**kwargs):
        calls.append(kwargs)
        return "ok summary", [("Example", "https://example.com")], object()

    monkeypatch.setattr("llm_utils.call_llm_text", fake_call_llm_text)

    result = agent_tools._run_search_web("microsoft antitrust")

    assert result["summary"] == "ok summary"
    assert result["citation_count"] == 1
    assert len(calls) == 1
    assert calls[0]["enable_web_search"] is True
    assert "allowed_domains" not in calls[0]
