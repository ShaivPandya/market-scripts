from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from api import agent_tools
from api.routers import agent as agent_router


def test_cached_singleflight_fetches_once(monkeypatch):
    store: dict[str, object] = {}

    monkeypatch.setattr("api.agent_tools.get_cached", lambda _cache, key: store.get(key))
    monkeypatch.setattr("api.agent_tools.set_cached", lambda _cache, key, value: store.__setitem__(key, value))

    calls = 0
    calls_lock = threading.Lock()

    def loader():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return {"value": 1}

    cache_token = object()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(agent_tools._cached_singleflight, cache_token, "k", loader) for _ in range(4)]
        results = [f.result() for f in futs]

    assert calls == 1
    assert all(v[0] == {"value": 1} for v in results)
    assert {"miss_fetch", "miss_wait"} & {v[1] for v in results}


def test_fetch_with_cache_force_refresh_bypasses_cached_value(monkeypatch):
    store: dict[str, object] = {}

    monkeypatch.setattr("api.agent_tools.get_cached", lambda _cache, key: store.get(key))
    monkeypatch.setattr("api.agent_tools.set_cached", lambda _cache, key, value: store.__setitem__(key, value))

    calls = 0

    def loader():
        nonlocal calls
        calls += 1
        return {"value": calls}

    cache_token = object()
    first, first_meta = agent_tools._fetch_with_cache(cache_token, "k", loader)
    second, second_meta = agent_tools._fetch_with_cache(cache_token, "k", loader)
    refreshed, refreshed_meta = agent_tools._fetch_with_cache(cache_token, "k", loader, force_refresh=True)

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert refreshed == {"value": 2}
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

    raw = agent_tools.execute_tool("dummy_tool", {})
    payload = json.loads(raw)

    assert isinstance(payload, dict)
    assert "_meta" in payload
    assert payload["_meta"]["tool"] == "dummy_tool"
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
        "get_research_notes",
        "get_weekly_report",
        "search_agent_capabilities",
    }

    assert expected <= names
    assert len(names) == len(agent_tools.AGENT_CAPABILITIES)
    assert all(cap.category and cap.access_mode and cap.aliases for cap in agent_tools.AGENT_CAPABILITIES)
    assert all(cap.schema_safe for cap in agent_tools.AGENT_CAPABILITIES)


def test_agent_capability_registry_does_not_expose_direct_mutations():
    names = {cap.name for cap in agent_tools.AGENT_CAPABILITIES}

    forbidden_direct_tools = {
        "update_portfolio_positions",
        "update_hedge_positions",
        "save_thesis",
        "create_research_note",
        "delete_portfolio_news_digest",
        "approve_item",
        "reject_item",
        "bulk_approve",
        "bulk_reject",
    }
    proposal_tools = {
        "propose_portfolio_positions_update",
        "propose_hedge_positions_update",
        "propose_thesis_content_update",
        "propose_research_note",
        "propose_news_digest_delete",
    }

    assert names.isdisjoint(forbidden_direct_tools)
    assert proposal_tools <= names


def test_agent_proposal_tools_create_action_backed_approvals(tmp_path, monkeypatch):
    import portfolio.core_db as core_db

    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)

    payload, meta = agent_tools._dispatch(
        "propose_catalyst",
        {
            "ticker": "mu",
            "description": "HBM ramp",
            "category": "fundamental",
            "reason": "Track material catalyst",
        },
    )

    approval = core_db.get_pending_approval(payload["approval_id"])
    assert meta == {"cache": "n/a"}
    assert payload["entity_type"] == "catalyst"
    assert approval["action_id"] == "create_catalyst"
    assert approval["action_schema_version"] == 1
    assert approval["proposed_change"]["ticker"] == "MU"
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)


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
    }

    for prompt, expected_tool in cases.items():
        selected = agent_router._select_tool_names(prompt)
        assert agent_router._is_data_seeking(prompt)
        assert expected_tool in selected
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


def test_get_portfolio_tool_includes_full_position_context_and_short_semantics(monkeypatch):
    dates = pd.date_range("2026-04-24", periods=2, freq="D")
    raw = {
        "positions": {
            "OKLO": pd.Series([10.0, 9.0], index=dates),
            "CRWD": pd.Series([300.0, 315.0], index=dates),
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

    payload = json.loads(agent_tools.execute_tool("get_portfolio", {}))
    rows = {row["ticker"]: row for row in payload["positions"]}
    oklo = rows["OKLO"]

    assert oklo["direction"] == "short"
    assert oklo["cost_basis"] == 10.0
    assert oklo["shares"] == 100.0
    assert oklo["quantity"] == 100.0
    assert oklo["conviction"] == 4
    assert oklo["asset"] == "equity"
    assert oklo["contrarian"] is True
    assert oklo["role"] == "position"
    assert oklo["raw_price_return_pct"] == -10.0
    assert oklo["weekly_return_pct"] == 10.0
    assert payload["semantics"]["short_price_declines_are_favorable"] is True


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


def test_extract_inaccessible_domains_parses_error_message():
    err = RuntimeError(
        "Error code: 400 - {'message': \"The following domains are not accessible to our user agent: "
        "['ft.com', 'WSJ.com']\"}"
    )
    blocked = agent_tools._extract_inaccessible_domains(err)
    assert blocked == {"ft.com", "wsj.com"}


def test_run_search_web_prunes_blocked_domains_and_retries(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    calls: list[list[str]] = []

    def fake_call_llm_text(*, allowed_domains=None, **_kwargs):
        domains = list(allowed_domains or [])
        calls.append(domains)
        if len(calls) == 1:
            raise RuntimeError(
                "Error code: 400 - {'message': \"The following domains are not accessible to our user agent: "
                "['axios.com']\"}"
            )
        return "ok summary", [("Example", "https://example.com")], object()

    monkeypatch.setattr("llm_utils.call_llm_text", fake_call_llm_text)

    result = agent_tools._run_search_web("microsoft antitrust")

    assert result["summary"] == "ok summary"
    assert result["citation_count"] == 1
    assert len(calls) == 2
    assert "axios.com" in calls[0]
    assert "axios.com" not in calls[1]
