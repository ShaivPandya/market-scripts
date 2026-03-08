from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from api import agent_tools


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
