from __future__ import annotations

import json
from datetime import datetime, timedelta


def test_yield_curve_cache_refreshes_when_latest_close_advanced_under_ttl(monkeypatch, tmp_path):
    from government_bonds import yield_curve as yc

    cache_path = tmp_path / "yield_curve_90.json"
    cached = {"countries": [{"code": "US", "as_of_date": "2000-01-01"}], "lookback_days": 90}
    yc._write_cache(
        path=cache_path,
        payload=cached,
        lookback_days=90,
        as_of_date="2000-01-01",
        fetched_at=(datetime.now() - timedelta(hours=2)).isoformat(),
    )

    monkeypatch.setattr(yc, "_cache_path", lambda _lookback_days: cache_path)
    monkeypatch.setattr(yc, "_latest_market_close_date", lambda: "2099-01-01")
    monkeypatch.setattr(yc, "_build_fred_client", lambda: (None, None))
    monkeypatch.setattr(
        yc,
        "_build_country_curve",
        lambda country_code, country_name, **_kwargs: {
            "code": country_code,
            "name": country_name,
            "as_of_date": "2099-01-01",
            "current": [],
        },
    )

    out = yc.get_data(lookback_days=90)

    assert out["_meta"]["market_cache"]["status"] == "refresh"
    assert out["_meta"]["market_cache"]["stale"] is False
    assert {row["as_of_date"] for row in out["countries"]} == {"2099-01-01"}
    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert stored["as_of_date"] == "2099-01-01"


def test_bond_dashboard_returns_stale_fallback_when_refresh_fails(monkeypatch, tmp_path):
    from government_bonds import bond_dashboard as bd

    cache_path = tmp_path / "bond_dashboard.json"
    monkeypatch.setattr(bd, "_CACHE_PATH", cache_path)
    cached = {
        "timestamp": "2000-01-01T00:00:00",
        "lookback_days": 365,
        "tenors": bd.DASHBOARD_TENORS,
        "country_order": bd.COUNTRY_ORDER,
        "countries": {},
    }
    bd._write_cache(cached, "2000-01-01", fetched_at=(datetime.now() - timedelta(hours=2)).isoformat())

    monkeypatch.setattr(bd, "_latest_market_close_date", lambda: "2099-01-01")
    monkeypatch.setattr(
        bd,
        "_fetch_all_countries",
        lambda: (_ for _ in ()).throw(RuntimeError("bond refresh failed")),
    )

    out = bd.get_data()

    assert out["timestamp"] == cached["timestamp"]
    assert out["_meta"]["market_cache"]["status"] == "stale_fallback"
    assert out["_meta"]["market_cache"]["stale"] is True
