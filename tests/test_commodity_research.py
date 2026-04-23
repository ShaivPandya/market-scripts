"""Tests for the commodity proxy screener."""

from __future__ import annotations

import pandas as pd


def _make_daily_series(
    *,
    base: float = 100.0,
    drift: float = 0.0002,
    boost: float = 0.0,
    boost_days: int = 120,
    periods: int = 1300,
) -> pd.Series:
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=periods)
    price = base
    values: list[float] = []

    for i in range(periods):
        daily_return = drift
        if i >= periods - boost_days:
            daily_return += boost
        price *= 1.0 + daily_return
        values.append(price)

    return pd.Series(values, index=dates, dtype=float)


def _make_monthly_series(daily: pd.Series) -> pd.Series:
    return daily.resample("ME").last().dropna()


def _curve_payload(
    *,
    valid_contracts: int = 12,
    warning_count: int = 0,
    prompt_to_m3: float | None = -2.5,
    prompt_to_m6: float | None = -4.0,
    prompt_to_m12: float | None = -6.0,
    shape: str = "Backwardation",
    newest_date: str | None = None,
) -> dict:
    today = newest_date or pd.Timestamp.now().normalize().date().isoformat()
    warnings = [f"warning-{i}" for i in range(warning_count)]
    return {
        "analysis": {
            "shape": shape,
            "spread_pct": prompt_to_m12,
            "prompt_to_m3_spread_pct": prompt_to_m3,
            "prompt_to_m6_spread_pct": prompt_to_m6,
            "prompt_to_m12_spread_pct": prompt_to_m12,
            "valid_contract_count": valid_contracts,
            "contracts_available": valid_contracts,
            "warning_count": warning_count,
            "newest_valid_contract_date": today,
        },
        "warnings": warnings,
    }


def _fake_fetch_all_payload():
    from commodities.commodities_dashboard import COMMODITIES

    specs = {
        "Gold": {"base": 100.0, "drift": 0.00025, "boost": 0.0019},
        "Silver": {"base": 90.0, "drift": 0.00022, "boost": 0.0010},
        "Copper": {"base": 85.0, "drift": 0.00018, "boost": 0.0008},
        "Platinum": {"base": 80.0, "drift": 0.00015, "boost": 0.0006},
        "Palladium": {"base": 78.0, "drift": 0.00012, "boost": 0.0002},
        "Aluminum": {"base": 70.0, "drift": 0.00010, "boost": 0.0004},
        "WTI Crude Oil": {"base": 75.0, "drift": 0.00024, "boost": 0.0012},
        "Brent Crude Oil": {"base": 77.0, "drift": 0.00022, "boost": 0.0009},
        "Natural Gas": {"base": 50.0, "drift": 0.00005, "boost": 0.0001},
        "Dutch TTF Gas": {"base": 55.0, "drift": 0.00004, "boost": 0.0001},
    }

    daily_prices = {}
    monthly_prices = {}
    for name in COMMODITIES:
        daily = _make_daily_series(**specs[name])
        daily_prices[name] = daily
        monthly_prices[name] = _make_monthly_series(daily)

    curves = {
        "WTI Crude Oil": _curve_payload(valid_contracts=12, warning_count=0),
        "Brent Crude Oil": _curve_payload(
            valid_contracts=7,
            warning_count=1,
            prompt_to_m3=1.2,
            prompt_to_m6=2.4,
            prompt_to_m12=None,
            shape="Contango",
        ),
        "Natural Gas": _curve_payload(
            valid_contracts=10, warning_count=0, prompt_to_m3=-1.0, prompt_to_m6=-1.8, prompt_to_m12=-2.5
        ),
        "Dutch TTF Gas": _curve_payload(
            valid_contracts=5, warning_count=0, prompt_to_m3=None, prompt_to_m6=None, prompt_to_m12=None, shape="N/A"
        ),
    }

    macro = {
        "status": "degraded",
        "as_of": pd.Timestamp.now().normalize().date().isoformat(),
        "regime": {
            "label": "risk-off",
            "score": 72.0,
            "confidence": 0.9,
            "history_percentile": 88.0,
        },
        "forward_outlook": {"label": "opportunity"},
    }

    daily = {"commodities": daily_prices, "timeframe": "ResearchDaily", "timestamp": pd.Timestamp.now()}
    monthly = {"commodities": monthly_prices, "timeframe": "Monthly", "timestamp": pd.Timestamp.now()}
    return daily, monthly, curves, macro


def test_trend_score_uses_own_history_percentile():
    from commodities.commodity_research import _score_trend

    series = _make_daily_series(drift=0.00015, boost=0.0020, boost_days=90)
    score, label = _score_trend(series)

    assert score is not None
    assert score >= 70.0
    assert label in ("strong_up", "moderate_up")


def test_trend_score_requires_min_history():
    from commodities.commodity_research import _score_trend

    series = _make_daily_series(periods=200, drift=0.0002, boost=0.001)
    score, label = _score_trend(series)

    assert score is None
    assert label == "no_data"


def test_acceleration_uses_own_history_percentile():
    from commodities.commodity_research import _score_acceleration

    series = _make_daily_series(drift=0.00012, boost=0.0025, boost_days=45)
    score = _score_acceleration(series)

    assert score is not None
    assert score > 60.0


def test_family_relative_strength_scores_are_family_ranked():
    from commodities.commodity_research import _family_relative_strength_scores

    scores = _family_relative_strength_scores({"Gold": 0.4, "Silver": 0.2, "Copper": -0.1})

    assert scores["Gold"] is not None
    assert scores["Silver"] is not None
    assert scores["Copper"] is not None
    assert scores["Gold"] > scores["Silver"] > scores["Copper"]


def test_curve_quality_states():
    from commodities.commodity_research import _curve_quality

    assert _curve_quality(_curve_payload(valid_contracts=10, warning_count=0)) == "ok"
    assert _curve_quality(_curve_payload(valid_contracts=7, warning_count=1)) == "degraded"
    assert _curve_quality(_curve_payload(valid_contracts=5, warning_count=0)) == "missing"
    assert _curve_quality(None) == "error"


def test_curve_structure_score_prefers_backwardation():
    from commodities.commodity_research import _score_curve_structure

    backwardated = _score_curve_structure(_curve_payload(prompt_to_m3=-3.0, prompt_to_m6=-4.0, prompt_to_m12=-5.0))
    contango = _score_curve_structure(
        _curve_payload(prompt_to_m3=3.0, prompt_to_m6=4.0, prompt_to_m12=5.0, shape="Contango")
    )

    assert backwardated is not None
    assert contango is not None
    assert backwardated > 50.0
    assert contango < 50.0


def test_confidence_ignores_na_sources_and_macro_overlay():
    from commodities.commodity_research import assign_confidence

    confidence = assign_confidence(
        68.0,
        1.0,
        {
            "prices_daily": "ok",
            "prices_monthly": "ok",
            "curve": "n/a",
            "macro_overlay": "degraded",
        },
        "n/a",
    )

    assert confidence == "high"


def test_composite_shrinks_toward_neutral_when_coverage_is_missing():
    from commodities.commodity_research import _compute_composite

    composite, coverage_ratio, effective_weights, observed = _compute_composite(
        {
            "trend": 80.0,
            "relative_strength": 80.0,
            "acceleration": None,
            "curve_structure": None,
            "market_stress_overlay": 90.0,
        },
        ["trend", "relative_strength", "acceleration"],
    )

    assert observed == 80.0
    assert coverage_ratio < 1.0
    assert composite is not None
    assert composite < observed
    assert "market_stress_overlay" not in effective_weights


def test_build_research_handles_curve_quality_and_overlay(monkeypatch):
    import commodities.commodity_research as cr_mod

    monkeypatch.setattr(cr_mod, "_fetch_all", _fake_fetch_all_payload)

    data = cr_mod.build_commodity_research()
    ideas = {idea["commodity"]: idea for idea in data["ideas"]}

    gold = ideas["Gold"]
    wti = ideas["WTI Crude Oil"]
    brent = ideas["Brent Crude Oil"]
    ttf = ideas["Dutch TTF Gas"]

    assert data["schema_version"] == 2
    assert data["methodology"]["name"] == "Commodity Proxy Screener"
    assert data["macro_overlay"]["status"] == "degraded"
    assert "strongest_tailwind" not in data["summary"]
    assert "strongest_headwind" not in data["summary"]

    assert gold["data_quality"]["curve"] == "n/a"
    assert gold["factors"]["curve_structure"]["quality"] == "n/a"

    assert wti["factors"]["curve_structure"]["quality"] == "ok"
    assert wti["factors"]["curve_structure"]["included_in_composite"] is True

    assert brent["factors"]["curve_structure"]["quality"] == "degraded"
    assert brent["confidence"] in ("medium", "low")

    assert ttf["factors"]["curve_structure"]["quality"] == "missing"
    assert ttf["factors"]["curve_structure"]["included_in_composite"] is False

    assert wti["factors"]["market_stress_overlay"]["included_in_composite"] is False
    assert wti["factors"]["market_stress_overlay"]["effective_weight"] == 0.0


def test_router_payload_uses_new_schema(auth_client, monkeypatch):
    import api.routers.commodity_research as router_mod

    def fake_build():
        return {
            "schema_version": 2,
            "status": "ok",
            "timestamp": "2026-03-20T12:00:00",
            "methodology": {
                "name": "Commodity Proxy Screener",
                "note": "Test note",
                "ranking_mode": "proxy_rank_v2",
            },
            "macro_overlay": {
                "label": "risk-off",
                "score": 72.0,
                "forward_outlook": "opportunity",
                "as_of": "2026-03-20",
                "status": "degraded",
                "quality": "degraded",
            },
            "ideas": [
                {
                    "commodity": "Gold",
                    "ticker": "GC=F",
                    "sector": "metals",
                    "spot_price": 2650.0,
                    "returns": {"1m": 3.2, "3m": 8.1, "12m": 22.5},
                    "factors": {},
                    "composite_score": 61.8,
                    "observed_composite_score": 61.8,
                    "coverage_ratio": 1.0,
                    "direction": "long",
                    "confidence": "medium",
                    "rationale": ["Test bullet"],
                    "data_quality": {"prices_daily": "ok", "curve": "n/a"},
                    "price_series": [],
                }
            ],
            "summary": {
                "top_long": {"commodity": "Gold", "score": 61.8},
                "top_short": None,
                "data_health": {"ok": 1, "degraded": 0, "missing": 0},
            },
        }

    monkeypatch.setattr(router_mod, "get_cached", lambda *a, **kw: None)
    monkeypatch.setattr(router_mod, "set_cached", lambda *a, **kw: None)

    import commodities.commodity_research as cr_mod

    monkeypatch.setattr(cr_mod, "build_commodity_research", fake_build)

    resp = auth_client.get("/api/v1/commodity-research")
    assert resp.status_code == 200
    data = resp.json()

    assert data["schema_version"] == 2
    assert "macro_overlay" in data
    assert "methodology" in data
    assert "strongest_tailwind" not in data["summary"]
    assert "strongest_headwind" not in data["summary"]
