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
    end = pd.Timestamp.now().normalize()
    if end.dayofweek >= 5:
        end = end - pd.offsets.BDay(1)
    dates = pd.bdate_range(end=end, periods=periods)
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


def _fake_fetch_all_payload(*, macro_status: str = "ok", degraded_proxy: bool = True):
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

    if degraded_proxy:
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
                valid_contracts=10,
                warning_count=0,
                prompt_to_m3=-1.0,
                prompt_to_m6=-1.8,
                prompt_to_m12=-2.5,
            ),
            "Dutch TTF Gas": _curve_payload(
                valid_contracts=5,
                warning_count=0,
                prompt_to_m3=None,
                prompt_to_m6=None,
                prompt_to_m12=None,
                shape="N/A",
            ),
        }
    else:
        curves = {
            "WTI Crude Oil": _curve_payload(valid_contracts=12, warning_count=0),
            "Brent Crude Oil": _curve_payload(valid_contracts=10, warning_count=0),
            "Natural Gas": _curve_payload(
                valid_contracts=10,
                warning_count=0,
                prompt_to_m3=-1.0,
                prompt_to_m6=-1.8,
                prompt_to_m12=-2.5,
            ),
            "Dutch TTF Gas": _curve_payload(
                valid_contracts=10,
                warning_count=0,
                prompt_to_m3=-0.8,
                prompt_to_m6=-1.4,
                prompt_to_m12=-2.0,
                shape="Backwardation",
            ),
        }

    macro = {
        "status": macro_status,
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

    eia_dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=12, freq="MS")
    eia_df = pd.DataFrame({"date": eia_dates, "value": [100.0 + i * 0.1 for i in range(len(eia_dates))]})

    return daily, monthly, curves, macro, eia_df


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


def test_signal_conviction_ignores_na_sources_and_macro_overlay():
    from commodities.commodity_research import assign_signal_conviction

    signal_conviction = assign_signal_conviction(
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

    assert signal_conviction == "high"


def test_proxy_score_shrinks_toward_neutral_when_coverage_is_missing():
    from commodities.commodity_research import _compute_composite

    proxy_score, proxy_coverage_ratio, effective_weights, observed_proxy_score = _compute_composite(
        {
            "trend": 80.0,
            "relative_strength": 80.0,
            "acceleration": None,
            "curve_structure": None,
            "market_stress_overlay": 90.0,
        },
        ["trend", "relative_strength", "acceleration"],
    )

    assert observed_proxy_score == 80.0
    assert proxy_coverage_ratio < 1.0
    assert proxy_score is not None
    assert proxy_score < observed_proxy_score
    assert "market_stress_overlay" not in effective_weights


def test_build_research_splits_proxy_and_fundamental_sections(monkeypatch):
    import commodities.commodity_research as cr_mod

    monkeypatch.setattr(cr_mod, "_fetch_all", _fake_fetch_all_payload)

    data = cr_mod.build_commodity_research()
    commodities = {item["commodity"]: item for item in data["commodities"]}

    gold = commodities["Gold"]
    wti = commodities["WTI Crude Oil"]
    brent = commodities["Brent Crude Oil"]
    ttf = commodities["Dutch TTF Gas"]

    assert data["schema_version"] == 3
    assert data["status"] == "degraded"
    assert "ideas" not in data
    assert data["methodology"]["proxy_signals"]["name"] == "Commodity Proxy Screener"
    assert "limitations" in data["methodology"]["proxy_signals"]
    assert "current_status" in data["methodology"]["fundamental_inputs"]

    assert "composite_score" not in gold
    assert "confidence" not in gold
    assert gold["proxy_signals"]["bias"] in ("bullish", "bearish", "neutral")
    assert gold["proxy_signals"]["signal_conviction"] in ("high", "medium", "low")
    assert gold["fundamental_inputs"]["coverage_status"] == "unavailable"
    assert gold["fundamental_inputs"]["available_inputs"] == []

    assert gold["proxy_signals"]["data_quality"]["curve"] == "n/a"
    assert gold["proxy_signals"]["factors"]["curve_structure"]["quality"] == "n/a"

    assert wti["proxy_signals"]["factors"]["curve_structure"]["quality"] == "ok"
    assert wti["proxy_signals"]["factors"]["curve_structure"]["included_in_composite"] is True

    assert brent["proxy_signals"]["factors"]["curve_structure"]["quality"] == "degraded"
    assert brent["proxy_signals"]["signal_conviction"] in ("medium", "low")

    assert ttf["proxy_signals"]["factors"]["curve_structure"]["quality"] == "missing"
    assert ttf["proxy_signals"]["factors"]["curve_structure"]["included_in_composite"] is False

    assert wti["proxy_signals"]["factors"]["market_stress_overlay"]["included_in_composite"] is False
    assert wti["proxy_signals"]["factors"]["market_stress_overlay"]["effective_weight"] == 0.0

    assert data["summary"]["strongest_bullish_bias"] is not None
    assert "top_long" not in data["summary"]
    assert "top_short" not in data["summary"]
    assert "data_health" not in data["summary"]
    assert data["summary"]["fundamental_coverage"]["energy"]["coverage_status"] == "unavailable"
    assert data["summary"]["fundamental_coverage"]["metals"]["coverage_status"] == "unavailable"

    assert all(item["fundamental_inputs"]["coverage_status"] == "unavailable" for item in data["commodities"])


def test_status_is_not_degraded_solely_for_unavailable_fundamentals(monkeypatch):
    import commodities.commodity_research as cr_mod

    monkeypatch.setattr(cr_mod, "_fetch_all", lambda: _fake_fetch_all_payload(degraded_proxy=False))

    data = cr_mod.build_commodity_research()

    assert data["status"] == "ok"
    assert all(item["fundamental_inputs"]["coverage_status"] == "unavailable" for item in data["commodities"])


def test_router_payload_uses_v3_schema(auth_client, monkeypatch):
    import api.routers.commodity_research as router_mod

    def fake_build():
        return {
            "schema_version": 3,
            "status": "ok",
            "timestamp": "2026-03-20T12:00:00",
            "methodology": {
                "proxy_signals": {
                    "name": "Commodity Proxy Screener",
                    "note": "Test note",
                    "limitations": "Proxy-only test payload",
                    "ranking_mode": "proxy_rank_v3",
                },
                "fundamental_inputs": {
                    "coverage_policy": "Only real fundamentals are shown.",
                    "current_status": "No real commodity-specific inputs are currently available.",
                },
            },
            "macro_overlay": {
                "label": "risk-off",
                "score": 72.0,
                "forward_outlook": "opportunity",
                "as_of": "2026-03-20",
                "status": "ok",
                "quality": "ok",
            },
            "commodities": [
                {
                    "commodity": "Gold",
                    "ticker": "GC=F",
                    "sector": "metals",
                    "spot_price": 2650.0,
                    "returns": {"1m": 3.2, "3m": 8.1, "12m": 22.5},
                    "price_series": [],
                    "proxy_signals": {
                        "proxy_score": 61.8,
                        "observed_proxy_score": 61.8,
                        "proxy_coverage_ratio": 1.0,
                        "bias": "bullish",
                        "signal_conviction": "medium",
                        "factors": {},
                        "rationale": ["Test bullet"],
                        "data_quality": {"prices_daily": "ok", "curve": "n/a"},
                    },
                    "fundamental_inputs": {
                        "coverage_status": "unavailable",
                        "coverage_note": "No real fundamental inputs yet.",
                        "available_inputs": [],
                    },
                }
            ],
            "summary": {
                "strongest_bullish_bias": {"commodity": "Gold", "proxy_score": 61.8},
                "strongest_bearish_bias": None,
                "proxy_data_health": {"ok": 1, "degraded": 0, "missing": 0},
                "fundamental_coverage": {
                    "metals": {
                        "coverage_status": "unavailable",
                        "coverage_note": "No real fundamental inputs yet.",
                        "available_inputs": [],
                    },
                    "energy": {
                        "coverage_status": "unavailable",
                        "coverage_note": "No real fundamental inputs yet.",
                        "available_inputs": [],
                    },
                },
            },
        }

    monkeypatch.setattr(router_mod, "get_or_set_cached", lambda _cache, _key, loader, **_kwargs: loader())

    import commodities.commodity_research as cr_mod

    monkeypatch.setattr(cr_mod, "build_commodity_research", fake_build)

    resp = auth_client.get("/api/commodity-research")
    assert resp.status_code == 200
    data = resp.json()

    assert data["schema_version"] == 3
    assert "commodities" in data
    assert "ideas" not in data
    assert "proxy_signals" in data["methodology"]
    assert "fundamental_inputs" in data["methodology"]
    assert "strongest_bullish_bias" in data["summary"]
    assert "top_long" not in data["summary"]
    assert "top_short" not in data["summary"]
    assert "composite_score" not in data["commodities"][0]
    assert "confidence" not in data["commodities"][0]
