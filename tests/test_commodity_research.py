"""Tests for commodity research scoring engine and router."""

from __future__ import annotations

# -- Scoring unit tests -------------------------------------------------------


def test_momentum_score_positive_return():
    from commodities.commodity_research import _score_momentum

    score, label = _score_momentum(ret_1m=5.0, ret_3m=15.0, ret_12m=25.0)
    assert score is not None
    assert score > 0.5
    assert label in ("strong_up", "moderate_up")


def test_momentum_score_negative_return():
    from commodities.commodity_research import _score_momentum

    score, label = _score_momentum(ret_1m=-10.0, ret_3m=-20.0, ret_12m=-30.0)
    assert score is not None
    assert score < 0.5
    assert label in ("strong_down", "moderate_down")


def test_momentum_score_neutral():
    from commodities.commodity_research import _score_momentum

    score, label = _score_momentum(ret_1m=0.0, ret_3m=0.0, ret_12m=0.0)
    assert score is not None
    assert 0.45 <= score <= 0.55
    assert label == "neutral"


def test_momentum_score_partial_missing():
    from commodities.commodity_research import _score_momentum

    score, label = _score_momentum(ret_1m=10.0, ret_3m=None, ret_12m=None)
    assert score is not None
    assert score > 0.5


def test_momentum_score_all_missing():
    from commodities.commodity_research import _score_momentum

    score, label = _score_momentum(ret_1m=None, ret_3m=None, ret_12m=None)
    assert score is None
    assert label == "no_data"


def test_curve_score_backwardation():
    from commodities.commodity_research import _score_curve

    score = _score_curve(shape="Backwardation", spread_pct=-5.0)
    assert score is not None
    assert score > 0.6


def test_curve_score_contango():
    from commodities.commodity_research import _score_curve

    score = _score_curve(shape="Contango", spread_pct=5.0)
    assert score is not None
    assert score < 0.4


def test_curve_score_flat():
    from commodities.commodity_research import _score_curve

    score = _score_curve(shape="Flat", spread_pct=0.0)
    assert score == 0.5


def test_curve_score_na():
    from commodities.commodity_research import _score_curve

    score = _score_curve(shape="N/A", spread_pct=None)
    assert score is None


def test_relative_value_above_median():
    from commodities.commodity_research import _score_relative_value

    score = _score_relative_value(commodity_3m=15.0, median_3m=5.0)
    assert score > 0.5


def test_relative_value_below_median():
    from commodities.commodity_research import _score_relative_value

    score = _score_relative_value(commodity_3m=-5.0, median_3m=10.0)
    assert score < 0.5


def test_composite_reweighting_missing_factors():
    from commodities.commodity_research import _compute_composite

    scores = {
        "momentum": 0.7,
        "relative_value": 0.6,
        "macro": None,
        "supply_demand": 0.5,
        "velocity": 0.4,
    }
    composite, weights = _compute_composite(scores)
    assert composite is not None
    assert abs(sum(weights.values()) - 1.0) < 0.001
    assert "macro" not in weights


def test_composite_all_factors():
    from commodities.commodity_research import _compute_composite

    scores = {
        "momentum": 0.8,
        "relative_value": 0.7,
        "macro": 0.6,
        "supply_demand": 0.5,
        "velocity": 0.4,
    }
    composite, weights = _compute_composite(scores)
    assert composite is not None
    assert 0 < composite < 1
    assert abs(sum(weights.values()) - 1.0) < 0.001


def test_composite_all_none():
    from commodities.commodity_research import _compute_composite

    scores = {
        "momentum": None,
        "relative_value": None,
        "macro": None,
        "supply_demand": None,
        "velocity": None,
    }
    composite, weights = _compute_composite(scores)
    assert composite is None
    assert weights == {}


# -- Direction and confidence -------------------------------------------------


def test_direction_long():
    from commodities.commodity_research import assign_direction

    assert assign_direction(0.75, "strong_up") == "long"
    assert assign_direction(0.65, "moderate_up") == "long"


def test_direction_short():
    from commodities.commodity_research import assign_direction

    assert assign_direction(0.25, "strong_down") == "short"
    assert assign_direction(0.35, "moderate_down") == "short"


def test_direction_watchlist():
    from commodities.commodity_research import assign_direction

    assert assign_direction(0.50, "neutral") == "watchlist"
    assert assign_direction(0.70, "strong_down") == "watchlist"
    assert assign_direction(0.30, "strong_up") == "watchlist"


def test_confidence_high():
    from commodities.commodity_research import assign_confidence

    dq = {"prices_daily": "ok", "prices_monthly": "ok", "curve": "ok", "macro": "ok"}
    assert assign_confidence(0.80, dq) == "high"


def test_confidence_medium():
    from commodities.commodity_research import assign_confidence

    dq = {"prices_daily": "ok", "prices_monthly": "ok", "curve": "n/a", "macro": "error"}
    assert assign_confidence(0.65, dq) == "medium"


def test_confidence_low():
    from commodities.commodity_research import assign_confidence

    dq = {"prices_daily": "missing", "prices_monthly": "missing", "curve": "error", "macro": "error"}
    assert assign_confidence(0.52, dq) == "low"


# -- Supply/demand proxy -----------------------------------------------------


def test_supply_demand_proxy():
    from commodities.commodity_research import _score_supply_demand

    score = _score_supply_demand(trend=0.7, curve=0.6, cross_rank=0.8, macro=0.5)
    assert score is not None
    assert 0 < score < 1


def test_supply_demand_partial():
    from commodities.commodity_research import _score_supply_demand

    score = _score_supply_demand(trend=0.7, curve=None, cross_rank=None, macro=0.5)
    assert score is not None


def test_supply_demand_all_none():
    from commodities.commodity_research import _score_supply_demand

    score = _score_supply_demand(trend=None, curve=None, cross_rank=None, macro=None)
    assert score is None


# -- Date-based return --------------------------------------------------------


def test_date_return_basic():
    import pandas as pd

    from commodities.commodity_research import _date_return

    dates = pd.date_range("2026-01-01", periods=90, freq="B")
    prices = pd.Series([100.0 + i * 0.5 for i in range(90)], index=dates)
    ret = _date_return(prices, 30)
    assert ret is not None
    assert ret > 0


def test_date_return_3m_from_90d():
    """90 calendar days of daily data should still produce a 3M return."""
    import pandas as pd

    from commodities.commodity_research import _date_return

    dates = pd.date_range("2025-12-20", periods=65, freq="B")
    prices = pd.Series([50.0 + i for i in range(65)], index=dates)
    ret = _date_return(prices, 90)
    assert ret is not None
    assert ret > 0


def test_date_return_insufficient():
    import pandas as pd

    from commodities.commodity_research import _date_return

    dates = pd.date_range("2026-03-01", periods=5, freq="B")
    prices = pd.Series([100.0] * 5, index=dates)
    # 90-day lookback with only 5 days of data should return None
    ret = _date_return(prices, 90)
    assert ret is None


# -- Velocity -----------------------------------------------------------------


def test_velocity_accelerating():
    from commodities.commodity_research import _score_velocity

    score = _score_velocity(ret_1m=10.0, ret_3m=5.0)
    assert score is not None
    assert score > 0.5


def test_velocity_decelerating():
    from commodities.commodity_research import _score_velocity

    score = _score_velocity(ret_1m=-5.0, ret_3m=15.0)
    assert score is not None
    assert score < 0.5


# -- Data quality flags -------------------------------------------------------


def test_stale_detection():
    import pandas as pd

    from commodities.commodity_research import _check_staleness

    old_date = pd.Timestamp("2020-01-01")
    series = pd.Series([100.0], index=[old_date])
    assert _check_staleness(series) == "stale"


def test_fresh_detection():
    import pandas as pd

    from commodities.commodity_research import _check_staleness

    recent = pd.Timestamp.now() - pd.Timedelta(days=1)
    series = pd.Series([100.0], index=[recent])
    assert _check_staleness(series) == "ok"


def test_missing_detection():
    from commodities.commodity_research import _check_staleness

    assert _check_staleness(None) == "missing"


def test_empty_series_detection():
    import pandas as pd

    from commodities.commodity_research import _check_staleness

    assert _check_staleness(pd.Series(dtype=float)) == "missing"


# -- Rationale generation -----------------------------------------------------


def test_rationale_long():
    from commodities.commodity_research import _generate_rationale

    bullets = _generate_rationale(
        commodity="Gold",
        returns={"1m": 3.0, "3m": 8.0, "12m": 22.0},
        trend_label="strong_up",
        direction="long",
        curve_shape=None,
        cross_rank=0.8,
        macro_label="risk-on",
        macro_outlook="complacent",
    )
    assert len(bullets) >= 2
    assert len(bullets) <= 4
    assert any("3M" in b for b in bullets)
    assert any("proxy-based" in b for b in bullets)


def test_rationale_watchlist():
    from commodities.commodity_research import _generate_rationale

    bullets = _generate_rationale(
        commodity="Copper",
        returns={"1m": 1.0, "3m": 2.0, "12m": None},
        trend_label="neutral",
        direction="watchlist",
        curve_shape=None,
        cross_rank=0.5,
        macro_label="transitional",
        macro_outlook="neutral",
    )
    assert not any("proxy-based" in b for b in bullets)


def test_rationale_with_curve():
    from commodities.commodity_research import _generate_rationale

    bullets = _generate_rationale(
        commodity="WTI Crude Oil",
        returns={"1m": -2.0, "3m": -5.0, "12m": -10.0},
        trend_label="moderate_down",
        direction="short",
        curve_shape="Contango",
        cross_rank=None,
        macro_label="risk-off",
        macro_outlook="opportunity",
    )
    assert any("contango" in b.lower() for b in bullets)


# -- Curve n/a for metals ---------------------------------------------------


def test_curve_na_for_metals():
    from commodities.commodity_research import CURVE_CODES

    metals = ["Gold", "Silver", "Copper", "Platinum", "Palladium", "Aluminum"]
    for m in metals:
        assert m not in CURVE_CODES


# -- Router endpoint test ----------------------------------------------------


def test_commodity_research_endpoint(auth_client, monkeypatch):
    import api.routers.commodity_research as router_mod

    def fake_build():
        return {
            "status": "ok",
            "timestamp": "2026-03-20T12:00:00",
            "macro_regime": {"label": "risk-on", "score": 25.0, "forward_outlook": "complacent"},
            "ideas": [
                {
                    "commodity": "Gold",
                    "ticker": "GC=F",
                    "sector": "metals",
                    "spot_price": 2650.0,
                    "returns": {"1m": 3.2, "3m": 8.1, "12m": 22.5},
                    "factors": {},
                    "composite_score": 61.8,
                    "direction": "long",
                    "confidence": "medium",
                    "rationale": ["Test bullet"],
                    "data_quality": {"prices_daily": "ok"},
                    "price_series": [],
                }
            ],
            "summary": {
                "top_long": {"commodity": "Gold", "score": 61.8},
                "top_short": None,
                "strongest_tailwind": None,
                "strongest_headwind": None,
                "data_health": {"ok": 1, "degraded": 0, "missing": 0},
            },
            "methodology_note": "Test",
        }

    monkeypatch.setattr(router_mod, "get_cached", lambda *a, **kw: None)
    monkeypatch.setattr(router_mod, "set_cached", lambda *a, **kw: None)

    import commodities.commodity_research as cr_mod

    monkeypatch.setattr(cr_mod, "build_commodity_research", fake_build)

    resp = auth_client.get("/api/v1/commodity-research")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert len(data["ideas"]) == 1
    assert data["ideas"][0]["commodity"] == "Gold"
