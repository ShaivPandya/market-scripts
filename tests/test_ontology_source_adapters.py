from __future__ import annotations

from ontology.sources.liquidity import LiquidityAdapter
from ontology.sources.macro import EconomicGrowthAdapter, LaborMarketAdapter, PositioningAdapter, SentimentAdapter
from ontology.sources.market_technicals import MarketBreadthAdapter, Top50BreadthAdapter, VixTermStructureAdapter
from ontology.sources.portfolio import PortfolioAdapter
from ontology.sources.sector_metrics import SectorMetricsAdapter


def test_portfolio_adapter_normalizes_positions():
    result = PortfolioAdapter(timeframe="Daily").normalize(
        {
            "metadata": {"mu": {"asset": "Equity", "direction": "Long"}},
            "positions": {"MU": [{"date": "2026-05-01", "value": 111.5}]},
            "timeframe": "Daily",
            "timestamp": "2026-05-01T20:00:00",
            "position_order": ["MU"],
            "analytics": {"count": 1},
        }
    )

    assert result.status == "ok"
    assert result.quality == "ok"
    assert result.data is not None
    assert result.data.positions["MU"].latest_price == 111.5
    assert result.to_status_dict()["source_version"] == "1"


def test_vix_adapter_marks_missing_latest_as_partial():
    result = VixTermStructureAdapter().normalize({"latest_df": [], "recent_df": [], "hits_df": []})

    assert result.status == "partial"
    assert result.quality == "missing"
    assert result.schema_drift
    assert result.data is not None
    assert result.data.signal == "Neutral"


def test_market_breadth_adapter_detects_schema_drift():
    result = MarketBreadthAdapter().normalize({"pct_above_200dma": 50.0, "total_analyzed": 500})

    assert result.status == "partial"
    assert result.quality == "schema_drift"
    assert any(issue.severity == "warning" for issue in result.schema_drift)


def test_top50_adapter_allows_unknown_additive_fields():
    result = Top50BreadthAdapter().normalize(
        {
            "pct_below_50dma": 40,
            "pct_3plus_dist": 30,
            "pct_broke_20low": 20,
            "universe_size": 50,
            "new_additive_field": True,
        }
    )

    assert result.status == "ok"
    assert result.quality == "ok"
    assert any(issue.severity == "info" for issue in result.schema_drift)


def test_sector_metrics_adapter_normalizes_rows_and_empty_payloads():
    ok = SectorMetricsAdapter().normalize(
        {
            "weights_df": [
                {
                    "Sector": "Information Technology",
                    "Weight_Now": 30,
                    "Chg_3M_pp": -1,
                    "RelPerf_3M_pp": -4,
                    "Pct_Above_200DMA": -3,
                }
            ],
            "timestamp": "2026-05-01T20:00:00",
        }
    )
    empty = SectorMetricsAdapter().normalize({"weights_df": [], "timestamp": "2026-05-01T20:00:00"})

    assert ok.status == "ok"
    assert ok.data is not None
    assert ok.data.rows[0].sector == "Information Technology"
    assert empty.status == "partial"
    assert empty.quality == "missing"


def test_sector_metrics_adapter_repairs_source_rows_without_sector():
    result = SectorMetricsAdapter().normalize(
        {
            "weights_df": [
                {
                    "Weight_Now": 17.8,
                    "Chg_3M_pp": 0.4,
                    "RelPerf_3M_pp": -6.9,
                    "Pct_Above_200DMA": 2.4,
                }
            ],
            "timestamp": "2026-05-01T20:00:00",
        }
    )

    assert result.status == "ok"
    assert result.data is not None
    assert result.data.rows[0].sector == "Communication Services"


def test_liquidity_adapter_normalizes_regime():
    result = LiquidityAdapter().normalize(
        {
            "composite_score": -0.2,
            "regime": "tight",
            "latest_date": "2026-05-01",
            "regional_scores": {"us": {"score": -0.1}},
            "components": [{"label": "Net Liquidity", "contribution": -0.2}],
            "changes": {},
            "df_weekly": "large",
            "composite_series": "large",
        }
    )

    assert result.status == "ok"
    assert result.data is not None
    assert result.data.regime == "tight"
    assert "df_weekly" not in result.lineage.payload_fingerprint


def test_sentiment_adapter_degrades_on_survey_errors():
    result = SentimentAdapter().normalize(
        {
            "put_call": {"equity": {"ratio": 0.7}},
            "surveys": {"errors": {"aaii": "blocked"}},
            "volatility": [{"date": "2026-05-01", "vvix": 115}],
        }
    )

    assert result.status == "partial"
    assert result.quality == "degraded"
    assert result.data is not None
    assert result.data.latest_vvix == 115


def test_positioning_adapter_normalizes_rows():
    result = PositioningAdapter().normalize(
        [{"instrument": "SP500", "report_date": "2026-05-01", "lf_net": 10, "lf_z": 2.3}]
    )

    assert result.status == "ok"
    assert result.data is not None
    assert result.data.rows[0].instrument == "SP500"


def test_economic_growth_adapter_backfills_currency_periods():
    result = EconomicGrowthAdapter().normalize(
        {
            "commodities": {"Copper": {"1M": -1}},
            "equities": {},
            "currencies": {"AUDJPY": {"1-mo": 1.2}},
            "timestamp": "2026-05-01T20:00:00",
        }
    )

    assert result.status == "ok"
    assert result.data is not None
    assert result.data.currencies["AUDJPY"]["1-yr"] is None


def test_labor_market_adapter_exposes_initial_claims_change():
    result = LaborMarketAdapter().normalize(
        {
            "series": {"initial_claims": {"label": "Initial Claims", "unit": "thousands"}},
            "latest": {"initial_claims": {"value": 230, "date": "2026-05-01", "change": 12.5}},
            "timestamp": "2026-05-01T20:00:00",
        }
    )

    assert result.status == "ok"
    assert result.data is not None
    assert result.data.initial_claims_change == 12.5
