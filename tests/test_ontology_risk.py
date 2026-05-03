from __future__ import annotations

from ontology.risk import (
    compute_breadth_stress,
    compute_macro_regime,
    compute_sector_stress_map,
    compute_volatility_cluster,
    risk_level,
    score_position,
)
from ontology.sources.dtos import (
    EconomicGrowthSnapshot,
    LaborMarketSnapshot,
    LiquiditySnapshot,
    MarketBreadthSnapshot,
    PositioningRow,
    PositioningSnapshot,
    SectorMetricRow,
    SectorMetricsSnapshot,
    SentimentSnapshot,
    Top50BreadthSnapshot,
    VixTermStructureSnapshot,
)


def test_score_position_weighted_formula():
    score = score_position(
        volatility_cluster=1.0,
        breadth_stress=0.5,
        sector_stress=0.2,
        macro_regime=0.0,
    )
    # 0.35*1 + 0.25*0.5 + 0.25*0.2 + 0.15*0
    assert round(score, 4) == 0.525


def test_risk_level_buckets():
    assert risk_level(0.8) == "high"
    assert risk_level(0.5) == "medium"
    assert risk_level(0.2) == "low"


def test_risk_functions_accept_normalized_dtos():
    vol, vol_evidence = compute_volatility_cluster(
        VixTermStructureSnapshot(date="2026-05-01", vix=22, vix3m=24, ratio=1.09, signal="Neutral"),
        SentimentSnapshot(volatility=[{"date": "2026-05-01", "vvix": 120}], latest_vvix=120),
    )
    breadth, _breadth_evidence = compute_breadth_stress(
        MarketBreadthSnapshot(
            total_analyzed=500,
            pct_above_200dma=45,
            pct_above_20dma=40,
            pct_at_20day_low=30,
            pct_at_52wk_low=12,
            as_of_date="2026-05-01",
        ),
        Top50BreadthSnapshot(pct_below_50dma=50, pct_3plus_dist=40, pct_broke_20low=25, universe_size=50),
    )
    sector_scores, sector_evidence = compute_sector_stress_map(
        SectorMetricsSnapshot(
            rows=[
                SectorMetricRow(
                    sector="Information Technology",
                    weight_now=30,
                    chg_1m_pp=None,
                    chg_3m_pp=-1.0,
                    chg_6m_pp=None,
                    relperf_3m_pp=-4.0,
                    relperf_12m_pp=None,
                    pct_above_200dma=-3.0,
                )
            ],
            timestamp="2026-05-01T20:00:00",
        )
    )
    macro, macro_evidence = compute_macro_regime(
        LiquiditySnapshot(composite_score=-0.2, regime="tight", latest_date="2026-05-01"),
        PositioningSnapshot(
            rows=[
                PositioningRow(
                    instrument="SP500",
                    report_date="2026-05-01",
                    lf_net=None,
                    lf_net_pct_oi=None,
                    lf_z=2.1,
                    lf_deleveraging_z=None,
                    lf_forced=None,
                )
            ]
        ),
        EconomicGrowthSnapshot(
            commodities={"Copper": {"1M": -1, "3M": -2, "6M": -3}},
            equities={},
            currencies={},
        ),
        LaborMarketSnapshot(latest={}, timestamp="2026-05-01T20:00:00", initial_claims_change=10),
    )

    assert vol == 0.65
    assert vol_evidence[-1]["name"] == "VVIX"
    assert breadth > 0
    assert sector_scores["Information Technology"] > 0
    assert sector_evidence[0]["sector"] == "Information Technology"
    assert macro > 0.8
    assert {item["source"] for item in macro_evidence} >= {"liquidity", "positioning", "labor_market"}
