from __future__ import annotations

from ontology.risk import risk_level, score_position


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
