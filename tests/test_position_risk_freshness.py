from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from api.position_risk import _freshness_state

EASTERN = ZoneInfo("America/New_York")


def test_position_risk_market_freshness_uses_nyse_holidays():
    freshness = _freshness_state(
        "2026-05-22",
        now=datetime(2026, 5, 25, 18, 0, tzinfo=EASTERN),
        policy="market_day",
    )

    assert freshness["fresh"] is True
    assert freshness["policy"] == "market_session"
    assert freshness["expected_as_of_date"] == "2026-05-22"
    assert freshness["expected_market_date"] == "2026-05-22"


def test_position_risk_macro_freshness_uses_source_window():
    freshness = _freshness_state(
        "2026-05-22",
        now=datetime(2026, 5, 31, 12, 0, tzinfo=EASTERN),
        policy="max_age_days",
        max_age_days=10,
    )

    assert freshness["fresh"] is True
    assert freshness["oldest_acceptable_date"] == "2026-05-21"
