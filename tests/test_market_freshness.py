from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from utils.market_freshness import expected_market_date, market_cache_decision

EASTERN = ZoneInfo("America/New_York")


def test_expected_market_date_before_cutoff_uses_previous_business_day():
    now = datetime(2026, 5, 6, 15, 30, tzinfo=EASTERN)
    assert expected_market_date(now).isoformat() == "2026-05-05"


def test_expected_market_date_after_cutoff_uses_current_business_day():
    now = datetime(2026, 5, 6, 16, 30, tzinfo=EASTERN)
    assert expected_market_date(now).isoformat() == "2026-05-06"


def test_expected_market_date_weekend_uses_previous_weekday():
    now = datetime(2026, 5, 9, 12, 0, tzinfo=EASTERN)
    assert expected_market_date(now).isoformat() == "2026-05-08"


def test_expected_market_date_uses_nyse_calendar_for_memorial_day():
    now = datetime(2026, 5, 25, 18, 0, tzinfo=EASTERN)
    assert expected_market_date(now).isoformat() == "2026-05-22"


def test_expected_market_date_sunday_after_friday_close_uses_friday():
    now = datetime(2026, 5, 31, 12, 0, tzinfo=EASTERN)
    assert expected_market_date(now).isoformat() == "2026-05-29"


def test_expected_market_date_after_holiday_before_cutoff_uses_prior_session():
    now = datetime(2026, 5, 26, 15, 30, tzinfo=EASTERN)
    assert expected_market_date(now).isoformat() == "2026-05-22"


def test_expected_market_date_after_holiday_after_cutoff_uses_current_session():
    now = datetime(2026, 5, 26, 16, 30, tzinfo=EASTERN)
    assert expected_market_date(now).isoformat() == "2026-05-26"


def test_older_cached_as_of_requires_probe_even_when_age_under_ttl():
    now = datetime(2026, 5, 6, 16, 30, tzinfo=EASTERN)
    decision = market_cache_decision(
        cached_as_of="2026-05-05",
        fetched_at=now - timedelta(hours=1),
        ttl_seconds=24 * 60 * 60,
        now=now,
    )

    assert decision.action == "probe"
    assert decision.reason == "cache as_of is older than expected market date"


def test_unchanged_latest_close_allows_cache_reuse_and_touch():
    now = datetime(2026, 5, 6, 16, 30, tzinfo=EASTERN)
    decision = market_cache_decision(
        cached_as_of="2026-05-05",
        fetched_at=now - timedelta(hours=30),
        ttl_seconds=24 * 60 * 60,
        latest_close="2026-05-05",
        latest_close_probed=True,
        now=now,
    )

    assert decision.action == "use_cache"
    assert decision.status == "hit_unchanged"
    assert decision.stale is False


def test_probe_failure_allows_stale_fallback_when_cache_exists():
    now = datetime(2026, 5, 6, 16, 30, tzinfo=EASTERN)
    decision = market_cache_decision(
        cached_as_of="2026-05-05",
        fetched_at=now - timedelta(hours=1),
        ttl_seconds=24 * 60 * 60,
        latest_close=None,
        latest_close_probed=True,
        now=now,
    )

    assert decision.action == "use_cache"
    assert decision.status == "stale_fallback"
    assert decision.stale is True
