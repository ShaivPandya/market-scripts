"""Smoke test for weekly report generation."""

from paths import setup_paths

setup_paths()

from api.cache import long_cache, set_cached
from api.routers.weekly_report import get_weekly_report


def test_weekly_report_returns_dict():
    """Weekly report should return cached data when available."""
    set_cached(long_cache, "weekly_report_generated", {"report": "ok"})
    res = get_weekly_report(cached_only=True)
    assert isinstance(res, dict)
    assert "report" in res
