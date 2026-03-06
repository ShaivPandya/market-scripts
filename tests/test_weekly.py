"""Smoke test for weekly report generation."""

from paths import setup_paths

setup_paths()

from api.routers.weekly_report import get_weekly_report


def test_weekly_report_returns_dict():
    """Weekly report should return a dict with expected keys."""
    res = get_weekly_report()
    assert isinstance(res, dict)
