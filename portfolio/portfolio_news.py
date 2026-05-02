"""Compatibility facade for the digest-based portfolio news replacement."""

from __future__ import annotations

from typing import Any

from portfolio.news_digests import get_report_context, list_digests


def get_data(refresh: bool = False) -> dict[str, Any]:
    """Return uploaded news digest data.

    The old IBKR/Google RSS implementation has been removed. ``refresh`` is
    accepted for callers that still pass the old parameter.
    """
    del refresh
    return list_digests()


__all__ = ["get_data", "get_report_context", "list_digests"]
