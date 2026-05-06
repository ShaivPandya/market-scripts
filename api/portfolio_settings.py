"""Persistent portfolio-level settings."""

from __future__ import annotations

import math
from typing import Any

from api.llm_settings import get_setting, set_setting

PORTFOLIO_BOOK_SIZE_KEY = "portfolio.book_size"
DEFAULT_PORTFOLIO_BOOK_SIZE = 100_000.0
MIN_PORTFOLIO_BOOK_SIZE = 10_000.0
MAX_PORTFOLIO_BOOK_SIZE = 10_000_000.0


def normalize_portfolio_book_size(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Portfolio book size must be a number.") from exc

    if not math.isfinite(parsed):
        raise ValueError("Portfolio book size must be finite.")
    if parsed < MIN_PORTFOLIO_BOOK_SIZE or parsed > MAX_PORTFOLIO_BOOK_SIZE:
        raise ValueError(
            f"Portfolio book size must be between ${MIN_PORTFOLIO_BOOK_SIZE:,.0f} and ${MAX_PORTFOLIO_BOOK_SIZE:,.0f}."
        )
    return round(parsed, 2)


def get_configured_portfolio_book_size() -> float | None:
    row = get_setting(PORTFOLIO_BOOK_SIZE_KEY)
    if not row:
        return None

    try:
        return normalize_portfolio_book_size(row.get("value"))
    except ValueError:
        return None


def get_portfolio_book_size() -> float:
    return get_configured_portfolio_book_size() or DEFAULT_PORTFOLIO_BOOK_SIZE


def set_portfolio_book_size(value: Any) -> dict[str, Any]:
    normalized = normalize_portfolio_book_size(value)
    return set_setting(PORTFOLIO_BOOK_SIZE_KEY, str(normalized))
