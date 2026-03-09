"""
Centralized retry utilities for external API calls.

Provides retry-with-exponential-backoff wrappers for:
- yfinance downloads and ticker lookups
- requests.get / requests.post
- FRED API series fetches

All wrappers log retries at WARNING level and raise on final failure.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import pandas as pd
import requests as _requests  # type: ignore[import-untyped]
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0  # seconds; doubles each retry: 2, 4, 8
MAX_BACKOFF = 30.0


def _sleep_backoff(attempt: int, base: float = DEFAULT_BACKOFF_BASE) -> None:
    delay = min(base * (2**attempt), MAX_BACKOFF)
    time.sleep(delay)


# ---------------------------------------------------------------------------
# yfinance wrappers
# ---------------------------------------------------------------------------


def yf_download(
    tickers: str | list[str],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    **kwargs: Any,
) -> pd.DataFrame:
    """Wrap ``yf.download()`` with retry on failure or empty result.

    All keyword arguments are forwarded to ``yf.download()``.
    Returns an empty DataFrame if all retries are exhausted.
    """
    if "progress" not in kwargs:
        kwargs["progress"] = False

    ticker_label = tickers if isinstance(tickers, str) else f"{len(tickers)} tickers"

    for attempt in range(max_retries + 1):
        try:
            df = yf.download(tickers, **kwargs)
            if df is not None and not df.empty:
                return df
            if attempt < max_retries:
                logger.warning(
                    "yf.download(%s) returned empty (attempt %d/%d), retrying",
                    ticker_label,
                    attempt + 1,
                    max_retries + 1,
                )
                _sleep_backoff(attempt, backoff_base)
            else:
                logger.warning("yf.download(%s) returned empty after %d attempts", ticker_label, max_retries + 1)
                return pd.DataFrame()
        except Exception as exc:
            if attempt < max_retries:
                logger.warning(
                    "yf.download(%s) failed (attempt %d/%d): %s — retrying",
                    ticker_label,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                _sleep_backoff(attempt, backoff_base)
            else:
                logger.error("yf.download(%s) failed after %d attempts: %s", ticker_label, max_retries + 1, exc)
                raise

    return pd.DataFrame()  # unreachable, but satisfies type checker


def yf_ticker_info(
    ticker: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
) -> dict[str, Any]:
    """Fetch ``yf.Ticker(ticker).info`` with retry. Returns empty dict on failure."""
    for attempt in range(max_retries + 1):
        try:
            info = yf.Ticker(ticker).info
            if info:
                return cast(dict[str, Any], info)
            if attempt < max_retries:
                logger.warning(
                    "yf.Ticker(%s).info returned empty (attempt %d/%d), retrying",
                    ticker,
                    attempt + 1,
                    max_retries + 1,
                )
                _sleep_backoff(attempt, backoff_base)
        except Exception as exc:
            if attempt < max_retries:
                logger.warning(
                    "yf.Ticker(%s).info failed (attempt %d/%d): %s — retrying",
                    ticker,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                _sleep_backoff(attempt, backoff_base)
            else:
                logger.error("yf.Ticker(%s).info failed after %d attempts: %s", ticker, max_retries + 1, exc)
                return {}
    return {}


# ---------------------------------------------------------------------------
# requests wrappers
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def requests_get(
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    **kwargs: Any,
) -> _requests.Response:
    """``requests.get()`` with retry on network errors and 429/5xx status codes."""
    return _requests_with_retry("GET", url, max_retries=max_retries, backoff_base=backoff_base, **kwargs)


def requests_post(
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    **kwargs: Any,
) -> _requests.Response:
    """``requests.post()`` with retry on network errors and 429/5xx status codes."""
    return _requests_with_retry("POST", url, max_retries=max_retries, backoff_base=backoff_base, **kwargs)


def _requests_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    **kwargs: Any,
) -> _requests.Response:
    if "timeout" not in kwargs:
        kwargs["timeout"] = 30

    for attempt in range(max_retries + 1):
        try:
            resp = _requests.request(method, url, **kwargs)

            if resp.status_code < 400:
                return resp

            if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = min(float(retry_after), MAX_BACKOFF)
                    except ValueError:
                        delay = min(backoff_base * (2**attempt), MAX_BACKOFF)
                else:
                    delay = min(backoff_base * (2**attempt), MAX_BACKOFF)
                logger.warning(
                    "%s %s returned %d (attempt %d/%d), retrying in %.1fs",
                    method,
                    url,
                    resp.status_code,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                time.sleep(delay)
                continue

            return resp  # non-retryable status or final attempt

        except (_requests.ConnectionError, _requests.Timeout) as exc:
            if attempt < max_retries:
                logger.warning(
                    "%s %s failed (attempt %d/%d): %s — retrying",
                    method,
                    url,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                _sleep_backoff(attempt, backoff_base)
            else:
                raise

    raise RuntimeError(f"Unreachable: {method} {url}")  # satisfies type checker


# ---------------------------------------------------------------------------
# FRED API wrapper
# ---------------------------------------------------------------------------


def fred_get_series(
    fred_client: Any,
    series_id: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    **kwargs: Any,
) -> pd.Series:
    """Wrap ``fred.get_series()`` with retry. Returns empty Series on failure."""
    for attempt in range(max_retries + 1):
        try:
            result = fred_client.get_series(series_id, **kwargs)
            if result is not None and not result.empty:
                return result
            if attempt < max_retries:
                logger.warning(
                    "fred.get_series(%s) returned empty (attempt %d/%d), retrying",
                    series_id,
                    attempt + 1,
                    max_retries + 1,
                )
                _sleep_backoff(attempt, backoff_base)
        except Exception as exc:
            if attempt < max_retries:
                logger.warning(
                    "fred.get_series(%s) failed (attempt %d/%d): %s — retrying",
                    series_id,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                _sleep_backoff(attempt, backoff_base)
            else:
                logger.error("fred.get_series(%s) failed after %d attempts: %s", series_id, max_retries + 1, exc)
                return pd.Series(dtype=float)

    return pd.Series(dtype=float)
