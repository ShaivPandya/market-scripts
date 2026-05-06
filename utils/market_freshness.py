from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

MARKET_TIMEZONE = ZoneInfo("America/New_York")
AFTER_CLOSE_FRESHNESS_CUTOFF = time(hour=16, minute=15)

MarketCacheAction = Literal["use_cache", "probe", "refresh"]


@dataclass(frozen=True, slots=True)
class MarketCacheDecision:
    action: MarketCacheAction
    status: str
    stale: bool
    cached_as_of: str | None
    expected_market_date: str
    latest_close: str | None
    reason: str
    cache_age_seconds: int | None = None
    cache_ttl_seconds: int | None = None

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "status": self.status,
            "stale": self.stale,
            "cached_as_of": self.cached_as_of,
            "expected_market_date": self.expected_market_date,
            "latest_close": self.latest_close,
            "reason": self.reason,
        }
        if self.cache_age_seconds is not None:
            meta["cache_age_seconds"] = self.cache_age_seconds
        if self.cache_ttl_seconds is not None:
            meta["cache_ttl_seconds"] = self.cache_ttl_seconds
        return meta


def parse_market_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def expected_market_date(now: datetime | None = None) -> date:
    current = now or datetime.now(MARKET_TIMEZONE)
    if current.tzinfo is None:
        local = current.replace(tzinfo=MARKET_TIMEZONE)
    else:
        local = current.astimezone(MARKET_TIMEZONE)
    current_date = local.date()
    if local.weekday() >= 5:
        return previous_business_day(current_date)
    if local.time() < AFTER_CLOSE_FRESHNESS_CUTOFF:
        return previous_business_day(current_date)
    return current_date


def previous_business_day(value: date) -> date:
    cur = value - timedelta(days=1)
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def build_market_cache_metadata(
    *,
    status: str,
    stale: bool,
    reason: str,
    cached_as_of: str | None = None,
    expected_market_date_value: str | None = None,
    latest_close: str | None = None,
    cache_age_seconds: int | None = None,
    cache_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "status": status,
        "stale": stale,
        "cached_as_of": cached_as_of,
        "expected_market_date": expected_market_date_value or expected_market_date().isoformat(),
        "latest_close": latest_close,
        "reason": reason,
    }
    if cache_age_seconds is not None:
        meta["cache_age_seconds"] = cache_age_seconds
    if cache_ttl_seconds is not None:
        meta["cache_ttl_seconds"] = cache_ttl_seconds
    return meta


def metadata_from_decision(
    decision: MarketCacheDecision,
    *,
    status: str | None = None,
    stale: bool | None = None,
    reason: str | None = None,
    cached_as_of: str | None = None,
    latest_close: str | None = None,
) -> dict[str, Any]:
    meta = decision.metadata()
    if status is not None:
        meta["status"] = status
    if stale is not None:
        meta["stale"] = stale
    if reason is not None:
        meta["reason"] = reason
    if cached_as_of is not None:
        meta["cached_as_of"] = cached_as_of
    if latest_close is not None:
        meta["latest_close"] = latest_close
    return meta


def attach_market_cache_metadata(payload: dict[str, Any], market_cache: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    raw_meta = out.get("_meta")
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    meta["market_cache"] = market_cache
    out["_meta"] = meta
    return out


def market_cache_decision(
    *,
    cached_as_of: Any,
    fetched_at: Any,
    ttl_seconds: int,
    latest_close: Any = None,
    latest_close_probed: bool = False,
    now: datetime | None = None,
) -> MarketCacheDecision:
    expected = expected_market_date(now)
    cached_date = parse_market_date(cached_as_of)
    latest_date = parse_market_date(latest_close)
    age_seconds = _cache_age_seconds(fetched_at, now)
    cached_iso = cached_date.isoformat() if cached_date is not None else None
    latest_iso = latest_date.isoformat() if latest_date is not None else None

    if cached_date is None:
        return MarketCacheDecision(
            action="refresh",
            status="refresh",
            stale=True,
            cached_as_of=None,
            expected_market_date=expected.isoformat(),
            latest_close=latest_iso,
            reason="cache missing parseable as_of_date",
            cache_age_seconds=age_seconds,
            cache_ttl_seconds=ttl_seconds,
        )

    if cached_date >= expected:
        return MarketCacheDecision(
            action="use_cache",
            status="hit",
            stale=False,
            cached_as_of=cached_iso,
            expected_market_date=expected.isoformat(),
            latest_close=latest_iso,
            reason="cache as_of satisfies expected market date",
            cache_age_seconds=age_seconds,
            cache_ttl_seconds=ttl_seconds,
        )

    if not latest_close_probed and latest_date is None:
        return MarketCacheDecision(
            action="probe",
            status="probe",
            stale=True,
            cached_as_of=cached_iso,
            expected_market_date=expected.isoformat(),
            latest_close=None,
            reason="cache as_of is older than expected market date",
            cache_age_seconds=age_seconds,
            cache_ttl_seconds=ttl_seconds,
        )

    if latest_date is None:
        return MarketCacheDecision(
            action="use_cache",
            status="stale_fallback",
            stale=True,
            cached_as_of=cached_iso,
            expected_market_date=expected.isoformat(),
            latest_close=None,
            reason="latest close probe unavailable; returning cached payload",
            cache_age_seconds=age_seconds,
            cache_ttl_seconds=ttl_seconds,
        )

    if latest_date <= cached_date:
        return MarketCacheDecision(
            action="use_cache",
            status="hit_unchanged",
            stale=False,
            cached_as_of=cached_iso,
            expected_market_date=expected.isoformat(),
            latest_close=latest_iso,
            reason="latest close has not advanced beyond cached as_of",
            cache_age_seconds=age_seconds,
            cache_ttl_seconds=ttl_seconds,
        )

    return MarketCacheDecision(
        action="refresh",
        status="refresh",
        stale=True,
        cached_as_of=cached_iso,
        expected_market_date=expected.isoformat(),
        latest_close=latest_iso,
        reason="latest close advanced beyond cached as_of",
        cache_age_seconds=age_seconds,
        cache_ttl_seconds=ttl_seconds,
    )


def _cache_age_seconds(fetched_at: Any, now: datetime | None = None) -> int | None:
    fetched = _parse_datetime(fetched_at)
    if fetched is None:
        return None
    current = now or datetime.now(MARKET_TIMEZONE)
    if fetched.tzinfo is None:
        if current.tzinfo is not None:
            current = current.astimezone(MARKET_TIMEZONE).replace(tzinfo=None)
    elif current.tzinfo is None:
        current = current.replace(tzinfo=MARKET_TIMEZONE).astimezone(fetched.tzinfo)
    else:
        current = current.astimezone(fetched.tzinfo)
    return max(0, round((current - fetched).total_seconds()))


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
