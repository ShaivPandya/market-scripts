from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Any, Literal
from zoneinfo import ZoneInfo

MARKET_TIMEZONE = ZoneInfo("America/New_York")
AFTER_CLOSE_FRESHNESS_CUTOFF = time(hour=16, minute=15)
DEFAULT_MARKET_CALENDAR_ID = "XNYS"

MarketCacheAction = Literal["use_cache", "probe", "refresh"]
FreshnessPolicy = Literal["elapsed", "market_session", "max_age_days", "request_time"]


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


@dataclass(frozen=True, slots=True)
class SourceFreshnessState:
    policy: str
    fresh: bool
    basis: str
    observed_as_of_date: str | None
    reason: str | None
    expected_as_of_date: str | None = None
    calendar_id: str | None = None
    max_age_days: int | None = None
    oldest_acceptable_date: str | None = None
    age_seconds: int | None = None
    max_age_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "policy": self.policy,
            "fresh": self.fresh,
            "basis": self.basis,
            "observed_as_of_date": self.observed_as_of_date,
            "reason": self.reason,
        }
        if self.expected_as_of_date is not None:
            out["expected_as_of_date"] = self.expected_as_of_date
            # Keep the older market freshness field name for compatibility.
            if self.policy == "market_session":
                out["expected_market_date"] = self.expected_as_of_date
        if self.calendar_id is not None:
            out["calendar_id"] = self.calendar_id
        if self.max_age_days is not None:
            out["max_age_days"] = self.max_age_days
        if self.oldest_acceptable_date is not None:
            out["oldest_acceptable_date"] = self.oldest_acceptable_date
        if self.age_seconds is not None:
            out["age_seconds"] = self.age_seconds
        if self.max_age_seconds is not None:
            out["max_age_seconds"] = self.max_age_seconds
        return out


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


def expected_market_date(
    now: datetime | None = None,
    *,
    calendar_id: str = DEFAULT_MARKET_CALENDAR_ID,
    after_close_cutoff: time = AFTER_CLOSE_FRESHNESS_CUTOFF,
) -> date:
    current = now or datetime.now(MARKET_TIMEZONE)
    if current.tzinfo is None:
        local = current.replace(tzinfo=MARKET_TIMEZONE)
    else:
        local = current.astimezone(MARKET_TIMEZONE)
    current_date = local.date()
    reference_date = (
        current_date
        if local.time() >= after_close_cutoff
        else current_date - timedelta(days=1)
    )
    return previous_market_session(reference_date, calendar_id=calendar_id)


def previous_market_session(
    value: date, *, calendar_id: str = DEFAULT_MARKET_CALENDAR_ID
) -> date:
    try:
        import pandas as pd

        calendar = _exchange_calendar(calendar_id)
        timestamp = pd.Timestamp(value)
        try:
            session = calendar.date_to_session(timestamp, direction="none")
        except ValueError:
            session = calendar.date_to_session(timestamp, direction="previous")
        return session.date()
    except Exception:
        return previous_business_day(value)


def previous_business_day(value: date) -> date:
    cur = value
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


@lru_cache(maxsize=8)
def _exchange_calendar(calendar_id: str):
    import exchange_calendars as xcals

    return xcals.get_calendar(calendar_id)


def evaluate_source_freshness(
    value: Any,
    *,
    now: datetime | None = None,
    policy: str | None = None,
    max_age_seconds: int | None = None,
    max_age_days: int | None = None,
    calendar_id: str | None = None,
) -> SourceFreshnessState:
    normalized_policy = str(policy or "elapsed").strip().lower()
    current = now or datetime.now(MARKET_TIMEZONE)
    if current.tzinfo is None:
        local_now = current.replace(tzinfo=MARKET_TIMEZONE)
    else:
        local_now = current.astimezone(MARKET_TIMEZONE)

    if normalized_policy == "request_time":
        observed = local_now.date().isoformat()
        return SourceFreshnessState(
            policy="request_time",
            fresh=True,
            basis="request_time",
            observed_as_of_date=observed,
            reason=None,
            expected_as_of_date=observed,
        )

    if normalized_policy == "market_day":
        normalized_policy = "market_session"
    if normalized_policy == "weekly_report":
        normalized_policy = "max_age_days"
        max_age_days = max_age_days or 10

    if normalized_policy == "market_session":
        observed = parse_market_date(value)
        expected = expected_market_date(
            now=local_now, calendar_id=calendar_id or DEFAULT_MARKET_CALENDAR_ID
        )
        fresh = observed is not None and observed >= expected
        return SourceFreshnessState(
            policy="market_session",
            fresh=fresh,
            basis="as_of_or_fetched_at",
            observed_as_of_date=observed.isoformat() if observed is not None else None,
            expected_as_of_date=expected.isoformat(),
            calendar_id=calendar_id or DEFAULT_MARKET_CALENDAR_ID,
            reason=(
                None
                if fresh
                else (
                    "snapshot has no parseable as-of date"
                    if observed is None
                    else f"snapshot as-of {observed.isoformat()} is older than required market session {expected.isoformat()}"
                )
            ),
        )

    if normalized_policy == "max_age_days":
        window_days = max(1, int(max_age_days or 1))
        observed = parse_market_date(value)
        oldest_acceptable = local_now.date() - timedelta(days=window_days)
        fresh = observed is not None and observed >= oldest_acceptable
        return SourceFreshnessState(
            policy="max_age_days",
            fresh=fresh,
            basis="as_of_or_fetched_at",
            observed_as_of_date=observed.isoformat() if observed is not None else None,
            max_age_days=window_days,
            oldest_acceptable_date=oldest_acceptable.isoformat(),
            reason=(
                None
                if fresh
                else (
                    "snapshot has no parseable as-of date"
                    if observed is None
                    else f"snapshot as-of {observed.isoformat()} is older than freshness window {oldest_acceptable.isoformat()}"
                )
            ),
        )

    age_seconds = _cache_age_seconds(value, local_now)
    max_age = max(0, int(max_age_seconds or 0))
    fresh = age_seconds is not None and age_seconds <= max_age
    return SourceFreshnessState(
        policy="elapsed",
        fresh=fresh,
        basis="fetched_at",
        observed_as_of_date=(
            parse_market_date(value).isoformat()
            if parse_market_date(value) is not None
            else None
        ),
        age_seconds=age_seconds,
        max_age_seconds=max_age,
        reason=(
            None
            if fresh
            else (
                "snapshot has no parseable fetched timestamp"
                if age_seconds is None
                else f"snapshot age {age_seconds}s exceeds freshness SLA {max_age}s"
            )
        ),
    )


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
        "expected_market_date": expected_market_date_value
        or expected_market_date().isoformat(),
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


def attach_market_cache_metadata(
    payload: dict[str, Any], market_cache: dict[str, Any]
) -> dict[str, Any]:
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
