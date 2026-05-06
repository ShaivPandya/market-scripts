"""Disk-backed cache helpers for portfolio analyzer SPDR anchor signals."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from api.serializers import serialize_value
from utils.retry import yf_download

LOGGER = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "v1"
WEEKLY_FRESH_DAYS = 6
DAILY_FRESH_DAYS = 0
STALE_GRACE_TRADING_DAYS = 1


@dataclass(frozen=True)
class AnchorCacheResult:
    payload: dict[str, Any]
    status: str
    as_of: str | None
    stale: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cache_root() -> Path:
    env_path = (os.getenv("PORTFOLIO_ANALYZER_ANCHOR_CACHE_DIR") or "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return _repo_root() / "data_cache" / "portfolio_analyzer_anchor"


def _today() -> date:
    return datetime.now(UTC).date()


def _iso_date(value: date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _week_bucket(day: date) -> tuple[str, date]:
    iso = day.isocalendar()
    week_start = day - timedelta(days=day.weekday())
    return f"{iso.year}-W{iso.week:02d}", week_start


def _business_days_between(start: date, end: date) -> int:
    if end <= start:
        return 0
    days = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() < 5:
            days += 1
        cursor += timedelta(days=1)
    return days


def _within_stale_grace(
    *,
    record_as_of: str | None,
    current_date: date,
    fresh_days_after_as_of: int,
    grace_trading_days: int = STALE_GRACE_TRADING_DAYS,
) -> bool:
    as_of = _parse_date(record_as_of)
    if as_of is None:
        return False
    fresh_until = as_of + timedelta(days=fresh_days_after_as_of)
    if current_date <= fresh_until:
        return True
    return _business_days_between(fresh_until, current_date) <= grace_trading_days


def _hash_json(value: Any) -> str:
    blob = json.dumps(serialize_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _record_path(kind: str, key_prefix: str, freshness_token: str) -> Path:
    logical = f"{CACHE_SCHEMA_VERSION}:{kind}:{key_prefix}:{freshness_token}"
    digest = hashlib.sha1(logical.encode("utf-8")).hexdigest()
    return _cache_root() / kind / f"{digest}.json"


def _read_record(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            return None
        return raw
    except Exception:
        LOGGER.debug("anchor cache read failed path=%s", path, exc_info=True)
        return None


def _write_record(
    *,
    kind: str,
    key_prefix: str,
    freshness_token: str,
    as_of: str,
    payload: dict[str, Any],
) -> None:
    path = _record_path(kind, key_prefix, freshness_token)
    record = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": kind,
        "key_prefix": key_prefix,
        "freshness_token": freshness_token,
        "as_of": as_of,
        "created_at": datetime.now(UTC).isoformat(),
        "payload": serialize_value(payload),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        LOGGER.debug("anchor cache write failed path=%s", path, exc_info=True)


def _latest_stale_record(
    *,
    kind: str,
    key_prefix: str,
    current_date: date,
    fresh_days_after_as_of: int,
) -> dict[str, Any] | None:
    folder = _cache_root() / kind
    if not folder.exists():
        return None

    candidates: list[dict[str, Any]] = []
    for path in folder.glob("*.json"):
        record = _read_record(path)
        if not record or record.get("key_prefix") != key_prefix:
            continue
        if not _within_stale_grace(
            record_as_of=_iso_date(record.get("as_of")),
            current_date=current_date,
            fresh_days_after_as_of=fresh_days_after_as_of,
        ):
            continue
        candidates.append(record)

    if not candidates:
        return None
    return max(candidates, key=lambda item: str(item.get("as_of") or ""))


def _load_or_refresh(
    *,
    kind: str,
    key_prefix: str,
    freshness_token: str,
    as_of: str,
    current_date: date,
    fresh_days_after_as_of: int,
    loader: Callable[[], dict[str, Any]],
) -> AnchorCacheResult:
    path = _record_path(kind, key_prefix, freshness_token)
    record = _read_record(path)
    if record:
        return AnchorCacheResult(
            payload=dict(record["payload"]),
            status="hit",
            as_of=_iso_date(record.get("as_of")),
            stale=False,
        )

    try:
        payload = loader()
    except Exception:
        stale = _latest_stale_record(
            kind=kind,
            key_prefix=key_prefix,
            current_date=current_date,
            fresh_days_after_as_of=fresh_days_after_as_of,
        )
        if stale:
            return AnchorCacheResult(
                payload=dict(stale["payload"]),
                status="stale_fallback",
                as_of=_iso_date(stale.get("as_of")),
                stale=True,
            )
        raise

    _write_record(
        kind=kind,
        key_prefix=key_prefix,
        freshness_token=freshness_token,
        as_of=as_of,
        payload=payload,
    )
    return AnchorCacheResult(payload=payload, status="refresh", as_of=as_of, stale=False)


def _df_to_payload(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {"index": [], "columns": [], "data": []}
    return {
        "index": [str(item) for item in df.index.tolist()],
        "columns": [str(item) for item in df.columns.tolist()],
        "data": [[serialize_value(value) for value in row] for row in df.to_numpy(dtype=object).tolist()],
    }


def _df_from_payload(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    index = payload.get("index") or []
    columns = payload.get("columns") or []
    data = payload.get("data") or []
    if not isinstance(index, list) or not isinstance(columns, list) or not isinstance(data, list):
        return pd.DataFrame()
    return pd.DataFrame(data, index=[str(item) for item in index], columns=[str(item) for item in columns])


def _holdings_to_payload(holdings: dict[str, pd.Series]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for etf, weights in holdings.items():
        rows: list[dict[str, Any]] = []
        if weights is not None and not weights.empty:
            for ticker, weight in weights.items():
                rows.append({"ticker": str(ticker), "weight": serialize_value(weight)})
        out[str(etf)] = rows
    return out


def latest_market_close_date(benchmark: str = "SPY") -> str:
    try:
        probe = yf_download(
            benchmark,
            period="10d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            max_retries=0,
        )
        if probe is not None and not probe.empty:
            idx = pd.to_datetime(probe.index, errors="coerce").dropna()
            if not idx.empty:
                return str(idx[-1].date().isoformat())
    except Exception:
        LOGGER.debug("anchor cache latest close probe failed benchmark=%s", benchmark, exc_info=True)
    return _today().isoformat()


def _cache_meta(result: AnchorCacheResult, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_cache_status": result.status,
        f"{prefix}_as_of": result.as_of,
        f"{prefix}_stale": result.stale,
    }


def combine_cache_metadata(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    statuses = [str(item.get("cache_status")) for item in items if item.get("cache_status")]
    stale = any(bool(item.get("stale")) for item in items)
    as_of_values = [str(item.get("as_of")) for item in items if item.get("as_of")]
    if "stale_fallback" in statuses:
        status = "stale_fallback"
    elif "refresh" in statuses:
        status = "refresh"
    elif statuses:
        status = "hit"
    else:
        status = "unknown"
    return {
        "cache_status": status,
        "as_of": min(as_of_values) if as_of_values else None,
        "stale": stale,
    }


def get_spdr_anchor_universe(
    *,
    top_n: int,
    min_unique: int,
    sector_etfs: Sequence[str],
    holdings_fetcher: Callable[..., dict[str, pd.Series]],
) -> tuple[list[str], dict[str, Any]]:
    today = _today()
    week_token, week_start = _week_bucket(today)
    key_prefix = f"spdr_holdings:top_n={int(top_n)}:min_unique={int(min_unique)}:etfs={_hash_json(list(sector_etfs))}"

    def loader() -> dict[str, Any]:
        holdings = holdings_fetcher(list(sector_etfs), top_n=top_n)
        anchor_ordered: list[str] = []
        per_etf_counts: dict[str, int] = {}
        for etf in sector_etfs:
            series = holdings.get(str(etf))
            count = int(len(series)) if series is not None else 0
            per_etf_counts[str(etf)] = count
            if series is None or series.empty:
                continue
            for ticker in series.index:
                ticker_str = str(ticker)
                if ticker_str not in anchor_ordered:
                    anchor_ordered.append(ticker_str)
        is_available = len(anchor_ordered) >= int(min_unique)
        if not is_available:
            raise RuntimeError(f"SPDR anchor universe unavailable: {len(anchor_ordered)} names, need {int(min_unique)}")
        return {
            "holdings": _holdings_to_payload(holdings),
            "anchor_universe": anchor_ordered,
            "metadata": {
                "etfs_requested": [str(etf) for etf in sector_etfs],
                "etfs_fetched": sorted(str(etf) for etf in holdings.keys()),
                "top_n": int(top_n),
                "per_etf_counts": per_etf_counts,
                "anchor_universe_size": len(anchor_ordered),
                "anchor_min_required": int(min_unique),
                "is_available": bool(is_available),
            },
        }

    result = _load_or_refresh(
        kind="spdr_holdings",
        key_prefix=key_prefix,
        freshness_token=week_token,
        as_of=week_start.isoformat(),
        current_date=today,
        fresh_days_after_as_of=WEEKLY_FRESH_DAYS,
        loader=loader,
    )
    payload = result.payload
    anchor_universe = [str(ticker) for ticker in payload.get("anchor_universe") or []]
    metadata = dict(payload.get("metadata") or {})
    metadata.update(_cache_meta(result, "holdings"))
    metadata["cache_status"] = result.status
    metadata["as_of"] = result.as_of
    metadata["stale"] = result.stale
    return anchor_universe, metadata


def get_anchor_price_raw(
    *,
    anchor_universe: Sequence[str],
    benchmark: str,
    years: int,
    price_loader: Callable[..., pd.DataFrame],
    price_momentum_fetcher: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tickers = [str(ticker) for ticker in anchor_universe]
    market_close = latest_market_close_date(benchmark)
    current = _parse_date(market_close) or _today()
    key_prefix = f"anchor_price_raw:benchmark={benchmark}:years={int(years)}:universe={_hash_json(tickers)}"

    def loader() -> dict[str, Any]:
        all_tickers = list(dict.fromkeys([*tickers, benchmark]))
        prices = price_loader(all_tickers, years=years)
        benchmark_map = {ticker: benchmark for ticker in tickers}
        raw = price_momentum_fetcher(tickers, benchmark_map, prices)
        if raw is None or raw.empty:
            raise RuntimeError("anchor price momentum unavailable")
        return {"price_raw": _df_to_payload(raw)}

    result = _load_or_refresh(
        kind="anchor_price_raw",
        key_prefix=key_prefix,
        freshness_token=market_close,
        as_of=market_close,
        current_date=current,
        fresh_days_after_as_of=DAILY_FRESH_DAYS,
        loader=loader,
    )
    meta = _cache_meta(result, "price")
    meta["cache_status"] = result.status
    meta["as_of"] = result.as_of
    meta["stale"] = result.stale
    return _df_from_payload(result.payload.get("price_raw")), meta


def get_anchor_fundamentals(
    *,
    anchor_universe: Sequence[str],
    benchmark: str,
    years: int,
    use_edgar: bool,
    quality_fetcher: Callable[..., pd.DataFrame],
    eps_fetcher: Callable[..., pd.DataFrame],
    revenue_fetcher: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    today = _today()
    week_token, week_start = _week_bucket(today)
    tickers = [str(ticker) for ticker in anchor_universe]
    key_prefix = (
        f"anchor_fundamentals:benchmark={benchmark}:years={int(years)}:use_edgar={bool(use_edgar)}:"
        f"universe={_hash_json(tickers)}"
    )

    def loader() -> dict[str, Any]:
        quality_raw = quality_fetcher(tickers, market=benchmark, growth_years=years)
        eps_raw = eps_fetcher(tickers, growth_years=3, use_edgar=use_edgar)
        revenue_raw = revenue_fetcher(tickers, growth_years=3, use_edgar=use_edgar)
        if (
            (quality_raw is None or quality_raw.empty)
            and (eps_raw is None or eps_raw.empty)
            and (revenue_raw is None or revenue_raw.empty)
        ):
            raise RuntimeError("anchor fundamentals unavailable")
        return {
            "quality_raw": _df_to_payload(quality_raw),
            "eps_raw": _df_to_payload(eps_raw),
            "revenue_raw": _df_to_payload(revenue_raw),
        }

    result = _load_or_refresh(
        kind="anchor_fundamentals",
        key_prefix=key_prefix,
        freshness_token=week_token,
        as_of=week_start.isoformat(),
        current_date=today,
        fresh_days_after_as_of=WEEKLY_FRESH_DAYS,
        loader=loader,
    )
    meta = _cache_meta(result, "fundamentals")
    meta["cache_status"] = result.status
    meta["as_of"] = result.as_of
    meta["stale"] = result.stale
    return (
        _df_from_payload(result.payload.get("quality_raw")),
        _df_from_payload(result.payload.get("eps_raw")),
        _df_from_payload(result.payload.get("revenue_raw")),
        meta,
    )


def get_anchor_signal_table(
    *,
    anchor_universe: Sequence[str],
    benchmark: str,
    years: int,
    weights: dict[str, float],
    clip_bounds: tuple[float, float],
    loader: Callable[[], pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    market_close = latest_market_close_date(benchmark)
    current = _parse_date(market_close) or _today()
    key_prefix = (
        f"anchor_signal_table:benchmark={benchmark}:years={int(years)}:"
        f"weights={_hash_json(weights)}:clip={_hash_json(list(clip_bounds))}:"
        f"universe={_hash_json([str(ticker) for ticker in anchor_universe])}"
    )

    def payload_loader() -> dict[str, Any]:
        loaded = loader()
        source_meta: dict[str, Any] = {}
        if isinstance(loaded, tuple):
            signals, raw_source_meta = loaded
            source_meta = dict(raw_source_meta or {})
        else:
            signals = loaded
        if signals is None or signals.empty:
            raise RuntimeError("anchor signal table unavailable")
        return {"signals": _df_to_payload(signals), "source_meta": source_meta}

    result = _load_or_refresh(
        kind="anchor_signal_table",
        key_prefix=key_prefix,
        freshness_token=market_close,
        as_of=market_close,
        current_date=current,
        fresh_days_after_as_of=DAILY_FRESH_DAYS,
        loader=payload_loader,
    )
    meta = _cache_meta(result, "signals")
    meta["cache_status"] = result.status
    meta["as_of"] = result.as_of
    meta["stale"] = result.stale
    source_meta = result.payload.get("source_meta")
    if isinstance(source_meta, dict):
        meta["source_cache_status"] = source_meta.get("cache_status")
        meta["source_as_of"] = source_meta.get("as_of")
        meta["stale"] = bool(meta["stale"] or source_meta.get("stale"))
    return _df_from_payload(result.payload.get("signals")), meta
