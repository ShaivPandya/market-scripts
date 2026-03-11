"""
Bond dashboard data: past-year time series for 2Y, 10Y, 30Y across US, UK, DE, JP.

Reuses fetcher functions from yield_curve.py.  Returns all tenors and countries
in a single payload so the frontend can filter client-side.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from government_bonds.yield_curve import (
    FRED_SERIES,
    _build_fred_client,
    _fetch_boe_gilts,
    _fetch_bundesbank_germany,
    _fetch_fred_series,
    _fetch_mof_japan,
    _latest_market_close_date,
    _normalize_series,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DASHBOARD_TENORS = ["2Y", "10Y", "30Y"]

COUNTRIES: list[tuple[str, str, str]] = [
    # (code, display_name, source_label)
    ("US", "United States", "FRED"),
    ("UK", "United Kingdom", "BoE (monthly)"),
    ("DE", "Germany", "Bundesbank"),
    ("JP", "Japan", "MOF"),
]

COUNTRY_ORDER = [c[0] for c in COUNTRIES]

_LOOKBACK_DAYS = 365

# ---------------------------------------------------------------------------
# File-based cache
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data_cache" / "bond_dashboard"
_CACHE_PATH = _CACHE_DIR / "bond_dashboard.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_VERSION = 1


def _load_cache() -> dict | None:
    try:
        if not _CACHE_PATH.exists():
            return None
        raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        if raw.get("version") != _CACHE_VERSION:
            return None
        if not isinstance(raw.get("payload"), dict):
            return None
        fetched_at = raw.get("fetched_at")
        if not isinstance(fetched_at, str):
            return None
        datetime.fromisoformat(fetched_at)
        return raw
    except Exception:
        return None


def _write_cache(payload: dict, as_of_date: str | None, fetched_at: str | None = None) -> None:
    record = {
        "version": _CACHE_VERSION,
        "fetched_at": fetched_at or datetime.now().isoformat(),
        "as_of_date": as_of_date,
        "payload": payload,
    }
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(_CACHE_PATH)
    except Exception:
        return


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series_to_points(series: pd.Series, cutoff: pd.Timestamp) -> list[dict]:
    """Trim series to cutoff and convert to [{date, value}, ...]."""
    trimmed = series[series.index >= cutoff]
    if trimmed.empty:
        return []
    return [
        {"date": ts.date().isoformat(), "value": round(float(val), 4)}
        for ts, val in zip(trimmed.index, trimmed.values, strict=False)
    ]


def _tenor_summary(series: pd.Series, cutoff: pd.Timestamp) -> dict:
    """Build per-tenor summary: series points, latest, year_ago, change_bps."""
    points = _series_to_points(series, cutoff)
    if not points:
        return {
            "series": [],
            "latest": None,
            "latest_date": None,
            "year_ago": None,
            "year_ago_date": None,
            "change_bps": None,
        }

    latest_val = points[-1]["value"]
    latest_date = points[-1]["date"]
    year_ago_val = points[0]["value"]
    year_ago_date = points[0]["date"]
    change_bps = (
        round((latest_val - year_ago_val) * 100.0, 1) if latest_val is not None and year_ago_val is not None else None
    )

    return {
        "series": points,
        "latest": latest_val,
        "latest_date": latest_date,
        "year_ago": year_ago_val,
        "year_ago_date": year_ago_date,
        "change_bps": change_bps,
    }


# ---------------------------------------------------------------------------
# Country fetchers → dict[tenor, pd.Series]
# ---------------------------------------------------------------------------


def _fetch_us() -> dict[str, pd.Series]:
    fred_client, _ = _build_fred_client()
    if fred_client is None:
        return {}
    result: dict[str, pd.Series] = {}
    for tenor in DASHBOARD_TENORS:
        series_id = FRED_SERIES.get("US", {}).get(tenor)
        if series_id is None:
            continue
        s = _fetch_fred_series(fred_client, series_id)
        if s is not None and not s.empty:
            result[tenor] = s
    return result


def _fetch_all_countries() -> dict[str, dict[str, pd.Series]]:
    """Fetch all tenors for all countries. Returns {country_code: {tenor: Series}}."""
    us_data = _fetch_us()

    # Bulk fetchers return all tenors at once
    uk_data = _fetch_boe_gilts()
    de_data = _fetch_bundesbank_germany()
    jp_data = _fetch_mof_japan()

    def _pick(bulk: dict[str, pd.Series]) -> dict[str, pd.Series]:
        return {t: bulk[t] for t in DASHBOARD_TENORS if t in bulk}

    return {
        "US": _pick(us_data),
        "UK": _pick(uk_data),
        "DE": _pick(de_data),
        "JP": _pick(jp_data),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_data() -> dict:
    """
    Build bond dashboard payload with past-year time series for 2Y, 10Y, 30Y.

    Returns JSON-serializable dict consumable by the frontend and agent tool.
    """
    # --- cache check ---
    cached_record = _load_cache()
    cached_payload = cached_record.get("payload") if cached_record else None

    if cached_record and isinstance(cached_payload, dict):
        try:
            fetched_at = datetime.fromisoformat(str(cached_record["fetched_at"]))
            age_seconds = (datetime.now() - fetched_at).total_seconds()
        except Exception:
            age_seconds = _CACHE_TTL_SECONDS + 1

        if age_seconds < _CACHE_TTL_SECONDS:
            return cached_payload

        # TTL expired — check if market has new close
        cached_as_of = cached_record.get("as_of_date")
        latest_close = _latest_market_close_date()
        if isinstance(cached_as_of, str) and latest_close is not None and latest_close <= cached_as_of:
            _write_cache(cached_payload, cached_as_of, fetched_at=datetime.now().isoformat())
            return cached_payload

    # --- fetch live ---
    try:
        all_data = _fetch_all_countries()

        now = pd.Timestamp.now()  # tz-naive to match series indices
        cutoff = now - pd.Timedelta(days=_LOOKBACK_DAYS)

        # Determine global as_of_date
        as_of_date: str | None = None
        for country_series in all_data.values():
            for s in country_series.values():
                if not s.empty:
                    d = s.index[-1].date().isoformat()
                    if as_of_date is None or d > as_of_date:
                        as_of_date = d

        countries: dict[str, dict] = {}
        for code, name, source in COUNTRIES:
            tenor_data = all_data.get(code, {})
            tenors: dict[str, dict] = {}
            for tenor in DASHBOARD_TENORS:
                s = tenor_data.get(tenor)
                if s is not None and not s.empty:
                    tenors[tenor] = _tenor_summary(s, cutoff)
                else:
                    tenors[tenor] = _tenor_summary(pd.Series(dtype=float), cutoff)

            countries[code] = {
                "code": code,
                "name": name,
                "source": source,
                "tenors": tenors,
            }

        result = {
            "timestamp": now.isoformat(),
            "lookback_days": _LOOKBACK_DAYS,
            "tenors": DASHBOARD_TENORS,
            "country_order": COUNTRY_ORDER,
            "countries": countries,
        }
    except Exception:
        if isinstance(cached_payload, dict):
            return cached_payload
        raise

    _write_cache(result, as_of_date)
    return result


if __name__ == "__main__":
    import json as _json

    print(_json.dumps(get_data(), indent=2))
