"""
Yield curve data snapshot for the web dashboard.

Provides current and historical (N days lookback) curve points for:
- United States (FRED)
- United Kingdom (Bank of England yield curve)
- Germany (Deutsche Bundesbank)
- Japan (MOF)
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import requests  # type: ignore[import-untyped]

from load_env import load_env
from utils.market_freshness import (
    attach_market_cache_metadata,
    build_market_cache_metadata,
    expected_market_date,
    market_cache_decision,
    metadata_from_decision,
)

load_env()

try:
    from fredapi import Fred

    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    Fred = None  # type: ignore[assignment]

from utils.retry import fred_get_series, yf_download


class TenorMeta(TypedDict):
    tenor: str
    years: float


TENOR_ORDER: list[TenorMeta] = [
    {"tenor": "3M", "years": 0.25},
    {"tenor": "6M", "years": 0.50},
    {"tenor": "1Y", "years": 1.00},
    {"tenor": "2Y", "years": 2.00},
    {"tenor": "5Y", "years": 5.00},
    {"tenor": "10Y", "years": 10.00},
    {"tenor": "30Y", "years": 30.00},
]

TENOR_TO_YEARS = {str(row["tenor"]): float(row["years"]) for row in TENOR_ORDER}

COUNTRIES = [
    ("US", "United States"),
    ("UK", "United Kingdom"),
    ("DE", "Germany"),
    ("JP", "Japan"),
]
COUNTRY_NAME_BY_CODE = dict(COUNTRIES)

FRED_SERIES = {
    "US": {
        "3M": "DGS3MO",
        "6M": "DGS6MO",
        "1Y": "DGS1",
        "2Y": "DGS2",
        "5Y": "DGS5",
        "10Y": "DGS10",
        "30Y": "DGS30",
    }
}

# ---------------------------------------------------------------------------
# Japan - Ministry of Finance
# ---------------------------------------------------------------------------
MOF_JGB_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"

_MOF_COLUMN_MAP: dict[str, str] = {
    "1Y": "1Y",
    "2Y": "2Y",
    "5Y": "5Y",
    "10Y": "10Y",
    "30Y": "30Y",
}

# ---------------------------------------------------------------------------
# United Kingdom - Bank of England GLC nominal yield curve (monthly ZIP)
# ---------------------------------------------------------------------------
BOE_GLC_URL = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/glcnominalmonthedata.zip"

_BOE_MATURITY_MAP: dict[float, str] = {
    0.5: "6M",
    1.0: "1Y",
    2.0: "2Y",
    5.0: "5Y",
    10.0: "10Y",
    30.0: "30Y",
}

# ---------------------------------------------------------------------------
# Germany - Deutsche Bundesbank SDMX API
# ---------------------------------------------------------------------------
BUNDESBANK_BASE_URL = "https://api.statistiken.bundesbank.de/rest/data"

_BUNDESBANK_DE_SERIES: dict[str, str] = {
    # Daily yields derived from term structure on listed Federal securities.
    "1Y": "BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R01XX.R.A.A._Z._Z.A",
    "2Y": "BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A",
    "5Y": "BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R05XX.R.A.A._Z._Z.A",
    "10Y": "BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A",
    "30Y": "BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R30XX.R.A.A._Z._Z.A",
}

# ---------------------------------------------------------------------------
# File-based cache (same pattern as market_breadth.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data_cache" / "yield_curve"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_VERSION = 2
_CLOSE_PROBE_TICKER = "SPY"


def _cache_path(lookback_days: int) -> Path:
    return _CACHE_DIR / f"yield_curve_{lookback_days}d.json"


def _country_cache_path(lookback_days: int, country_code: str) -> Path:
    return _CACHE_DIR / f"yield_curve_{lookback_days}d_{country_code.lower()}.json"


def _normalize_country_code(country: str | None) -> str | None:
    if country is None:
        return None
    code = country.strip().upper()
    if not code:
        return None
    if code not in COUNTRY_NAME_BY_CODE:
        supported = ", ".join(COUNTRY_NAME_BY_CODE)
        raise ValueError(f"Unsupported country code {code!r}; expected one of: {supported}")
    return code


def _load_cache(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        if raw.get("version") != _CACHE_VERSION:
            return None
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            return None
        fetched_at = raw.get("fetched_at")
        if not isinstance(fetched_at, str):
            return None
        datetime.fromisoformat(fetched_at)
        return raw
    except Exception:
        return None


def _write_cache(
    path: Path,
    payload: dict,
    lookback_days: int,
    as_of_date: str | None,
    fetched_at: str | None = None,
) -> None:
    record = {
        "version": _CACHE_VERSION,
        "fetched_at": fetched_at or datetime.now().isoformat(),
        "as_of_date": as_of_date,
        "lookback_days": lookback_days,
        "payload": payload,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return


def _latest_market_close_date() -> str | None:
    try:
        probe = yf_download(
            tickers=_CLOSE_PROBE_TICKER,
            period="10d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if probe is None or probe.empty:
            return None
        idx = pd.to_datetime(probe.index, errors="coerce").dropna()
        if idx.empty:
            return None
        return str(idx[-1].date().isoformat())
    except Exception:
        return None


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _normalize_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notna()]
    out = out.dropna()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _build_fred_client() -> tuple[Fred | None, str | None]:
    if not FRED_AVAILABLE:
        return None, "fredapi not installed; FRED data unavailable."
    api_key = (os.environ.get("FRED_API_KEY") or "").strip()
    if not api_key:
        return None, "FRED_API_KEY not configured; FRED data unavailable."
    try:
        return Fred(api_key=api_key), None
    except Exception as exc:
        return None, f"Failed to initialize FRED client: {exc}"


def _fetch_fred_series(fred: Fred, series_id: str) -> pd.Series | None:
    try:
        raw = fred_get_series(fred, series_id)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    normalized = _normalize_series(raw)
    return normalized if not normalized.empty else None


# ---------------------------------------------------------------------------
# Bulk fetchers - one HTTP call returns multiple tenors
# ---------------------------------------------------------------------------


def _fetch_mof_japan() -> dict[str, pd.Series]:
    """Download JGB yields from Japan Ministry of Finance."""
    try:
        resp = requests.get(MOF_JGB_URL, timeout=30)
        resp.raise_for_status()
    except Exception:
        return {}
    text = None
    for enc in ("utf-8", "shift_jis", "latin-1"):
        try:
            text = resp.content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return {}
    try:
        df = pd.read_csv(io.StringIO(text), header=1)
    except Exception:
        return {}
    if df.empty:
        return {}
    date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], format="%Y/%m/%d", errors="coerce")
    result: dict[str, pd.Series] = {}
    for col in df.columns:
        tenor = _MOF_COLUMN_MAP.get(str(col).strip())
        if tenor is None:
            continue
        s = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=dates)
        ns = _normalize_series(s)
        if not ns.empty:
            result[tenor] = ns
    return result


def _fetch_boe_gilts() -> dict[str, pd.Series]:
    """Download UK gilt spot yields from BoE nominal yield curve ZIP."""
    try:
        resp = requests.get(
            BOE_GLC_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception:
        return {}
    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
    except Exception:
        return {}
    result: dict[str, pd.Series] = {}
    for name in sorted(z.namelist()):
        if not name.endswith(".xlsx"):
            continue
        try:
            with z.open(name) as f:
                _parse_boe_spot_curve(f, result)
        except Exception:
            continue
    return result


def _parse_bundesbank_csv(text: str) -> pd.Series | None:
    """Parse Bundesbank SDMX/BBK CSV payload into a normalized time series."""
    if not text.strip():
        return None

    # Bundesbank CSV payloads are typically ';' delimited with metadata rows.
    df: pd.DataFrame | None = None
    for sep in (";", ","):
        try:
            parsed = pd.read_csv(io.StringIO(text), sep=sep, comment="#", engine="python")
        except Exception:
            continue
        if parsed.empty:
            continue
        df = parsed
        break
    if df is None or df.empty:
        return None

    cols = {str(c).strip().lower(): c for c in df.columns}
    date_col = None
    value_col = None

    for candidate in ("time_period", "date", "time", "zeitraum"):
        if candidate in cols:
            date_col = cols[candidate]
            break
    for candidate in ("obs_value", "value", "wert"):
        if candidate in cols:
            value_col = cols[candidate]
            break

    if date_col is None:
        return None

    if value_col is None:
        # Fallback to the right-most column that parses to numeric values.
        for col in reversed(list(df.columns)):
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                value_col = col
                break
    if value_col is None:
        return None

    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    if values.notna().sum() == 0:
        values = pd.to_numeric(df[value_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out = _normalize_series(pd.Series(values.values, index=dates.values))
    return out if not out.empty else None


def _fetch_bundesbank_series(ts_id: str) -> pd.Series | None:
    """Fetch one Bundesbank time series via /data/{flowRef}/{key}."""
    if "." not in ts_id:
        return None
    flow_ref, key = ts_id.split(".", 1)
    url = f"{BUNDESBANK_BASE_URL}/{flow_ref}/{key}"
    try:
        resp = requests.get(
            url,
            headers={"Accept": "text/csv, application/vnd.sdmx.data+csv;version=1.0.0"},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception:
        return None
    return _parse_bundesbank_csv(resp.text)


def _fetch_bundesbank_germany() -> dict[str, pd.Series]:
    """Download Germany sovereign curve points from Bundesbank."""
    result: dict[str, pd.Series] = {}
    for tenor, ts_id in _BUNDESBANK_DE_SERIES.items():
        series = _fetch_bundesbank_series(ts_id)
        if series is not None and not series.empty:
            result[tenor] = series
    return result


def _parse_boe_spot_curve(f, result: dict[str, pd.Series]) -> None:
    """Parse a single BoE GLC nominal Excel file, appending to *result*."""
    df = pd.read_excel(f, sheet_name="4. spot curve", header=None)

    # Find the row starting with "years:"
    mat_row_idx: int | None = None
    for i in range(min(10, len(df))):
        if str(df.iloc[i, 0]).strip().lower() == "years:":
            mat_row_idx = i
            break
    if mat_row_idx is None:
        return

    # Build column -> tenor mapping from the maturity row
    maturities = df.iloc[mat_row_idx, 1:]
    col_to_tenor: dict[int, str] = {}
    for offset, val in enumerate(maturities):
        try:
            years = float(val)
        except (ValueError, TypeError):
            continue
        tenor = _BOE_MATURITY_MAP.get(years)
        if tenor is not None:
            col_to_tenor[offset + 1] = tenor  # +1 for date column

    # Extract data rows
    data = df.iloc[mat_row_idx + 1 :]
    dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    for col_idx, tenor in col_to_tenor.items():
        values = pd.to_numeric(data.iloc[:, col_idx], errors="coerce")
        s = pd.Series(values.values, index=dates.values)
        ns = _normalize_series(s)
        if ns.empty:
            continue
        if tenor in result:
            result[tenor] = pd.concat([result[tenor], ns])
            result[tenor] = result[tenor][~result[tenor].index.duplicated(keep="last")]
            result[tenor] = result[tenor].sort_index()
        else:
            result[tenor] = ns


def _value_on_or_before(series: pd.Series, target: pd.Timestamp) -> tuple[float | None, str | None]:
    if series.empty:
        return None, None
    eligible = series[series.index <= target]
    if eligible.empty:
        return None, None
    ts = eligible.index[-1]
    val = float(eligible.iloc[-1])
    return val, ts.date().isoformat()


def _build_country_curve(
    country_code: str,
    country_name: str,
    lookback_days: int,
    fred_client: Fred | None,
    fred_unavailable_warning: str | None,
) -> dict:
    warnings: list[str] = []
    missing_unconfigured: list[str] = []
    series_map: dict[str, tuple[pd.Series, str]] = {}

    fred_map = FRED_SERIES.get(country_code, {})

    # Fetch bulk data for non-FRED sources
    bulk_data: dict[str, pd.Series] = {}
    bulk_label = ""
    bulk_configured: set[str] = set()

    if country_code == "JP":
        bulk_data = _fetch_mof_japan()
        bulk_label = "mof"
        bulk_configured = set(_MOF_COLUMN_MAP.values())
        if not bulk_data:
            warnings.append("MOF Japan: could not fetch JGB yields.")
    elif country_code == "UK":
        bulk_data = _fetch_boe_gilts()
        bulk_label = "boe"
        bulk_configured = set(_BOE_MATURITY_MAP.values())
        if not bulk_data:
            warnings.append("BoE: could not fetch gilt yields.")
    elif country_code == "DE":
        bulk_data = _fetch_bundesbank_germany()
        bulk_label = "bundesbank"
        bulk_configured = set(_BUNDESBANK_DE_SERIES.keys())
        if not bulk_data:
            warnings.append("Bundesbank: could not fetch German sovereign yields.")

    for tenor_meta in TENOR_ORDER:
        tenor = str(tenor_meta["tenor"])

        series: pd.Series | None = None
        source: str | None = None

        # FRED (US)
        fred_series_id = fred_map.get(tenor)
        if fred_series_id is not None:
            if fred_client is None:
                if fred_unavailable_warning:
                    warnings.append(fred_unavailable_warning)
            else:
                fetched = _fetch_fred_series(fred_client, fred_series_id)
                if fetched is not None:
                    series = fetched
                    source = f"fred:{fred_series_id}"
                else:
                    warnings.append(f"{tenor}: FRED series {fred_series_id} returned no usable data.")

        # Bulk source (MOF / BoE / Bundesbank)
        if series is None and bulk_configured:
            bulk_series = bulk_data.get(tenor)
            if bulk_series is not None:
                series = bulk_series
                source = f"{bulk_label}:{tenor}"
            elif tenor in bulk_configured:
                warnings.append(f"{tenor}: {bulk_label.upper()} returned no usable data.")

        if series is not None and source is not None:
            series_map[tenor] = (series, source)
            continue

        if tenor not in fred_map and tenor not in bulk_configured:
            missing_unconfigured.append(tenor)

    if missing_unconfigured:
        warnings.append("No configured source for tenors: " + ", ".join(missing_unconfigured) + ".")

    as_of: pd.Timestamp | None = None
    for series, _ in series_map.values():
        last_date = pd.Timestamp(series.index[-1])
        if as_of is None or last_date > as_of:
            as_of = last_date

    historical_target = as_of - pd.Timedelta(days=lookback_days) if as_of is not None else None
    historical_target_1y = as_of - pd.Timedelta(days=365) if as_of is not None else None

    points: list[dict] = []
    for tenor_meta in TENOR_ORDER:
        tenor = str(tenor_meta["tenor"])
        years = TENOR_TO_YEARS[tenor]
        series_pair = series_map.get(tenor)

        if series_pair is None or as_of is None or historical_target is None or historical_target_1y is None:
            points.append(
                {
                    "tenor": tenor,
                    "years": years,
                    "current": None,
                    "historical": None,
                    "change_bps": None,
                    "current_date": None,
                    "historical_date": None,
                    "source_current": None,
                    "source_historical": None,
                    "historical_1y": None,
                    "change_bps_1y": None,
                    "historical_date_1y": None,
                    "source_historical_1y": None,
                }
            )
            continue

        series, source = series_pair
        current_val, current_date = _value_on_or_before(series, as_of)
        historical_val, historical_date = _value_on_or_before(series, historical_target)
        historical_1y_val, historical_1y_date = _value_on_or_before(series, historical_target_1y)

        if current_val is None:
            warnings.append(f"{tenor}: no observation found on or before as-of date.")
        if historical_val is None:
            warnings.append(f"{tenor}: no observation found on or before {historical_target.date().isoformat()}.")
        if historical_1y_val is None:
            warnings.append(
                f"{tenor}: no observation found on or before {historical_target_1y.date().isoformat()} (1Y)."
            )

        change_bps = None
        if current_val is not None and historical_val is not None:
            change_bps = round((current_val - historical_val) * 100.0, 1)

        change_bps_1y = None
        if current_val is not None and historical_1y_val is not None:
            change_bps_1y = round((current_val - historical_1y_val) * 100.0, 1)

        points.append(
            {
                "tenor": tenor,
                "years": years,
                "current": round(current_val, 4) if current_val is not None else None,
                "historical": round(historical_val, 4) if historical_val is not None else None,
                "change_bps": change_bps,
                "current_date": current_date,
                "historical_date": historical_date,
                "source_current": source if current_val is not None else None,
                "source_historical": source if historical_val is not None else None,
                "historical_1y": round(historical_1y_val, 4) if historical_1y_val is not None else None,
                "change_bps_1y": change_bps_1y,
                "historical_date_1y": historical_1y_date,
                "source_historical_1y": source if historical_1y_val is not None else None,
            }
        )

    return {
        "code": country_code,
        "name": country_name,
        "as_of_date": as_of.date().isoformat() if as_of is not None else None,
        "historical_target_date": (historical_target.date().isoformat() if historical_target is not None else None),
        "historical_target_date_1y": (
            historical_target_1y.date().isoformat() if historical_target_1y is not None else None
        ),
        "points": points,
        "warnings": _dedupe(warnings),
    }


def _build_yield_curve_result(lookback_days: int, countries_to_fetch: list[tuple[str, str]]) -> dict[str, Any]:
    fred_client, fred_warn = _build_fred_client()

    countries: list[dict[str, Any]] = []
    for code, name in countries_to_fetch:
        countries.append(
            _build_country_curve(
                country_code=code,
                country_name=name,
                lookback_days=lookback_days,
                fred_client=fred_client,
                fred_unavailable_warning=fred_warn,
            )
        )

    return {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "lookback_days": lookback_days,
        "tenor_order": TENOR_ORDER,
        "countries": countries,
    }


def get_data(lookback_days: int = 90, country: str | None = None) -> dict:
    """
    Build yield curve snapshot for all supported countries, or one country.

    Returns:
        JSON-serializable dict with canonical tenor axis and country curve data.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    country_code = _normalize_country_code(country)
    countries_to_fetch = [(country_code, COUNTRY_NAME_BY_CODE[country_code])] if country_code is not None else COUNTRIES

    # --- cache check ---
    cache_p = (
        _country_cache_path(lookback_days, country_code) if country_code is not None else _cache_path(lookback_days)
    )
    cached_record = _load_cache(cache_p)
    cached_payload = cached_record.get("payload") if cached_record else None
    cache_decision = None

    if cached_record and isinstance(cached_payload, dict):
        cached_as_of = cached_record.get("as_of_date")
        fetched_at = cached_record.get("fetched_at")
        cache_decision = market_cache_decision(
            cached_as_of=cached_as_of,
            fetched_at=fetched_at,
            ttl_seconds=_CACHE_TTL_SECONDS,
        )
        if cache_decision.action == "probe":
            latest_close = _latest_market_close_date()
            cache_decision = market_cache_decision(
                cached_as_of=cached_as_of,
                fetched_at=fetched_at,
                ttl_seconds=_CACHE_TTL_SECONDS,
                latest_close=latest_close,
                latest_close_probed=True,
            )
        if cache_decision.action == "use_cache":
            if cache_decision.status == "hit_unchanged":
                _write_cache(
                    path=cache_p,
                    payload=cached_payload,
                    lookback_days=lookback_days,
                    as_of_date=str(cached_as_of) if cached_as_of is not None else None,
                    fetched_at=datetime.now().isoformat(),
                )
            return attach_market_cache_metadata(cached_payload, cache_decision.metadata())

    # --- fetch live ---
    try:
        result = _build_yield_curve_result(lookback_days, countries_to_fetch)
    except Exception as exc:
        if isinstance(cached_payload, dict):
            if cache_decision is not None:
                meta = metadata_from_decision(
                    cache_decision,
                    status="stale_fallback",
                    stale=True,
                    reason=f"refresh failed: {exc}",
                )
            else:
                meta = build_market_cache_metadata(
                    status="stale_fallback",
                    stale=True,
                    cached_as_of=cached_record.get("as_of_date") if cached_record else None,
                    reason=f"refresh failed: {exc}",
                    cache_ttl_seconds=_CACHE_TTL_SECONDS,
                )
            return attach_market_cache_metadata(cached_payload, meta)
        raise

    # Determine as_of_date from country curves (latest across all)
    as_of_date = None
    country_rows = result.get("countries", [])
    if isinstance(country_rows, list):
        for country in country_rows:
            if not isinstance(country, dict):
                continue
            d = country.get("as_of_date")
            if isinstance(d, str) and (as_of_date is None or d > as_of_date):
                as_of_date = d

    _write_cache(
        path=cache_p,
        payload=result,
        lookback_days=lookback_days,
        as_of_date=as_of_date,
    )

    return attach_market_cache_metadata(
        result,
        build_market_cache_metadata(
            status="refresh",
            stale=False,
            cached_as_of=as_of_date,
            expected_market_date_value=expected_market_date().isoformat(),
            latest_close=cache_decision.latest_close if cache_decision is not None else None,
            reason=(
                f"refreshed yield curve cache for {country_code}"
                if country_code is not None
                else "refreshed yield curve cache"
            ),
            cache_ttl_seconds=_CACHE_TTL_SECONDS,
        ),
    )


if __name__ == "__main__":
    import json

    print(json.dumps(get_data(), indent=2))
