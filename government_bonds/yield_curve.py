"""
Yield curve data snapshot for the web dashboard.

Provides current and historical (N days lookback) curve points for:
- United States (FRED)
- United Kingdom (local CSV fallback)
- Germany (local CSV fallback)
- Japan (local CSV fallback)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Load environment variables from .env file when run standalone.
sys.path.insert(0, str(Path(__file__).parent.parent))
from load_env import load_env

load_env()

try:
    from fredapi import Fred

    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    Fred = None  # type: ignore[assignment]


TENOR_ORDER = [
    {"tenor": "3M", "years": 0.25},
    {"tenor": "6M", "years": 0.50},
    {"tenor": "1Y", "years": 1.00},
    {"tenor": "2Y", "years": 2.00},
    {"tenor": "5Y", "years": 5.00},
    {"tenor": "10Y", "years": 10.00},
    {"tenor": "30Y", "years": 30.00},
]

TENOR_TO_YEARS = {row["tenor"]: row["years"] for row in TENOR_ORDER}

COUNTRIES = [
    ("US", "United States"),
    ("UK", "United Kingdom"),
    ("DE", "Germany"),
    ("JP", "Japan"),
]

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

CSV_SERIES = {
    "UK": {
        "2Y": "Download Data - BOND_BX_XTUP_TMBMKGB-02Y.csv",
        "10Y": "Download Data - BOND_BX_XTUP_TMBMKGB-10Y.csv",
    },
    "DE": {
        "2Y": "Download Data - BOND_BX_XTUP_TMBMKDE-02Y.csv",
        "10Y": "Download Data - BOND_BX_XTUP_TMBMKDE-10Y.csv",
    },
    "JP": {
        "2Y": "Download Data - BOND_BX_XTUP_TMBMKJP-02Y.csv",
        "10Y": "Download Data - BOND_BX_XTUP_TMBMKJP-10Y.csv",
    },
}

DATA_DIR = Path(__file__).resolve().parent / "data"


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _to_float_percent(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "").replace(",", "")
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notna()]
    out = out.dropna()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _build_fred_client() -> tuple[Optional[Fred], Optional[str]]:
    if not FRED_AVAILABLE:
        return None, "fredapi not installed; FRED data unavailable."
    api_key = (os.environ.get("FRED_API_KEY") or "").strip()
    if not api_key:
        return None, "FRED_API_KEY not configured; FRED data unavailable."
    try:
        return Fred(api_key=api_key), None
    except Exception as exc:
        return None, f"Failed to initialize FRED client: {exc}"


def _fetch_fred_series(fred: Fred, series_id: str) -> Optional[pd.Series]:
    try:
        raw = fred.get_series(series_id)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    normalized = _normalize_series(raw)
    return normalized if not normalized.empty else None


def _load_csv_series(filename: str) -> Optional[pd.Series]:
    path = DATA_DIR / filename
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "Date" not in df.columns:
        return None

    value_col = "Close" if "Close" in df.columns else "Yield" if "Yield" in df.columns else None
    if value_col is None:
        return None

    dates = pd.to_datetime(df["Date"], errors="coerce")
    values = df[value_col].map(_to_float_percent)
    series = pd.Series(values.values, index=dates)
    normalized = _normalize_series(series)
    return normalized if not normalized.empty else None


def _value_on_or_before(series: pd.Series, target: pd.Timestamp) -> tuple[Optional[float], Optional[str]]:
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
    fred_client: Optional[Fred],
    fred_unavailable_warning: Optional[str],
) -> dict:
    warnings: list[str] = []
    missing_unconfigured: list[str] = []
    series_map: dict[str, tuple[pd.Series, str]] = {}

    fred_map = FRED_SERIES.get(country_code, {})
    csv_map = CSV_SERIES.get(country_code, {})

    for tenor_meta in TENOR_ORDER:
        tenor = tenor_meta["tenor"]

        series: Optional[pd.Series] = None
        source: Optional[str] = None

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

        if series is None:
            csv_file = csv_map.get(tenor)
            if csv_file is not None:
                fallback = _load_csv_series(csv_file)
                if fallback is not None:
                    series = fallback
                    source = f"csv:{csv_file}"
                else:
                    warnings.append(f"{tenor}: CSV fallback {csv_file} unavailable or invalid.")

        if series is not None and source is not None:
            series_map[tenor] = (series, source)
            continue

        if tenor not in fred_map and tenor not in csv_map:
            missing_unconfigured.append(tenor)

    if missing_unconfigured:
        warnings.append(
            "No configured source for tenors: " + ", ".join(missing_unconfigured) + "."
        )

    as_of: Optional[pd.Timestamp] = None
    for series, _ in series_map.values():
        last_date = pd.Timestamp(series.index[-1])
        if as_of is None or last_date > as_of:
            as_of = last_date

    historical_target = as_of - pd.Timedelta(days=lookback_days) if as_of is not None else None

    points: list[dict] = []
    for tenor_meta in TENOR_ORDER:
        tenor = tenor_meta["tenor"]
        years = TENOR_TO_YEARS[tenor]
        series_pair = series_map.get(tenor)

        if series_pair is None or as_of is None or historical_target is None:
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
                }
            )
            continue

        series, source = series_pair
        current_val, current_date = _value_on_or_before(series, as_of)
        historical_val, historical_date = _value_on_or_before(series, historical_target)

        if current_val is None:
            warnings.append(f"{tenor}: no observation found on or before as-of date.")
        if historical_val is None:
            warnings.append(
                f"{tenor}: no observation found on or before {historical_target.date().isoformat()}."
            )

        change_bps = None
        if current_val is not None and historical_val is not None:
            change_bps = round((current_val - historical_val) * 100.0, 1)

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
            }
        )

    return {
        "code": country_code,
        "name": country_name,
        "as_of_date": as_of.date().isoformat() if as_of is not None else None,
        "historical_target_date": (
            historical_target.date().isoformat() if historical_target is not None else None
        ),
        "points": points,
        "warnings": _dedupe(warnings),
    }


def get_data(lookback_days: int = 90) -> dict:
    """
    Build yield curve snapshot for all supported countries.

    Returns:
        JSON-serializable dict with canonical tenor axis and country curve data.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    fred_client, fred_warn = _build_fred_client()

    countries: list[dict] = []
    for code, name in COUNTRIES:
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


if __name__ == "__main__":
    import json

    print(json.dumps(get_data(), indent=2))
