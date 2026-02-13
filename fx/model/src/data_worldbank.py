"""Fetch CPI data from the World Bank API (annual, interpolated to monthly)."""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

WB_BASE = "https://api.worldbank.org/v2"

# ISO2 codes for World Bank API, keyed by ISO3
ISO3_TO_ISO2 = {
    "USA": "US",
    "CAN": "CA",
    "GBR": "GB",
    "AUS": "AU",
    "JPN": "JP",
}


class WorldBankError(RuntimeError):
    pass


def fetch_worldbank_cpi(
    iso3: str,
    start_year: int = 1960,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.Series:
    """Fetch annual CPI index from World Bank and interpolate to monthly.

    Uses indicator CPTOTSAXN (CPI: All Items, SA, Index).
    Returns a monthly pd.Series with DatetimeIndex, suitable for extending
    a stale FRED CPI series.
    """
    iso2 = ISO3_TO_ISO2.get(iso3)
    if not iso2:
        raise WorldBankError(f"Unknown ISO3 code: {iso3}. Add mapping to ISO3_TO_ISO2.")

    cache_path = cache_dir / f"wb_CPI_{iso3}.csv" if cache_dir else None
    if cache_path and cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        s = pd.to_numeric(df["value"], errors="coerce")
        return pd.Series(s.values, index=df["date"], name=f"wb_cpi_{iso3}").sort_index()

    url = (
        f"{WB_BASE}/country/{iso2}/indicator/CPTOTSAXN"
        f"?format=json&per_page=200&date={start_year}:2030"
    )
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise WorldBankError(f"World Bank request failed ({r.status_code})")

    j = r.json()
    if len(j) < 2 or not j[1]:
        raise WorldBankError(f"No CPI data returned for {iso3}")

    records = [(d["date"], d["value"]) for d in j[1] if d["value"] is not None]
    if not records:
        raise WorldBankError(f"All CPI values are null for {iso3}")

    annual = pd.Series(
        {pd.Timestamp(f"{yr}-12-31"): float(val) for yr, val in records}
    ).sort_index()

    # Interpolate to monthly: resample to month-end, then interpolate
    monthly = annual.resample("ME").interpolate(method="linear")
    monthly = monthly.dropna()

    # Cache
    if cache_path:
        out = pd.DataFrame({"date": monthly.index, "value": monthly.values})
        out.to_csv(cache_path, index=False)

    return pd.Series(
        monthly.values, index=monthly.index, name=f"wb_cpi_{iso3}"
    ).sort_index()
