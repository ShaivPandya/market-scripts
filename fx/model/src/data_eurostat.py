from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests


class EurostatError(RuntimeError):
    pass


def _eurostat_base_url() -> str:
    import os

    return os.environ.get(
        "EUROSTAT_API_BASE_URL",
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0",
    ).rstrip("/")


def _fetch_eurostat_jsonstat_series(
    *,
    dataset: str,
    query_params: dict,
    freq: str,
    timeout: int = 30,
) -> pd.Series:
    base_url = _eurostat_base_url()
    qs = "&".join(f"{k}={v}" for k, v in query_params.items())
    url = f"{base_url}/data/{dataset}?format=JSON&{qs}"

    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise EurostatError(f"Eurostat request failed ({resp.status_code}): {resp.text[:2000]}")
    data = resp.json()

    time_dim = data.get("dimension", {}).get("time", {})
    time_index = time_dim.get("category", {}).get("index", {})
    if not time_index:
        raise EurostatError(f"Eurostat {dataset}: no time dimension in response")

    raw_values = data.get("value", {})
    if not raw_values:
        raise EurostatError(f"Eurostat {dataset}: no values in response")

    dates = []
    values = []
    for period, pos in sorted(time_index.items(), key=lambda x: x[1]):
        val = raw_values.get(str(pos))
        if val is None:
            continue

        if freq == "quarterly":
            p = pd.Period(period, freq="Q")
            dt = p.to_timestamp(how="end")
        else:
            dt = pd.to_datetime(period, format="%Y-%m", errors="coerce")
            if pd.isna(dt):
                continue

        dates.append(dt)
        values.append(float(val))

    if not dates:
        raise EurostatError(f"Eurostat {dataset}: no data points could be parsed")

    series = pd.Series(values, index=pd.to_datetime(dates), name=dataset).sort_index()
    return series[~series.index.duplicated(keep="last")]


def fetch_eurostat_current_account_pct_gdp(
    *,
    geo: str,
    start: str,
    cache_dir: Path,
    refresh: bool = False,
    dataset: str = "teibp051",
    timeout: int = 30,
) -> pd.Series:
    """Fetch Current Account balance (% of GDP) for a Eurostat geo.

    Uses Eurostat table `teibp051` by default (quarterly, % of GDP).
    The resulting series is suitable as a replacement for IMF's BCA_NGDPD
    (current account balance as % of GDP).
    """
    cache_path = cache_dir / f"eurostat_{dataset}_CA_PC_GDP_{geo}.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        s = pd.to_numeric(df["value"], errors="coerce")
        series = pd.Series(s.values, index=df["date"], name=f"EUROSTAT_{dataset}_CA_{geo}").sort_index()
        return series[series.index >= pd.Timestamp(start)]

    query_params = {
        "geo": geo,
        "freq": "Q",
        "unit": "PC_GDP",
        "s_adj": "NSA",
        "stk_flow": "BAL",
        "bop_item": "CA",
        "partner": "WRL_REST",
    }
    series = _fetch_eurostat_jsonstat_series(
        dataset=dataset,
        query_params=query_params,
        freq="quarterly",
        timeout=timeout,
    ).rename(f"EUROSTAT_{dataset}_CA_{geo}")
    series = series[series.index >= pd.Timestamp(start)]

    if series.empty:
        raise EurostatError(f"Eurostat {dataset}: no observations returned for geo={geo} from {start}")

    pd.DataFrame({"date": series.index, "value": series.values}).to_csv(cache_path, index=False)
    return series


def fetch_euro_area_current_account_pct_gdp(
    *,
    start: str,
    cache_dir: Path,
    refresh: bool = False,
    timeout: int = 30,
) -> pd.Series:
    """Fetch Euro area CA%GDP with sensible fallbacks (EA20 -> EA19)."""
    last_err: Optional[Exception] = None
    for geo in ("EA20", "EA19"):
        try:
            return fetch_eurostat_current_account_pct_gdp(
                geo=geo,
                start=start,
                cache_dir=cache_dir,
                refresh=refresh,
                dataset="teibp051",
                timeout=timeout,
            )
        except Exception as e:
            last_err = e
            continue
    raise EurostatError("Eurostat CA%GDP fetch failed for EA20 and EA19") from last_err

