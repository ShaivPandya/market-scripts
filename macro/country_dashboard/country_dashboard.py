#!/usr/bin/env python3
"""
Country Dashboard

Fetches macroeconomic time series (Inflation, Unemployment, GDP) for 9 countries
primarily from FRED (Canada inflation via Statistics Canada WDS, UK inflation
via ONS). Displays a 3x3 grid
of line charts with metric radio toggles (GUI),
or prints summary tables in the terminal.

Terminal:
  python macro/country_dashboard/country_dashboard.py

GUI:
  Accessed via sidebar in gui/app.py
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from load_env import load_env

load_env()

import pandas as pd
import requests
from fredapi import Fred

warnings.filterwarnings("ignore")

_DEFAULT_STATCAN_CPI_CANADA_VECTOR_ID = 41690973
try:
    STATCAN_CPI_CANADA_VECTOR_ID = int(
        os.environ.get(
            "STATCAN_CPI_CANADA_VECTOR_ID",
            str(_DEFAULT_STATCAN_CPI_CANADA_VECTOR_ID),
        )
    )
except ValueError:
    STATCAN_CPI_CANADA_VECTOR_ID = _DEFAULT_STATCAN_CPI_CANADA_VECTOR_ID

_DEFAULT_ONS_CPI_UK_SERIES_ID = "d7g7"
_DEFAULT_ONS_CPI_UK_DATASET_ID = "mm23"
ONS_CPI_UK_SERIES_ID = os.environ.get(
    "ONS_CPI_UK_SERIES_ID", _DEFAULT_ONS_CPI_UK_SERIES_ID
)
ONS_CPI_UK_DATASET_ID = os.environ.get(
    "ONS_CPI_UK_DATASET_ID", _DEFAULT_ONS_CPI_UK_DATASET_ID
)

# ── Country definitions: display_name -> FRED series IDs per metric ──────────
# For inflation, we prefer direct YoY series and keep fallbacks where needed.
COUNTRIES = {
    "US": {
        "inflation": [
            # Match FRED's "Percent Change from Year Ago" view exactly.
            {"id": "CPIAUCSL", "transform": "none", "params": {"units": "pc1"}},
        ],
        "unemployment": "UNRATE",
        "gdp": "NAEXKP01USQ189S",
    },
    "Canada": {
        "inflation": [
            # Statistics Canada WDS: CPI (2002=100), All-items, Canada.
            # We compute YoY % change to match the dashboard's inflation definition.
            {
                "source": "statcan_wds",
                "id": f"STATCAN v{STATCAN_CPI_CANADA_VECTOR_ID}",
                "vector_id": STATCAN_CPI_CANADA_VECTOR_ID,
                "transform": "yoy12",
            },
        ],
        "unemployment": "LRUNTTTTCAM156S",
        "gdp": "NAEXKP01CAQ189S",
    },
    "United Kingdom": {
        "inflation": [
            # ONS: CPI Annual Rate, All Items (2015=100) — already YoY %.
            {
                "source": "ons",
                "id": f"ONS {ONS_CPI_UK_SERIES_ID}/{ONS_CPI_UK_DATASET_ID}",
                "series_id": ONS_CPI_UK_SERIES_ID,
                "dataset_id": ONS_CPI_UK_DATASET_ID,
                "transform": "none",
            },
            # FRED fallbacks
            {"id": "CPALTT01GBM659N", "transform": "none"},
            {"id": "FPCPITOTLZGGBR", "transform": "none"},
        ],
        "unemployment": "LRUNTTTTGBM156S",
        "gdp": "NAEXKP01GBQ189S",
    },
    "EU": {
        "inflation": [
            {"id": "CP0000EZ19M086NEST", "transform": "yoy12"},
            {"id": "FPCPITOTLZGEMU", "transform": "none"},
        ],
        "unemployment": "LRUNTTTTEZM156S",
        "gdp": "NAEXKP01EZQ189S",
    },
    "Germany": {
        "inflation": [
            {"id": "CPALTT01DEM659N", "transform": "none"},
            {"id": "FPCPITOTLZGDEU", "transform": "none"},
        ],
        "unemployment": "LRUNTTTTDEM156S",
        "gdp": "NAEXKP01DEQ189S",
    },
    "Japan": {
        "inflation": [
            {"id": "FPCPITOTLZGJPN", "transform": "none"},
            {"id": "CPALTT01JPM659N", "transform": "none"},
        ],
        "unemployment": "LRUNTTTTJPM156S",
        "gdp": "NAEXKP01JPQ189S",
    },
    "France": {
        "inflation": [
            {"id": "CPALTT01FRM659N", "transform": "none"},
            {"id": "FPCPITOTLZGFRA", "transform": "none"},
        ],
        "unemployment": "LRUNTTTTFRM156S",
        "gdp": "NAEXKP01FRQ189S",
    },
    "Switzerland": {
        "inflation": [
            {"id": "CPALTT01CHM659N", "transform": "none"},
            {"id": "FPCPITOTLZGCHE", "transform": "none"},
        ],
        "unemployment": "LRUNTTTTCHM156S",
        "gdp": "NAEXKP01CHQ189S",
    },
    "Australia": {
        "inflation": [
            {"id": "CPALTT01AUQ659N", "transform": "none"},
            {"id": "FPCPITOTLZGAUS", "transform": "none"},
        ],
        "unemployment": "LRUNTTTTAUM156S",
        "gdp": "NAEXKP01AUQ189S",
    },
}

COUNTRY_ORDER = list(COUNTRIES.keys())

METRICS = ["Inflation", "Unemployment", "GDP"]

_FETCH_YEARS = 6
_DISPLAY_YEARS = 5


# ── Data fetching ────────────────────────────────────────────────────────────

def _statcan_wds_post(method: str, payload: dict, timeout: int = 20) -> dict:
    """
    Minimal Statistics Canada Web Data Service (WDS) client.

    Docs: https://www.statcan.gc.ca/en/developers/wds
    """
    base_url = os.environ.get(
        "STATCAN_WDS_BASE_URL",
        "https://www150.statcan.gc.ca/t1/wds/rest",
    ).rstrip("/")
    url = f"{base_url}/{method}"

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        if not data:
            raise RuntimeError("Statistics Canada WDS returned an empty list response")
        if not all(isinstance(item, dict) for item in data):
            raise RuntimeError("Statistics Canada WDS returned an unexpected list response")

        for item in data:
            status = item.get("status")
            if status and status != "SUCCESS":
                message = item.get("message") or item.get("object") or item
                raise RuntimeError(f"Statistics Canada WDS error: {message}")

        if len(data) == 1:
            data = data[0]
        else:
            data = {"status": "SUCCESS", "object": [item.get("object") for item in data]}

    status = data.get("status")
    if status and status != "SUCCESS":
        message = data.get("message") or data.get("object") or data
        raise RuntimeError(f"Statistics Canada WDS error: {message}")

    return data


def _fetch_statcan_vector_latest_n(
    *,
    vector_id: int,
    latest_n: int,
    timeout: int = 10,
) -> pd.Series:
    """
    Fetch the latest N observations for a StatCan vector.

    Returns a pd.Series indexed by datetime.
    """
    vector_id_int = int(vector_id)

    # StatCan WDS docs specify: POST array with numeric vectorId and latestN.
    payload = [{"vectorId": vector_id_int, "latestN": int(latest_n)}]
    data = _statcan_wds_post(
        "getDataFromVectorsAndLatestNPeriods",
        payload=payload,
        timeout=timeout,
    )

    obj = data.get("object")
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list) or not obj:
        raise RuntimeError("Statistics Canada WDS returned no data object")

    def _matches_vector(b: dict) -> bool:
        raw = b.get("vectorId") or b.get("vector_id") or b.get("vector") or ""
        s = str(raw).lstrip("v")
        return s == str(vector_id_int)

    block = obj[0] if len(obj) == 1 else next((b for b in obj if _matches_vector(b)), obj[0])
    points = (
        block.get("vectorDataPoint")
        or block.get("vectorDataPoints")
        or block.get("dataPoints")
        or []
    )
    if not isinstance(points, list) or not points:
        raise RuntimeError("Statistics Canada WDS returned no datapoints")

    dates = []
    values = []
    for p in points:
        ref = p.get("refPer") or p.get("refper") or p.get("ref_period")
        val = p.get("value")
        if ref is None or val in (None, ""):
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue

        dt = pd.to_datetime(ref, errors="coerce")
        if pd.isna(dt) and isinstance(ref, str):
            if ref.isdigit() and len(ref) == 6:
                dt = pd.to_datetime(ref + "01", format="%Y%m%d", errors="coerce")
            elif ref.isdigit() and len(ref) == 8:
                dt = pd.to_datetime(ref, format="%Y%m%d", errors="coerce")

        if pd.isna(dt):
            continue

        dates.append(dt)
        values.append(num)

    if not dates:
        raise RuntimeError("Statistics Canada WDS datapoints could not be parsed")

    series = pd.Series(values, index=pd.to_datetime(dates)).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series


def _fetch_ons_timeseries(
    *,
    series_id: str,
    dataset_id: str,
    timeout: int = 20,
) -> pd.Series:
    """
    Fetch monthly time series data from the ONS website API.

    Returns a pd.Series indexed by datetime.
    """
    base_url = os.environ.get(
        "ONS_API_BASE_URL",
        "https://www.ons.gov.uk",
    ).rstrip("/")
    url = (
        f"{base_url}/economy/inflationandpriceindices"
        f"/timeseries/{series_id}/{dataset_id}/data"
    )

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    months = data.get("months")
    if not months:
        raise RuntimeError(
            f"ONS timeseries {series_id}/{dataset_id}: no monthly data returned"
        )

    dates = []
    values = []
    for point in months:
        val_str = point.get("value", "").strip()
        if not val_str:
            continue
        try:
            val = float(val_str)
        except (TypeError, ValueError):
            continue

        date_str = point.get("date", "")
        dt = pd.to_datetime(date_str, format="%Y %b", errors="coerce")
        if pd.isna(dt):
            year = point.get("year", "")
            month_name = point.get("month", "")
            if year and month_name:
                dt = pd.to_datetime(
                    f"{year} {month_name}", format="%Y %B", errors="coerce"
                )
        if pd.isna(dt):
            continue

        dates.append(dt)
        values.append(val)

    if not dates:
        raise RuntimeError(
            f"ONS timeseries {series_id}/{dataset_id}: "
            "no data points could be parsed"
        )

    series = pd.Series(values, index=pd.to_datetime(dates)).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series


def _get_fred_client():
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FRED_API_KEY environment variable")
    return Fred(api_key=api_key)


def _apply_transform(series: pd.Series, transform: str) -> pd.Series:
    if transform == "none":
        return series
    if transform == "yoy12":
        return series.pct_change(12) * 100
    if transform == "yoy4":
        return series.pct_change(4) * 100
    raise ValueError(f"Unknown transform: {transform}")


def _metric_candidates(metric_key: str, config: dict) -> List[Dict[str, object]]:
    metric_config = config[metric_key]

    if isinstance(metric_config, str):
        transform = "yoy4" if metric_key == "gdp" else "none"
        return [{"source": "fred", "id": metric_config, "transform": transform, "params": {}}]

    candidates: List[Dict[str, object]] = []
    for item in metric_config:
        if isinstance(item, str):
            candidates.append({"source": "fred", "id": item, "transform": "none", "params": {}})
        else:
            candidate: Dict[str, object] = {
                "source": item.get("source", "fred"),
                "id": item["id"],
                "transform": item.get("transform", "none"),
                "params": item.get("params", {}),
            }
            for k, v in item.items():
                if k not in candidate:
                    candidate[k] = v
            candidates.append(candidate)
    return candidates


def fetch_country_data(metric: str = "Inflation") -> dict:
    """
    Fetch macroeconomic time series for all 9 countries.

    Returns dict with:
        countries – dict[country_name] -> pd.Series
        metric    – str
        timestamp – datetime
        error     – str (only on failure)
    """
    metric_key = metric.lower()
    if metric_key not in ("inflation", "unemployment", "gdp"):
        return {"error": f"Invalid metric: {metric}"}

    try:
        fred = _get_fred_client()
    except RuntimeError as e:
        return {"error": str(e)}

    now = datetime.now()
    observation_start = now.replace(year=now.year - _FETCH_YEARS)
    display_start = now.replace(year=now.year - _DISPLAY_YEARS)

    countries = {}
    errors = {}
    series_used = {}
    latest_observation_dates = {}

    for name, config in COUNTRIES.items():
        country_errors = []
        candidates = _metric_candidates(metric_key, config)

        for candidate in candidates:
            source = candidate.get("source", "fred")
            series_id = candidate["id"]
            transform = candidate["transform"]
            params = candidate.get("params", {})
            try:
                if source == "statcan_wds":
                    vector_id = candidate.get("vector_id")
                    if not vector_id:
                        raise ValueError("Missing vector_id for statcan_wds series")
                    series = _fetch_statcan_vector_latest_n(
                        vector_id=int(vector_id),
                        latest_n=int((_FETCH_YEARS + 1) * 12),
                    )
                    series = series[series.index >= observation_start]
                elif source == "ons":
                    sid = candidate.get("series_id")
                    did = candidate.get("dataset_id")
                    if not sid or not did:
                        raise ValueError(
                            "Missing series_id or dataset_id for ONS source"
                        )
                    series = _fetch_ons_timeseries(
                        series_id=sid,
                        dataset_id=did,
                    )
                    series = series[series.index >= observation_start]
                else:
                    series = fred.get_series(
                        series_id,
                        observation_start=observation_start,
                        **params,
                    )
                if series is None or series.empty:
                    country_errors.append(f"{series_id}: no observations returned")
                    continue

                series = series.dropna()
                series.index = pd.to_datetime(series.index)
                series = _apply_transform(series, transform).dropna()
                series = series[series.index >= display_start]

                if series.empty:
                    country_errors.append(
                        f"{series_id}: no data in last {_DISPLAY_YEARS} years"
                    )
                    continue

                countries[name] = series
                series_used[name] = series_id
                latest_observation_dates[name] = series.index[-1].to_pydatetime()
                break
            except Exception as e:
                country_errors.append(f"{series_id}: {type(e).__name__}: {e}")
        else:
            errors[name] = country_errors or ["No series configured"]

    return {
        "countries": countries,
        "errors": errors,
        "series_used": series_used,
        "latest_observation_dates": latest_observation_dates,
        "metric": metric,
        "timestamp": datetime.now(),
    }


def get_data(metric: str = "Inflation") -> dict:
    """GUI-facing entry point."""
    return fetch_country_data(metric=metric)


def format_value(value: float) -> str:
    return f"{value:.1f}%"


# ── Terminal output ──────────────────────────────────────────────────────────

def print_terminal():
    """Print Country dashboard results for all metrics."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    header = Panel(
        "[bold white]COUNTRY DASHBOARD[/bold white]\n"
        f"[dim]Data from FRED (Canada inflation via Statistics Canada WDS, UK inflation via ONS)[/dim]\n"
        f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="bold blue",
        padding=(1, 2),
    )
    console.print(header)

    for metric_name in METRICS:
        console.print(f"\n[bold yellow]Fetching {metric_name} data...[/bold yellow]")
        data = fetch_country_data(metric=metric_name)

        if "error" in data:
            console.print(f"[red]Error: {data['error']}[/red]")
            continue

        countries = data.get("countries", {})
        if not countries:
            console.print("[yellow]No data returned[/yellow]")
            continue

        table = Table(
            title=f"Country Dashboard — {metric_name}",
            show_header=True,
            header_style="bold cyan",
            title_style="bold white",
            border_style="blue",
        )
        table.add_column("Country", style="bold white", min_width=14)
        table.add_column("Latest", justify="right", min_width=10)
        table.add_column("Previous", justify="right", min_width=10)
        table.add_column("Change", justify="right", min_width=10)

        for name in COUNTRY_ORDER:
            series = countries.get(name)
            if series is None or series.empty:
                table.add_row(name, "N/A", "N/A", "N/A")
                continue

            latest = series.iloc[-1]
            previous = series.iloc[-2] if len(series) > 1 else latest
            change = latest - previous

            # Lower inflation/unemployment = good (green); higher GDP = good
            if metric_name == "GDP":
                style = "green" if change >= 0 else "red"
            else:
                style = "green" if change <= 0 else "red"

            table.add_row(
                name,
                format_value(latest),
                format_value(previous),
                f"[{style}]{change:+.1f}pp[/{style}]",
            )

        console.print(table)

    console.print()


def main():
    print_terminal()


if __name__ == "__main__":
    main()
