#!/usr/bin/env python3
"""
Country Dashboard

Fetches macroeconomic time series (Inflation, Unemployment, GDP) for 9 countries
from FRED. Displays a 3x3 grid of line charts with metric radio toggles (GUI),
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
from fredapi import Fred

warnings.filterwarnings("ignore")

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
            {"id": "CPALTT01CAM659N", "transform": "none"},
            {"id": "FPCPITOTLZGCAN", "transform": "none"},
        ],
        "unemployment": "LRUNTTTTCAM156S",
        "gdp": "NAEXKP01CAQ189S",
    },
    "United Kingdom": {
        "inflation": [
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
        return [{"id": metric_config, "transform": transform, "params": {}}]

    candidates: List[Dict[str, object]] = []
    for item in metric_config:
        if isinstance(item, str):
            candidates.append({"id": item, "transform": "none", "params": {}})
        else:
            candidates.append(
                {
                    "id": item["id"],
                    "transform": item.get("transform", "none"),
                    "params": item.get("params", {}),
                }
            )
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
            series_id = candidate["id"]
            transform = candidate["transform"]
            params = candidate.get("params", {})
            try:
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
        f"[dim]Data from FRED[/dim]\n"
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
