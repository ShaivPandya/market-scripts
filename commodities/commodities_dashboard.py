#!/usr/bin/env python3
"""
Commodities Dashboard

Fetches closing-price time series for 9 major commodities from Yahoo Finance.
Displays a 3x3 grid of line charts with Daily/Weekly/Monthly toggles (GUI),
or prints summary tables in the terminal.

Terminal:
  python commodities/commodities_dashboard.py

GUI:
  Accessed via sidebar in gui/app.py
"""

import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# -- Commodity definitions: display_name -> yfinance ticker ────────────────
COMMODITIES = {
    "Gold":            "GC=F",
    "Silver":          "SI=F",
    "Copper":          "HG=F",
    "Platinum":        "PL=F",
    "Palladium":       "PA=F",
    "Aluminum":        "ALI=F",
    "WTI Crude Oil":   "CL=F",
    "Brent Crude Oil": "BZ=F",
    "Natural Gas":     "NG=F",
}

COMMODITY_ORDER = list(COMMODITIES.keys())

# -- Timeframe configs: name -> yfinance (period, interval) ────────────────
TIMEFRAMES = {
    "Daily":   {"period": "90d",  "interval": "1d"},
    "Weekly":  {"period": "2y",   "interval": "1wk"},
    "Monthly": {"period": "5y",   "interval": "1mo"},
}


# -- Data fetching ─────────────────────────────────────────────────────────

def fetch_commodities_data(timeframe: str = "Daily") -> dict:
    """
    Fetch closing-price time series for all 9 commodities.

    Returns dict with:
        commodities – dict[commodity_name] -> pd.Series (Close prices)
        timeframe   – str
        timestamp   – datetime
        error       – str (only on failure)
    """
    tf = TIMEFRAMES.get(timeframe)
    if tf is None:
        return {"error": f"Invalid timeframe: {timeframe}"}

    tickers = list(COMMODITIES.values())

    try:
        raw = yf.download(
            tickers=tickers,
            period=tf["period"],
            interval=tf["interval"],
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as e:
        return {"error": f"yfinance download failed: {e}"}

    if raw is None or raw.empty:
        return {"error": "No data returned from yfinance"}

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    commodities = {}

    for name, ticker in COMMODITIES.items():
        try:
            if is_multi:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                series = raw[ticker]["Close"].dropna()
            else:
                series = raw["Close"].dropna()

            if series.empty:
                continue

            if hasattr(series.index, "tz") and series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            commodities[name] = series
        except Exception:
            continue

    return {
        "commodities": commodities,
        "timeframe": timeframe,
        "timestamp": datetime.now(),
    }


def get_data(timeframe: str = "Daily") -> dict:
    """GUI-facing entry point."""
    return fetch_commodities_data(timeframe=timeframe)


def format_price(value: float) -> str:
    """Format price based on magnitude for clean display."""
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:.4f}"


# -- Terminal output ───────────────────────────────────────────────────────

def print_terminal():
    """Print Commodities dashboard results for all timeframes."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    header = Panel(
        "[bold white]COMMODITIES DASHBOARD[/bold white]\n"
        f"[dim]Data from Yahoo Finance[/dim]\n"
        f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="bold blue",
        padding=(1, 2),
    )
    console.print(header)

    for tf_name in TIMEFRAMES:
        console.print(f"\n[bold yellow]Fetching {tf_name} data...[/bold yellow]")
        data = fetch_commodities_data(timeframe=tf_name)

        if "error" in data:
            console.print(f"[red]Error: {data['error']}[/red]")
            continue

        commodities = data.get("commodities", {})
        if not commodities:
            console.print("[yellow]No data returned[/yellow]")
            continue

        table = Table(
            title=f"Commodities Dashboard — {tf_name}",
            show_header=True,
            header_style="bold cyan",
            title_style="bold white",
            border_style="blue",
        )
        table.add_column("Commodity", style="bold white", min_width=16)
        table.add_column("Latest Close", justify="right", min_width=12)
        table.add_column("Period Start", justify="right", min_width=12)
        table.add_column("Change %", justify="right", min_width=10)

        for name in COMMODITY_ORDER:
            series = commodities.get(name)
            if series is None or series.empty:
                table.add_row(name, "N/A", "N/A", "N/A")
                continue

            latest = series.iloc[-1]
            first = series.iloc[0]
            pct_change = ((latest - first) / first) * 100

            style = "green" if pct_change >= 0 else "red"
            table.add_row(
                name,
                format_price(latest),
                format_price(first),
                f"[{style}]{pct_change:+.2f}%[/{style}]",
            )

        console.print(table)

    console.print()


def main():
    print_terminal()


if __name__ == "__main__":
    main()
