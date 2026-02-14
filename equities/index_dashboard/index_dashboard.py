#!/usr/bin/env python3
"""
Index Dashboard

Fetches closing-price time series for 5 major equity indices from Yahoo Finance.
Displays line-chart-ready series with Daily/Weekly/Monthly toggles (GUI),
or prints summary tables in the terminal.

Terminal:
  python equities/index_dashboard/index_dashboard.py
"""

import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# -- Index definitions: display_name -> yfinance ticker ----------------------
INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Russell 2000": "^RUT",
    "STOXX 600": "^STOXX",
    "Nikkei 225": "^N225",
}

INDEX_ORDER = list(INDICES.keys())

# -- Timeframe configs: name -> yfinance (period, interval) ------------------
TIMEFRAMES = {
    "Daily": {"period": "90d", "interval": "1d"},
    "Weekly": {"period": "2y", "interval": "1wk"},
    "Monthly": {"period": "5y", "interval": "1mo"},
}


# -- Data fetching ------------------------------------------------------------

def fetch_index_data(timeframe: str = "Daily") -> dict:
    """
    Fetch closing-price time series for all tracked indices.

    Returns dict with:
        indices   – dict[index_name] -> pd.Series (Close prices)
        timeframe – str
        timestamp – datetime
        error     – str (only on failure)
    """
    tf = TIMEFRAMES.get(timeframe)
    if tf is None:
        return {"error": f"Invalid timeframe: {timeframe}"}

    tickers = list(INDICES.values())

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
    indices = {}

    for name, ticker in INDICES.items():
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

            indices[name] = series
        except Exception:
            continue

    return {
        "indices": indices,
        "timeframe": timeframe,
        "timestamp": datetime.now(),
    }


def get_data(timeframe: str = "Daily") -> dict:
    """GUI-facing entry point."""
    return fetch_index_data(timeframe=timeframe)


def format_price(value: float) -> str:
    """Format index level for clean display."""
    return f"{value:,.2f}"


# -- Terminal output ----------------------------------------------------------

def print_terminal():
    """Print Index dashboard results for all timeframes."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    header = Panel(
        "[bold white]INDEX DASHBOARD[/bold white]\n"
        f"[dim]Data from Yahoo Finance[/dim]\n"
        f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="bold blue",
        padding=(1, 2),
    )
    console.print(header)

    for tf_name in TIMEFRAMES:
        console.print(f"\n[bold yellow]Fetching {tf_name} data...[/bold yellow]")
        data = fetch_index_data(timeframe=tf_name)

        if "error" in data:
            console.print(f"[red]Error: {data['error']}[/red]")
            continue

        indices = data.get("indices", {})
        if not indices:
            console.print("[yellow]No data returned[/yellow]")
            continue

        table = Table(
            title=f"Index Dashboard - {tf_name}",
            show_header=True,
            header_style="bold cyan",
            title_style="bold white",
            border_style="blue",
        )
        table.add_column("Index", style="bold white", min_width=14)
        table.add_column("Latest Close", justify="right", min_width=12)
        table.add_column("Period Start", justify="right", min_width=12)
        table.add_column("Change %", justify="right", min_width=10)

        for name in INDEX_ORDER:
            series = indices.get(name)
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
