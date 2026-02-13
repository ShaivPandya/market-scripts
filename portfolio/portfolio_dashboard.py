#!/usr/bin/env python3
"""
Portfolio Dashboard

Fetches closing-price time series for portfolio positions from Yahoo Finance.
Reads holdings from portfolio.csv (ticker, asset class, direction).
Displays a grid of line charts with Daily/Weekly/Monthly toggles (GUI),
or prints summary tables in the terminal.

Terminal:
  python portfolio/portfolio_dashboard.py

GUI:
  Accessed via sidebar in gui/app.py
"""

import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# -- Load portfolio from CSV ──────────────────────────────────────────────────
_CSV_PATH = Path(__file__).parent / "portfolio.csv"

def _load_portfolio() -> pd.DataFrame:
    """Read portfolio.csv and return DataFrame with ticker, asset, direction."""
    return pd.read_csv(_CSV_PATH)

_portfolio_df = _load_portfolio()

POSITIONS = {row.ticker: row.ticker for row in _portfolio_df.itertuples()}
POSITION_ORDER = list(_portfolio_df.ticker)
POSITION_META = {
    row.ticker: {"asset": row.asset, "direction": row.direction}
    for row in _portfolio_df.itertuples()
}

# -- Timeframe configs: name -> yfinance (period, interval) ──────────────────
TIMEFRAMES = {
    "Daily":   {"period": "90d",  "interval": "1d"},
    "Weekly":  {"period": "2y",   "interval": "1wk"},
    "Monthly": {"period": "5y",   "interval": "1mo"},
}


# -- Data fetching ────────────────────────────────────────────────────────────

def fetch_portfolio_data(timeframe: str = "Daily") -> dict:
    """
    Fetch closing-price time series for all portfolio positions.

    Returns dict with:
        positions – dict[ticker] -> pd.Series (Close prices)
        metadata  – dict[ticker] -> {asset, direction}
        timeframe – str
        timestamp – datetime
        error     – str (only on failure)
    """
    tf = TIMEFRAMES.get(timeframe)
    if tf is None:
        return {"error": f"Invalid timeframe: {timeframe}"}

    tickers = list(POSITIONS.values())

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
    positions = {}

    for ticker in POSITION_ORDER:
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

            positions[ticker] = series
        except Exception:
            continue

    return {
        "positions": positions,
        "metadata": POSITION_META,
        "timeframe": timeframe,
        "timestamp": datetime.now(),
    }


def get_data(timeframe: str = "Daily") -> dict:
    """GUI-facing entry point."""
    return fetch_portfolio_data(timeframe=timeframe)


def format_price(value: float) -> str:
    """Format price based on magnitude for clean display."""
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:.4f}"


# -- Terminal output ──────────────────────────────────────────────────────────

def print_terminal():
    """Print Portfolio dashboard results for all timeframes."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    header = Panel(
        "[bold white]PORTFOLIO DASHBOARD[/bold white]\n"
        f"[dim]Data from Yahoo Finance[/dim]\n"
        f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="bold blue",
        padding=(1, 2),
    )
    console.print(header)

    for tf_name in TIMEFRAMES:
        console.print(f"\n[bold yellow]Fetching {tf_name} data...[/bold yellow]")
        data = fetch_portfolio_data(timeframe=tf_name)

        if "error" in data:
            console.print(f"[red]Error: {data['error']}[/red]")
            continue

        positions = data.get("positions", {})
        if not positions:
            console.print("[yellow]No data returned[/yellow]")
            continue

        table = Table(
            title=f"Portfolio Dashboard — {tf_name}",
            show_header=True,
            header_style="bold cyan",
            title_style="bold white",
            border_style="blue",
        )
        table.add_column("Ticker", style="bold white", min_width=12)
        table.add_column("Direction", min_width=8)
        table.add_column("Asset", min_width=10)
        table.add_column("Latest Close", justify="right", min_width=12)
        table.add_column("Period Start", justify="right", min_width=12)
        table.add_column("Change %", justify="right", min_width=10)

        for ticker in POSITION_ORDER:
            series = positions.get(ticker)
            meta = POSITION_META.get(ticker, {})
            direction = meta.get("direction", "").upper()
            asset = meta.get("asset", "")

            dir_style = "green" if direction == "LONG" else "red"

            if series is None or series.empty:
                table.add_row(
                    ticker,
                    f"[{dir_style}]{direction}[/{dir_style}]",
                    asset,
                    "N/A", "N/A", "N/A",
                )
                continue

            latest = series.iloc[-1]
            first = series.iloc[0]
            pct_change = ((latest - first) / first) * 100

            chg_style = "green" if pct_change >= 0 else "red"
            table.add_row(
                ticker,
                f"[{dir_style}]{direction}[/{dir_style}]",
                asset,
                format_price(latest),
                format_price(first),
                f"[{chg_style}]{pct_change:+.2f}%[/{chg_style}]",
            )

        console.print(table)

    console.print()


def main():
    print_terminal()


if __name__ == "__main__":
    main()
