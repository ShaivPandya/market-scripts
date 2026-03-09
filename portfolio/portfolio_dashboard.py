#!/usr/bin/env python3
"""
Portfolio Dashboard

Fetches closing-price time series for portfolio positions from Yahoo Finance.
Reads holdings from portfolio.csv (ticker, asset class, direction).
Displays a grid of line charts with Daily/Weekly/Monthly toggles (GUI),
or prints summary tables in the terminal.

Terminal:
  python portfolio/portfolio_dashboard.py
"""

import logging
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf
from portfolio_analytics import compute_analytics
from portfolio_db import get_positions, get_positions_df

LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def _load_portfolio() -> pd.DataFrame:
    """Load portfolio positions from the database."""
    return get_positions_df()


def _build_globals(df: pd.DataFrame) -> tuple[dict, list, dict]:
    positions = {row.ticker: row.ticker for row in df.itertuples()}
    order = list(df.ticker)
    meta = {row.ticker: {"asset": row.asset, "direction": row.direction} for row in df.itertuples()}
    return positions, order, meta


_portfolio_df = _load_portfolio()
POSITIONS, POSITION_ORDER, POSITION_META = _build_globals(_portfolio_df)


def reload_portfolio() -> None:
    """Re-read positions from the database and update module-level globals."""
    global _portfolio_df, POSITIONS, POSITION_ORDER, POSITION_META
    _portfolio_df = _load_portfolio()
    POSITIONS, POSITION_ORDER, POSITION_META = _build_globals(_portfolio_df)


# -- Timeframe configs: name -> yfinance (period, interval) ──────────────────
TIMEFRAMES = {
    "This Week": {"period": "5d", "interval": "15m"},
    "Daily": {"period": "90d", "interval": "1d"},
    "Weekly": {"period": "2y", "interval": "1wk"},
    "Monthly": {"period": "5y", "interval": "1mo"},
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

    analytics = compute_analytics(positions, get_positions())

    return {
        "positions": positions,
        "metadata": POSITION_META,
        "timeframe": timeframe,
        "timestamp": datetime.now(),
        "analytics": analytics,
    }


def fetch_all_timeframes_data() -> dict:
    """Fetch portfolio data for all supported timeframes."""
    results = {}
    analytics = None
    for tf_name in TIMEFRAMES:
        data = fetch_portfolio_data(timeframe=tf_name)
        if "error" in data and data["error"]:
            return data
        results[tf_name] = data
        # Use Weekly (2y) analytics for top-level — good 52-week coverage
        if tf_name == "Weekly":
            analytics = data.get("analytics")
    return {
        "timeframes": results,
        "timestamp": datetime.now(),
        "analytics": analytics,
    }


def get_data(timeframe: str = "Daily", all_timeframes: bool = False) -> dict:
    """GUI-facing entry point."""
    if all_timeframes:
        return fetch_all_timeframes_data()
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

        analytics = data.get("analytics", {})
        per_pos = analytics.get("per_position", {})

        table = Table(
            title=f"Portfolio Dashboard — {tf_name}",
            show_header=True,
            header_style="bold cyan",
            title_style="bold white",
            border_style="blue",
        )
        table.add_column("Ticker", style="bold white", min_width=8)
        table.add_column("Dir", min_width=5)
        table.add_column("Asset", min_width=7)
        table.add_column("Price", justify="right", min_width=10)
        table.add_column("Cost Basis", justify="right", min_width=10)
        table.add_column("PnL %", justify="right", min_width=8)
        table.add_column("52w DD", justify="right", min_width=8)
        table.add_column("Wk Ret%", justify="right", min_width=8)
        table.add_column("Wk Attr%", justify="right", min_width=8)

        for ticker in POSITION_ORDER:
            series = positions.get(ticker)
            meta = POSITION_META.get(ticker, {})
            direction = meta.get("direction", "").upper()
            asset = meta.get("asset", "")
            a = per_pos.get(ticker, {})

            dir_style = "green" if direction == "LONG" else "red"

            if series is None or series.empty:
                table.add_row(ticker, f"[{dir_style}]{direction}[/{dir_style}]", asset, "N/A", "", "", "", "", "")
                continue

            latest = series.iloc[-1]
            cb = a.get("cost_basis")
            pnl = a.get("unrealized_pnl_pct")
            dd = a.get("drawdown_from_52w_pct")
            wk = a.get("weekly_return_pct")
            wk_attr = a.get("weekly_contribution_pct")

            pnl_str = f"[{'green' if pnl >= 0 else 'red'}]{pnl:+.1f}%[/]" if pnl is not None else "—"
            dd_str = f"[{'green' if dd == 0 else 'red'}]{dd:+.1f}%[/]" if dd is not None else "—"
            wk_str = f"[{'green' if wk >= 0 else 'red'}]{wk:+.1f}%[/]" if wk is not None else "—"
            attr_str = f"[{'green' if wk_attr >= 0 else 'red'}]{wk_attr:+.2f}%[/]" if wk_attr is not None else "—"

            table.add_row(
                ticker,
                f"[{dir_style}]{direction}[/{dir_style}]",
                asset,
                format_price(latest),
                format_price(cb) if cb is not None else "—",
                pnl_str,
                dd_str,
                wk_str,
                attr_str,
            )

        # Summary row
        port = analytics.get("portfolio", {})
        total_pnl = port.get("total_unrealized_pnl_pct")
        wk_port = port.get("weekly_portfolio_return_pct")
        n_prof = port.get("positions_profitable", 0)
        n_loss = port.get("positions_losing", 0)

        summary_parts = []
        if total_pnl is not None:
            c = "green" if total_pnl >= 0 else "red"
            summary_parts.append(f"Avg PnL: [{c}]{total_pnl:+.1f}%[/{c}]")
        summary_parts.append(f"W/L: {n_prof}/{n_loss}")
        if wk_port is not None:
            c = "green" if wk_port >= 0 else "red"
            summary_parts.append(f"Wk Return: [{c}]{wk_port:+.2f}%[/{c}]")

        if summary_parts:
            table.add_section()
            table.add_row("PORTFOLIO", "", "", "", "", *["" for _ in range(4)])
            console.print(table)
            console.print(f"  {'  |  '.join(summary_parts)}")
        else:
            console.print(table)

    console.print()


def main():
    print_terminal()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Starting script execution: %s", __file__)
    main()
